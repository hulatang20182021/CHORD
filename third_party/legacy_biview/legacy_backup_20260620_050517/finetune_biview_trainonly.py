#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import torch
import transformers
from torch.utils.data import DataLoader
from transformers import T5Config, T5Tokenizer

from collator import Collator
from modeling_matched_curriculum_letter import MatchedCurriculumLETTER
from utils import (
    ensure_dir,
    load_datasets,
    parse_dataset_args,
    parse_global_args,
    parse_train_args,
    set_seed,
)


def schedule(epoch):
    if epoch <= 5:
        return 0.1 + (epoch - 1) * 0.3 / 4
    if epoch <= 15:
        return 0.4 + (epoch - 6) * 0.5 / 9
    return 1.0


def pcsc_schedule(
    epoch, max_factor=1.0, schedule_type="warmup_hold",
    total_epochs=60,
):
    if epoch <= 5:
        value = 0.0
    elif epoch <= 15:
        value = (epoch - 5) / 10.0 * max_factor
    else:
        value = max_factor

    if schedule_type == "warmup_hold":
        return value
    if schedule_type == "warmup_hold_decay":
        if epoch <= 40:
            return value
        decay_span = max(total_epochs - 40, 1)
        remain = max(total_epochs - epoch, 0) / decay_span
        return max_factor * remain
    raise ValueError(f"Unknown pcsc_schedule_type: {schedule_type}")


class CurriculumTrainer(transformers.Trainer):
    def __init__(
        self, *args, metrics_path=None, formal_epochs=60,
        pcsc_max_factor=1.0, pcsc_schedule_type="warmup_hold",
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.metrics_path = Path(metrics_path)
        self.formal_epochs = formal_epochs
        self.pcsc_max_factor = float(pcsc_max_factor)
        self.pcsc_schedule_type = pcsc_schedule_type
        self.sums, self.count = {}, 0

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        epoch = min(self.formal_epochs, int(self.state.epoch or 0) + 1)
        model.set_curriculum(schedule(epoch), force_soft=False)
        model.set_pcsc_schedule_factor(
            pcsc_schedule(
                epoch,
                self.pcsc_max_factor,
                self.pcsc_schedule_type,
                self.formal_epochs,
            )
        )
        outputs = model(**inputs)
        for key, value in model.last_curriculum_metrics.items():
            if not isinstance(value, bool):
                self.sums[key] = self.sums.get(key, 0.0) + float(value)
        self.count += 1
        return (outputs.loss, outputs) if return_outputs else outputs.loss

    def log(self, logs, *args, **kwargs):
        if self.count:
            for key, value in self.sums.items():
                logs[key] = value / self.count
            logs["curriculum_nan_seen"] = bool(self.model.last_curriculum_metrics.get("nan", False))
            self.sums, self.count = {}, 0
        self.metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with self.metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(logs) + "\n")
        return super().log(logs, *args, **kwargs)


def projector_warmup(
    model, train_data, collator, device, batch_size, output, seed
):
    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in model.projector_parameters():
        parameter.requires_grad = True
    optimizer = torch.optim.AdamW(model.projector_parameters(), lr=5e-4, weight_decay=1e-4)
    loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=0,
        generator=torch.Generator().manual_seed(seed),
    )
    model.train()
    model.set_curriculum(0.0, force_soft=True)
    model.set_pcsc_schedule_factor(0.0)
    total, seen = 0.0, 0
    metric_sums = {}
    for batch in loader:
        batch = {key: value.to(device) for key, value in batch.items()}
        optimizer.zero_grad()
        outputs = model(**batch)
        outputs.loss.backward()
        optimizer.step()
        total += float(outputs.loss.detach()) * len(batch["labels"])
        seen += len(batch["labels"])
        for key, value in model.last_curriculum_metrics.items():
            if not isinstance(value, bool):
                metric_sums[key] = metric_sums.get(key, 0.0) + float(value)
    for parameter in model.parameters():
        parameter.requires_grad = True
    result = {
        "warmup_epochs": 1,
        "sid_ce_loss": total / seen,
        **{key: value / len(loader) for key, value in metric_sums.items()},
        "nan": bool(model.last_curriculum_metrics.get("nan", False)),
    }
    Path(output).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


def main(args):
    set_seed(args.seed)
    ensure_dir(args.output_dir)
    config = T5Config.from_pretrained(args.base_model)
    tokenizer = T5Tokenizer.from_pretrained(args.base_model, model_max_length=512)
    train_data, valid_data = load_datasets(args)
    tokenizer.add_tokens(train_data.datasets[0].get_new_tokens())
    config.vocab_size = len(tokenizer)
    tokenizer.save_pretrained(args.output_dir)
    config.save_pretrained(args.output_dir)
    collator = Collator(args, tokenizer)
    model = MatchedCurriculumLETTER(
        config,
        args.mode,
        pcsc_aux=args.pcsc_aux,
        pcsc_h12_mode=args.pcsc_h12_mode,
        lambda_cf=args.lambda_cf,
        lambda_cfres=args.lambda_cfres,
        lambda_base=args.lambda_base,
        lambda_res=args.lambda_res,
        lambda_comp=args.lambda_comp,
    )
    model.set_hyper(args.temperature)
    model.resize_token_embeddings(len(tokenizer))
    model.configure_items(
        tokenizer,
        args.index,
        args.item_order,
        args.cf_emb,
        args.sem_emb,
        rqvae_checkpoint_path=args.rqvae_checkpoint,
        cf_res_path=args.cf_res,
        sem_base_path=args.sem_base,
        sem_res_raw_path=args.sem_res_raw,
    )
    device = torch.device("cuda", 0)
    model.to(device)
    projector_warmup(
        model,
        train_data,
        collator,
        device,
        args.per_device_batch_size,
        args.warmup_summary,
        args.seed,
    )

    effective_save_total_limit = args.save_total_limit
    if args.load_best_model_at_end and effective_save_total_limit <= 0:
        effective_save_total_limit = 5
    elif effective_save_total_limit <= 0:
        effective_save_total_limit = None

    trainer = CurriculumTrainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=valid_data,
        args=transformers.TrainingArguments(
            seed=args.seed,
            per_device_train_batch_size=args.per_device_batch_size,
            per_device_eval_batch_size=args.per_device_batch_size,
            gradient_accumulation_steps=1,
            warmup_ratio=args.warmup_ratio,
            num_train_epochs=args.epochs,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            lr_scheduler_type=args.lr_scheduler_type,
            logging_steps=args.logging_step,
            optim=args.optim,
            eval_strategy="epoch",
            save_strategy="epoch",
            output_dir=args.output_dir,
            save_total_limit=effective_save_total_limit,
            load_best_model_at_end=args.load_best_model_at_end,
            metric_for_best_model=(
                args.metric_for_best_model if args.load_best_model_at_end else None
            ),
            greater_is_better=(
                args.greater_is_better if args.load_best_model_at_end else None
            ),
            eval_delay=1,
            report_to=[],
            remove_unused_columns=False,
        ),
        tokenizer=tokenizer,
        data_collator=collator,
        metrics_path=args.training_metrics,
        formal_epochs=args.epochs,
        pcsc_max_factor=args.pcsc_max_factor,
        pcsc_schedule_type=args.pcsc_schedule_type,
    )
    model.config.use_cache = False
    trainer.train()
    trainer.save_state()
    trainer.save_model(args.output_dir)
    pcsc_schedule_descriptions = {
        "warmup_hold": (
            "0 through epoch 5; linear to pcsc_max_factor by epoch 15; "
            "hold pcsc_max_factor through the final epoch"
        ),
        "warmup_hold_decay": (
            "0 through epoch 5; linear to pcsc_max_factor by epoch 15; "
            "hold through epoch 40; decay to 0 by the final epoch"
        ),
    }
    Path(args.run_summary).write_text(
        json.dumps(
            {
                "mode": args.mode,
                "seed": args.seed,
                "formal_epochs": args.epochs,
                "completed_epochs": trainer.state.epoch,
                "global_step": trainer.state.global_step,
                "projector_warmup_epochs": 1,
                "best_checkpoint": trainer.state.best_model_checkpoint,
                "best_metric": trainer.state.best_metric,
                "early_stopping": False,
                "load_best_model_at_end": args.load_best_model_at_end,
                "metric_for_best_model": args.metric_for_best_model if args.load_best_model_at_end else None,
                "greater_is_better": args.greater_is_better if args.load_best_model_at_end else None,
                "save_total_limit": effective_save_total_limit,
                "schedule": "p_sid 0.1->0.4 epochs 1-5; 0.4->0.9 epochs 6-15; 1.0 epochs 16+",
                "pcsc_aux": args.pcsc_aux,
                "pcsc_max_factor": args.pcsc_max_factor,
                "pcsc_schedule_type": args.pcsc_schedule_type,
                "lambda_cf": args.lambda_cf,
                "lambda_cfres": args.lambda_cfres,
                "lambda_base": args.lambda_base,
                "lambda_res": args.lambda_res,
                "lambda_comp": args.lambda_comp,
                "pcsc_schedule": pcsc_schedule_descriptions[
                    args.pcsc_schedule_type
                ],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = parse_global_args(parser)
    parser = parse_train_args(parser)
    parser = parse_dataset_args(parser)
    parser.add_argument("--mode", choices=["zcf", "layered_rq", "layered_rq_pcsc"], required=True)
    parser.add_argument("--index", required=True)
    parser.add_argument("--item_order", required=True)
    parser.add_argument("--cf_emb", required=True)
    parser.add_argument("--sem_emb", required=True)
    parser.add_argument("--rqvae_checkpoint")
    parser.add_argument("--cf_res")
    parser.add_argument("--sem_base")
    parser.add_argument("--sem_res_raw")
    parser.add_argument("--pcsc_aux", action="store_true")
    parser.add_argument("--pcsc_max_factor", type=float, default=1.0)
    parser.add_argument(
        "--pcsc_schedule_type",
        choices=["warmup_hold", "warmup_hold_decay"],
        default="warmup_hold",
    )
    parser.add_argument("--pcsc_h12_mode", choices=["mean", "h2"], default="mean")
    parser.add_argument("--lambda_cf", type=float, default=0.003)
    parser.add_argument("--lambda_cfres", type=float, default=0.001)
    parser.add_argument("--lambda_base", type=float, default=0.002)
    parser.add_argument("--lambda_res", type=float, default=0.001)
    parser.add_argument("--lambda_comp", type=float, default=0.003)
    parser.add_argument("--warmup_summary", required=True)
    parser.add_argument("--training_metrics", required=True)
    parser.add_argument("--run_summary", required=True)
    parser.add_argument("--load_best_model_at_end", action="store_true")
    parser.add_argument("--metric_for_best_model", default="eval_loss")
    parser.add_argument("--greater_is_better", type=lambda x: str(x).lower() in {"1", "true", "yes", "y"}, default=False)
    parser.add_argument("--save_total_limit", type=int, default=5, help="How many epoch checkpoints to keep when best checkpoint is enabled")
    main(parser.parse_args())
