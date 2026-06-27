#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import torch
import transformers
from transformers import T5Config, T5Tokenizer

from collator import Collator
from static_intersection_best_modeling import StaticBestHardOnlyLETTER
from utils import ensure_dir, load_datasets, parse_dataset_args, parse_global_args, parse_train_args, set_seed


def pcsc_schedule(epoch, max_factor=1.0, total_epochs=60):
    if epoch <= 5:
        return 0.0
    if epoch <= 15:
        return (epoch - 5) / 10.0 * max_factor
    if epoch <= 40:
        return max_factor
    return max_factor * max(total_epochs - epoch, 0) / max(total_epochs - 40, 1)


class HardOnlyTrainer(transformers.Trainer):
    def __init__(self, *args, metrics_path=None, formal_epochs=60, pcsc_max_factor=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.metrics_path = Path(metrics_path)
        self.formal_epochs = int(formal_epochs)
        self.pcsc_max_factor = float(pcsc_max_factor)
        self.sums = {}
        self.count = 0

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        epoch = min(self.formal_epochs, int(self.state.epoch or 0) + 1)
        model.set_pcsc_schedule_factor(pcsc_schedule(epoch, self.pcsc_max_factor, self.formal_epochs))
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
            logs["hard_only"] = True
            logs["soft_curriculum_enabled"] = False
            self.sums, self.count = {}, 0
        self.metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with self.metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(logs) + "\n")
        return super().log(logs, *args, **kwargs)


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
    model = StaticBestHardOnlyLETTER(
        config,
        pcsc_aux=args.pcsc_mode != "off",
        pcsc_mode=args.pcsc_mode,
        lambda_cf=args.lambda_cf,
        lambda_cfres=args.lambda_cfres,
        lambda_sem=args.lambda_sem,
        lambda_semres=args.lambda_semres,
    )
    model.set_hyper(args.temperature)
    model.resize_token_embeddings(len(tokenizer))
    model.configure_items(
        tokenizer, args.index, args.item_order, args.cf_emb, args.sem_emb,
        args.cf_res, args.sem_res_raw,
    )
    model.to(torch.device("cuda", 0))
    trainer = HardOnlyTrainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=valid_data,
        args=transformers.TrainingArguments(
            seed=args.seed,
            per_device_train_batch_size=args.per_device_batch_size,
            per_device_eval_batch_size=args.per_device_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
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
            save_total_limit=1,
            report_to=[],
            remove_unused_columns=False,
        ),
        tokenizer=tokenizer,
        data_collator=collator,
        metrics_path=args.training_metrics,
        formal_epochs=args.epochs,
        pcsc_max_factor=args.pcsc_max_factor,
    )
    model.config.use_cache = False
    trainer.train()
    trainer.save_state()
    trainer.save_model(args.output_dir)
    Path(args.run_summary).write_text(json.dumps({
        "mode": "static_best_hard_only",
        "pcsc_mode": args.pcsc_mode,
        "seed": args.seed,
        "formal_epochs": args.epochs,
        "completed_epochs": trainer.state.epoch,
        "global_step": trainer.state.global_step,
        "hard_only": True,
        "soft_curriculum": False,
        "tokenizer_checkpoint_loaded": False,
        "rqvae_checkpoint_loaded": False,
        "pcsc_max_factor": args.pcsc_max_factor,
        "lambda_cf": args.lambda_cf,
        "lambda_cfres": args.lambda_cfres,
        "lambda_sem": args.lambda_sem,
        "lambda_semres": args.lambda_semres,
    }, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = parse_global_args(parser)
    parser = parse_train_args(parser)
    parser = parse_dataset_args(parser)
    parser.add_argument("--index", required=True)
    parser.add_argument("--item_order", required=True)
    parser.add_argument("--cf_emb", required=True)
    parser.add_argument("--sem_emb", required=True)
    parser.add_argument("--cf_res", required=True)
    parser.add_argument("--sem_res_raw", required=True)
    parser.add_argument("--pcsc_mode", choices=["off", "original", "swapped_c1c2"], required=True)
    parser.add_argument("--pcsc_max_factor", type=float, default=1.0)
    parser.add_argument("--lambda_cf", type=float, default=1.0)
    parser.add_argument("--lambda_cfres", type=float, default=1.0)
    parser.add_argument("--lambda_sem", type=float, default=1.0)
    parser.add_argument("--lambda_semres", type=float, default=1.0)
    parser.add_argument("--training_metrics", required=True)
    parser.add_argument("--run_summary", required=True)
    main(parser.parse_args())
