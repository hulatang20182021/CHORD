#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
import transformers
from transformers import TrainerCallback, T5Config, T5Tokenizer

PROJECT = Path(os.environ.get("PROJECT", Path(__file__).resolve().parents[3]))
SCRIPT_DIR = Path(os.environ.get("FORMAL_SCRIPT_DIR", Path(__file__).resolve().parent))
THIS_DIR = Path(__file__).resolve().parent
TIGER = Path(os.environ.get("TIGER", "/home/huangxin/llmNrec/LETTER-master/LETTER-TIGER"))
for p in [str(THIS_DIR), str(SCRIPT_DIR), str(TIGER)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from collator import Collator
from modeling_chord_letter import CHORDLETTER
from static_intersection_downstream_finetune import HardOnlyTrainer, pcsc_schedule
from utils import ensure_dir, load_datasets, parse_dataset_args, parse_global_args, parse_train_args, set_seed


def str2bool(value):
    if isinstance(value, bool):
        return value
    lowered = str(value).strip().lower()
    if lowered in {"1", "true", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected boolean value, got {value!r}")


def optional_int_env(name):
    value = os.environ.get(name)
    return int(value) if value not in {None, ""} else None


def resolve_precision(args):
    precision = args.precision or os.environ.get("PRECISION", "fp32")
    precision = precision.lower()
    if precision == "fb32":
        print("[runtime][warning] PRECISION=fb32 is not valid; using fp32", flush=True)
        precision = "fp32"
    if precision not in {"fp32", "tf32", "bf16", "fp16"}:
        raise ValueError(f"Unsupported precision={precision!r}")
    args.precision = precision
    torch.backends.cuda.matmul.allow_tf32 = precision == "tf32"
    if precision == "tf32":
        torch.set_float32_matmul_precision("high")
    return precision


class EpochIntervalCallback(TrainerCallback):
    def __init__(self, eval_every_n_epochs=1, save_every_n_epochs=1):
        self.eval_every = max(1, int(eval_every_n_epochs))
        self.save_every = max(1, int(save_every_n_epochs))

    @staticmethod
    def _epoch_index(state):
        return max(1, int(float(state.epoch or 0.0) + 0.999999))

    def on_epoch_end(self, args, state, control, **kwargs):
        epoch = self._epoch_index(state)
        if self.eval_every > 1 and epoch % self.eval_every:
            control.should_evaluate = False
        if self.save_every > 1 and epoch % self.save_every:
            control.should_save = False
        return control


def configure_runtime_args(args):
    resolve_precision(args)
    if args.local_fast_mode:
        if args.eval_every_n_epochs is None:
            args.eval_every_n_epochs = 1
        if args.save_every_n_epochs is None:
            args.save_every_n_epochs = 1
        if args.save_total_limit is None:
            args.save_total_limit = 5
        print("[runtime] local_fast_mode enabled: eval/save every 1 epoch", flush=True)

    if args.eval_every_n_epochs is None:
        args.eval_every_n_epochs = 1
    if args.save_every_n_epochs is None:
        args.save_every_n_epochs = 1
    if args.save_total_limit is None:
        args.save_total_limit = 5

    if args.dataloader_num_workers == 0 and args.dataloader_persistent_workers:
        print("[runtime][warning] dataloader_persistent_workers=True requires dataloader_num_workers > 0; forcing False", flush=True)
        args.dataloader_persistent_workers = False


def print_startup_log(args):
    cuda_available = torch.cuda.is_available()
    gpu_name = "none"
    gpu_mem = "none"
    if cuda_available:
        props = torch.cuda.get_device_properties(0)
        gpu_name = props.name
        gpu_mem = f"{props.total_memory / 1024**3:.2f} GiB"
    print("[runtime] CHORD training configuration", flush=True)
    for key, value in [
        ("torch_version", torch.__version__),
        ("torch_cuda_version", torch.version.cuda),
        ("gpu_name", gpu_name),
        ("gpu_total_memory", gpu_mem),
        ("precision", args.precision),
        ("fp16", args.precision == "fp16"),
        ("bf16", args.precision == "bf16"),
        ("tf32", args.precision == "tf32"),
        ("torch.backends.cuda.matmul.allow_tf32", torch.backends.cuda.matmul.allow_tf32),
        ("torch.backends.cudnn.allow_tf32", torch.backends.cudnn.allow_tf32),
        ("torch.get_float32_matmul_precision()", torch.get_float32_matmul_precision()),
        ("train_batch_size", args.per_device_batch_size),
        ("eval_batch_size", args.per_device_batch_size),
        ("gradient_accumulation_steps", args.gradient_accumulation_steps),
        ("dataloader_num_workers", args.dataloader_num_workers),
        ("dataloader_pin_memory", args.dataloader_pin_memory),
        ("dataloader_persistent_workers", args.dataloader_persistent_workers),
        ("local_fast_mode", args.local_fast_mode),
        ("local_5060_bf16_fast", args.local_5060_bf16_fast),
        ("eval_every_n_epochs", args.eval_every_n_epochs),
        ("save_every_n_epochs", args.save_every_n_epochs),
        ("save_total_limit", args.save_total_limit),
        ("print_every", args.print_every),
        ("seed", args.seed),
    ]:
        print(f"[runtime] {key}={value}", flush=True)


def main(args):
    set_seed(args.seed)
    configure_runtime_args(args)
    print_startup_log(args)
    ensure_dir(args.output_dir)
    config = T5Config.from_pretrained(args.base_model)
    tokenizer = T5Tokenizer.from_pretrained(args.base_model, model_max_length=512)
    train_data, valid_data = load_datasets(args)
    tokenizer.add_tokens(train_data.datasets[0].get_new_tokens())
    config.vocab_size = len(tokenizer)
    tokenizer.save_pretrained(args.output_dir)
    config.save_pretrained(args.output_dir)
    collator = Collator(args, tokenizer)
    model = CHORDLETTER(
        config,
        order=args.order,
        pcsc_aux=args.pcsc_aux,
        pcsc_mode=args.pcsc_mode,
        pcsc_h12_mode=args.pcsc_h12_mode,
        lambda_shared=args.lambda_shared,
        lambda_level2=args.lambda_level2,
        lambda_level3=args.lambda_level3,
        lambda_cf=args.lambda_cf,
        lambda_cfres=args.lambda_cfres,
        lambda_base=args.lambda_base,
        lambda_res=args.lambda_res,
        lambda_comp=args.lambda_comp,
    )
    model.set_hyper(args.temperature)
    model.resize_token_embeddings(len(tokenizer))
    model.configure_pls_items(
        tokenizer,
        args.index,
        args.item_order,
        args.shared_emb,
        args.level2_emb,
        args.level3_emb,
        cf_path=args.cf_emb,
        sem_path=args.sem_emb,
        cf_res_path=args.cf_res,
        sem_base_path=args.sem_base,
        sem_res_raw_path=args.sem_res_raw,
    )
    model.set_curriculum(1.0, force_soft=False)
    model.set_pcsc_schedule_factor(0.0)
    model.to(torch.device("cuda", 0))

    callbacks = []
    if args.eval_every_n_epochs > 1 or args.save_every_n_epochs > 1:
        callbacks.append(EpochIntervalCallback(args.eval_every_n_epochs, args.save_every_n_epochs))

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
            logging_strategy="epoch",
            logging_steps=args.logging_step,
            optim=args.optim,
            eval_strategy="epoch",
            save_strategy="epoch",
            output_dir=args.output_dir,
            save_total_limit=args.save_total_limit if args.save_total_limit > 0 else None,
            fp16=args.precision == "fp16",
            bf16=args.precision == "bf16",
            tf32=(args.precision == "tf32"),
            dataloader_num_workers=args.dataloader_num_workers,
            dataloader_pin_memory=args.dataloader_pin_memory,
            dataloader_persistent_workers=args.dataloader_persistent_workers,
            load_best_model_at_end=args.load_best_model_at_end,
            metric_for_best_model=(args.metric_for_best_model if args.load_best_model_at_end else None),
            greater_is_better=(args.greater_is_better if args.load_best_model_at_end else None),
            eval_delay=1,
            report_to=[],
            run_name="CHORD",
            remove_unused_columns=False,
            disable_tqdm=True,
        ),
        tokenizer=tokenizer,
        data_collator=collator,
        metrics_path=args.training_metrics,
        formal_epochs=args.epochs,
        pcsc_max_factor=args.pcsc_max_factor,
        pcsc_schedule_type=args.pcsc_schedule_type,
        callbacks=callbacks,
    )
    model.config.use_cache = False
    trainer.train()
    trainer.save_state()
    trainer.save_model(args.output_dir)
    if args.run_summary:
        Path(args.run_summary).write_text(
            json.dumps(
                {
                    "order": args.order,
                    "pcsc_mode": args.pcsc_mode,
                    "pcsc_mapping": {
                        "shared_hidden": "shared_repr",
                        "level2_hidden": args.level2_name,
                        "level3_hidden": args.level3_name,
                    },
                    "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )


def parse_args():
    parser = argparse.ArgumentParser()
    parse_global_args(parser)
    parse_train_args(parser)
    parse_dataset_args(parser)
    parser.add_argument("--index", required=True)
    parser.add_argument("--item_order", required=True)
    parser.add_argument("--shared_emb", required=True)
    parser.add_argument("--level2_emb", required=True)
    parser.add_argument("--level3_emb", required=True)
    parser.add_argument("--level2_name", required=True)
    parser.add_argument("--level3_name", required=True)
    parser.add_argument("--order", choices=["sem_first", "cf_first"], required=True)
    parser.add_argument("--pcsc_aux", action="store_true")
    parser.add_argument("--pcsc_mode", choices=["simple3", "legacy5"], default=os.environ.get("PCSC_MODE", "simple3"))
    parser.add_argument("--cf_emb", default="")
    parser.add_argument("--sem_emb", default="")
    parser.add_argument("--cf_res", default="")
    parser.add_argument("--sem_base", default="")
    parser.add_argument("--sem_res_raw", default="")
    parser.add_argument("--pcsc_max_factor", type=float, default=1.0)
    parser.add_argument("--pcsc_schedule_type", choices=["warmup_hold", "warmup_hold_decay"], default="warmup_hold_decay")
    parser.add_argument("--pcsc_h12_mode", choices=["mean", "h2"], default="mean")
    parser.add_argument("--lambda_shared", type=float, default=1.0)
    parser.add_argument("--lambda_level2", type=float, default=1.0)
    parser.add_argument("--lambda_level3", type=float, default=1.0)
    parser.add_argument("--lambda_cf", type=float, default=float(os.environ.get("LAMBDA_CF", "1.0")))
    parser.add_argument("--lambda_cfres", type=float, default=float(os.environ.get("LAMBDA_CFRES", "1.0")))
    parser.add_argument("--lambda_base", type=float, default=float(os.environ.get("LAMBDA_BASE", "1.0")))
    parser.add_argument("--lambda_res", type=float, default=float(os.environ.get("LAMBDA_RES", "1.0")))
    parser.add_argument("--lambda_comp", type=float, default=float(os.environ.get("LAMBDA_COMP", "1.0")))
    parser.add_argument("--training_metrics", required=True)
    parser.add_argument("--run_summary", default="")
    parser.add_argument("--load_best_model_at_end", action="store_true")
    parser.add_argument("--metric_for_best_model", default="eval_loss")
    parser.add_argument("--greater_is_better", type=lambda x: str(x).lower() in {"1", "true", "yes", "y"}, default=False)
    parser.add_argument("--save_total_limit", type=int, default=optional_int_env("SAVE_TOTAL_LIMIT"))
    parser.add_argument("--precision", choices=["fp32", "fb32", "tf32", "bf16", "fp16"], default=None)
    parser.add_argument("--dataloader_num_workers", type=int, default=int(os.environ.get("DATALOADER_NUM_WORKERS", "0")))
    parser.add_argument("--dataloader_pin_memory", type=str2bool, default=str2bool(os.environ.get("DATALOADER_PIN_MEMORY", "true")))
    parser.add_argument("--dataloader_persistent_workers", type=str2bool, default=str2bool(os.environ.get("DATALOADER_PERSISTENT_WORKERS", "false")))
    parser.add_argument("--eval_every_n_epochs", type=int, default=optional_int_env("EVAL_EVERY_N_EPOCHS"))
    parser.add_argument("--save_every_n_epochs", type=int, default=optional_int_env("SAVE_EVERY_N_EPOCHS"))
    parser.add_argument("--local_fast_mode", type=str2bool, default=str2bool(os.environ.get("LOCAL_FAST_MODE", "false")))
    parser.add_argument("--local_5060_bf16_fast", type=str2bool, default=str2bool(os.environ.get("LOCAL_5060_BF16_FAST", "false")))
    parser.add_argument("--print_every", type=int, default=int(os.environ.get("PRINT_EVERY", "1")))
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
