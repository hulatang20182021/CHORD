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
from transformers import T5Config, T5Tokenizer

PROJECT = Path("/home/huangxin/llmNrec/Letter/LETTER-master/component_relation_sid/rqvae_supervision/res/pls_sd128_dpos_pcsc_pipeline")
SCRIPT_DIR = PROJECT / "scripts"
THIS_DIR = SCRIPT_DIR / "pls_consistent_residual"
for p in [str(THIS_DIR), str(SCRIPT_DIR), str(Path("/home/huangxin/llmNrec/Letter/LETTER-master/LETTER-TIGER"))]:
    if p not in sys.path:
        sys.path.insert(0, p)

from collator import Collator
from modeling_pls_consistent_letter import PLSConsistentLETTER
from static_intersection_downstream_finetune import HardOnlyTrainer, pcsc_schedule
from utils import ensure_dir, load_datasets, parse_dataset_args, parse_global_args, parse_train_args, set_seed


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
    model = PLSConsistentLETTER(
        config,
        order=args.order,
        pcsc_aux=args.pcsc_aux,
        pcsc_h12_mode=args.pcsc_h12_mode,
        lambda_shared=args.lambda_shared,
        lambda_level2=args.lambda_level2,
        lambda_level3=args.lambda_level3,
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
    )
    model.set_curriculum(1.0, force_soft=False)
    model.set_pcsc_schedule_factor(0.0)
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
            logging_strategy="epoch",
            logging_steps=args.logging_step,
            optim=args.optim,
            eval_strategy="epoch",
            save_strategy="epoch",
            output_dir=args.output_dir,
            save_total_limit=args.save_total_limit if args.save_total_limit > 0 else None,
            load_best_model_at_end=args.load_best_model_at_end,
            metric_for_best_model=(args.metric_for_best_model if args.load_best_model_at_end else None),
            greater_is_better=(args.greater_is_better if args.load_best_model_at_end else None),
            eval_delay=1,
            report_to=[],
            run_name="pls_consistent_residual",
            remove_unused_columns=False,
            disable_tqdm=True,
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
    if args.run_summary:
        Path(args.run_summary).write_text(
            json.dumps(
                {
                    "order": args.order,
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
    parser.add_argument("--pcsc_max_factor", type=float, default=1.0)
    parser.add_argument("--pcsc_schedule_type", choices=["warmup_hold", "warmup_hold_decay"], default="warmup_hold_decay")
    parser.add_argument("--pcsc_h12_mode", choices=["mean", "h2"], default="mean")
    parser.add_argument("--lambda_shared", type=float, default=1.0)
    parser.add_argument("--lambda_level2", type=float, default=1.0)
    parser.add_argument("--lambda_level3", type=float, default=1.0)
    parser.add_argument("--training_metrics", required=True)
    parser.add_argument("--run_summary", default="")
    parser.add_argument("--load_best_model_at_end", action="store_true")
    parser.add_argument("--metric_for_best_model", default="eval_loss")
    parser.add_argument("--greater_is_better", type=lambda x: str(x).lower() in {"1", "true", "yes", "y"}, default=False)
    parser.add_argument("--save_total_limit", type=int, default=5)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
