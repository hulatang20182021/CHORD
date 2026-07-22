#!/usr/bin/env python3
import argparse
import json
import os
import time
from pathlib import Path

import torch
import transformers
from transformers import T5Config, T5Tokenizer

from collator import Collator
from modeling_matched_curriculum_strict_symmetric_shared_anchor import MatchedCurriculumLETTER
from utils import (
    ensure_dir,
    load_datasets,
    parse_dataset_args,
    parse_global_args,
    parse_train_args,
    set_seed,
)


def pcsc_schedule(epoch, max_factor=1.0, schedule_type="warmup_hold_decay", total_epochs=60):
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




class StopAtEpochCallback(transformers.TrainerCallback):
    def __init__(self, stop_after_epoch=0):
        self.stop_after_epoch = float(stop_after_epoch or 0)

    def on_epoch_end(self, args, state, control, **kwargs):
        if self.stop_after_epoch > 0 and state.epoch is not None and float(state.epoch) >= self.stop_after_epoch - 1e-6:
            control.should_training_stop = True
        return control


class SelectiveEpochSaveCallback(transformers.TrainerCallback):
    def __init__(self, save_epochs=""):
        self.save_epochs = {
            int(value.strip())
            for value in str(save_epochs or "").split(",")
            if value.strip()
        }
        if any(epoch <= 0 for epoch in self.save_epochs):
            raise ValueError("--save_epochs must contain positive epoch numbers")

    def on_epoch_end(self, args, state, control, **kwargs):
        epoch = int(round(float(state.epoch or 0)))
        control.should_save = epoch in self.save_epochs
        return control


def enforce_worker_budget(worker_count: int) -> None:
    memory_limit_paths = (
        Path("/sys/fs/cgroup/memory.max"),
        Path("/sys/fs/cgroup/memory/memory.limit_in_bytes"),
    )
    limit = None
    for path in memory_limit_paths:
        if not path.is_file():
            continue
        value = path.read_text().strip()
        if value.isdigit():
            limit = int(value)
            break
    if limit is not None and limit <= 32 * 1024**3 and worker_count > 12:
        raise ValueError(
            f"Refusing dataloader_num_workers={worker_count} under a {limit / 1024**3:.1f} GiB "
            "cgroup memory limit; use at most 12 workers."
        )


class HardOnlyTrainer(transformers.Trainer):
    def __init__(
        self, *args, metrics_path=None, formal_epochs=60,
        pcsc_max_factor=1.0, pcsc_schedule_type="warmup_hold_decay", **kwargs
    ):
        super().__init__(*args, **kwargs)
        self._persistent_eval_dataloaders = {}
        self.metrics_path = Path(metrics_path)
        self.formal_epochs = int(formal_epochs)
        self.pcsc_max_factor = float(pcsc_max_factor)
        self.pcsc_schedule_type = pcsc_schedule_type
        self.sums = {}
        self.count = 0
        self.start_time = time.time()
        self.last_train_epoch_printed = 0

    def get_eval_dataloader(self, eval_dataset=None):
        dataset = self.eval_dataset if eval_dataset is None else eval_dataset
        if not self.args.dataloader_persistent_workers:
            return super().get_eval_dataloader(eval_dataset)
        key = id(dataset)
        if key not in self._persistent_eval_dataloaders:
            self._persistent_eval_dataloaders[key] = super().get_eval_dataloader(eval_dataset)
        return self._persistent_eval_dataloaders[key]

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        epoch = min(self.formal_epochs, int(self.state.epoch or 0) + 1)
        model.set_curriculum(1.0, force_soft=False)
        model.set_pcsc_schedule_factor(
            pcsc_schedule(epoch, self.pcsc_max_factor, self.pcsc_schedule_type, self.formal_epochs)
        )
        outputs = model(**inputs)
        for key, value in model.last_curriculum_metrics.items():
            if not isinstance(value, bool):
                self.sums[key] = self.sums.get(key, 0.0) + float(value)
        self.count += 1
        return (outputs.loss, outputs) if return_outputs else outputs.loss

    def _fmt_seconds(self, seconds):
        seconds = max(0.0, float(seconds))
        if seconds < 60:
            return f"{seconds:.0f}s"
        minutes = seconds / 60.0
        if minutes < 60:
            return f"{minutes:.1f}m"
        return f"{minutes / 60.0:.1f}h"

    def log(self, logs, *args, **kwargs):
        if self.count:
            for key, value in self.sums.items():
                logs[key] = value / self.count
            logs["hard_only"] = True
            logs["soft_curriculum_enabled"] = False
            logs["curriculum_nan_seen"] = bool(self.model.last_curriculum_metrics.get("nan", False))
            self.sums, self.count = {}, 0
        self.metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with self.metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(logs) + "\n")

        epoch_float = float(logs.get("epoch") or self.state.epoch or 0.0)
        epoch_idx = min(self.formal_epochs, max(1, int(epoch_float + 0.999999)))
        step = int(getattr(self.state, "global_step", 0) or 0)
        total_steps = int(getattr(self.state, "max_steps", 0) or 0)
        elapsed = time.time() - self.start_time
        avg_epoch = elapsed / max(epoch_float, 1e-9)
        eta = avg_epoch * max(self.formal_epochs - epoch_float, 0.0)

        if "loss" in logs and epoch_idx != self.last_train_epoch_printed:
            self.last_train_epoch_printed = epoch_idx
            parts = [
                f"[train] epoch {epoch_idx}/{self.formal_epochs}",
                f"step={step}/{total_steps}" if total_steps else f"step={step}",
                f"loss={float(logs['loss']):.6f}",
            ]
            if "learning_rate" in logs:
                parts.append(f"lr={float(logs['learning_rate']):.3e}")
            if "grad_norm" in logs:
                parts.append(f"grad={float(logs['grad_norm']):.4f}")
            parts.extend([
                f"elapsed={self._fmt_seconds(elapsed)}",
                f"avg_epoch={self._fmt_seconds(avg_epoch)}",
                f"eta={self._fmt_seconds(eta)}",
            ])
            print(" | ".join(parts), flush=True)
        elif "eval_loss" in logs:
            parts = [
                f"[eval] epoch {epoch_idx}/{self.formal_epochs}",
                f"step={step}/{total_steps}" if total_steps else f"step={step}",
                f"eval_loss={float(logs['eval_loss']):.6f}",
            ]
            if "eval_runtime" in logs:
                parts.append(f"eval_runtime={self._fmt_seconds(float(logs['eval_runtime']))}")
            parts.extend([
                f"elapsed={self._fmt_seconds(elapsed)}",
                f"avg_epoch={self._fmt_seconds(avg_epoch)}",
                f"eta={self._fmt_seconds(eta)}",
            ])
            print(" | ".join(parts), flush=True)
        try:
            import wandb
            if wandb.run is not None:
                wandb.log(dict(logs), step=step)
        except Exception:
            pass

        output = dict(logs)
        output["step"] = step
        self.state.log_history.append(output)
        return self.control



def main(args):
    enforce_worker_budget(args.dataloader_num_workers)
    if args.disable_train_eval and args.load_best_model_at_end:
        raise ValueError("--disable_train_eval is incompatible with --load_best_model_at_end")
    data_seed = args.data_seed if args.data_seed >= 0 else args.seed
    if args.full_determinism:
        os.environ.setdefault("PYTHONHASHSEED", str(args.seed))
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        transformers.enable_full_determinism(args.seed, warn_only=args.determinism_warn_only)
    else:
        set_seed(args.seed)
    ensure_dir(args.output_dir)
    contract = {
        "version": MatchedCurriculumLETTER.PCSC_CONTRACT_VERSION,
        "alignment": args.pcsc_alignment,
        "sid_component_order": args.sid_component_order,
        "h12_mode": args.pcsc_h12_mode,
        "contract": (
            MatchedCurriculumLETTER.POSITIONAL_PCSC_CONTRACT
            if args.pcsc_alignment == "positional"
            else "component_mapped_legacy5"
        ),
    }
    if args.resume_from_checkpoint and args.pcsc_alignment == "positional":
        resume_path = Path(args.resume_from_checkpoint).resolve()
        candidates = [resume_path / "pcsc_contract.json", resume_path.parent / "pcsc_contract.json"]
        existing = next((path for path in candidates if path.is_file()), None)
        if existing is None and not args.allow_unversioned_legacy_resume:
            raise RuntimeError(
                "Refusing to resume cross-view PCSC from an unversioned checkpoint. "
                "Use an explicitly versioned checkpoint for formal runs."
            )
        previous = json.loads(existing.read_text(encoding="utf-8")) if existing else contract
        if previous != contract:
            raise RuntimeError(
                f"Refusing incompatible positional PCSC resume: expected {contract}, got {previous}"
            )
    Path(args.output_dir, "pcsc_contract.json").write_text(
        json.dumps(contract, indent=2) + "\n", encoding="utf-8"
    )
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
        mode="zcf",
        pcsc_aux=args.pcsc_aux,
        pcsc_h12_mode=args.pcsc_h12_mode,
        pcsc_alignment=args.pcsc_alignment,
        sid_component_order=args.sid_component_order,
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
        rqvae_checkpoint_path=None,
        cf_res_path=args.cf_res,
        sem_base_path=args.sem_base,
        sem_res_raw_path=args.sem_res_raw,
        shared_path=args.shared_emb,
    )
    model.set_curriculum(1.0, force_soft=False)
    model.set_pcsc_schedule_factor(0.0)
    model.to(torch.device("cuda", 0))

    effective_save_total_limit = args.save_total_limit
    if args.load_best_model_at_end and effective_save_total_limit <= 0:
        effective_save_total_limit = 5
    elif effective_save_total_limit <= 0:
        effective_save_total_limit = None

    if args.use_wandb:
        try:
            import wandb
            wandb.init(
                project=args.wandb_project,
                entity=(args.wandb_entity or None),
                name=args.wandb_run_name,
                config=vars(args),
                mode=args.wandb_mode,
                dir=(args.wandb_dir or None),
            )
            print(f"[wandb] manual logging enabled: project={args.wandb_project} run={args.wandb_run_name} mode={args.wandb_mode} dir={args.wandb_dir}", flush=True)
        except Exception as exc:
            print(f"[wandb] disabled because init failed: {exc}", flush=True)

    callbacks = []
    if getattr(args, "stop_after_epoch", 0):
        callbacks.append(StopAtEpochCallback(args.stop_after_epoch))
    if args.save_epochs:
        callbacks.append(SelectiveEpochSaveCallback(args.save_epochs))

    trainer = HardOnlyTrainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=valid_data,
        args=transformers.TrainingArguments(
            seed=args.seed,
            data_seed=data_seed,
            full_determinism=(args.full_determinism and not args.determinism_warn_only),
            dataloader_num_workers=args.dataloader_num_workers,
            dataloader_persistent_workers=(
                args.dataloader_persistent_workers and args.dataloader_num_workers > 0
            ),
            dataloader_pin_memory=(not args.disable_dataloader_pin_memory),
            ignore_data_skip=False,
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
            eval_strategy=("no" if args.disable_train_eval else "epoch"),
            save_strategy="epoch",
            output_dir=args.output_dir,
            save_total_limit=effective_save_total_limit,
            load_best_model_at_end=args.load_best_model_at_end,
            metric_for_best_model=(args.metric_for_best_model if args.load_best_model_at_end else None),
            greater_is_better=(args.greater_is_better if args.load_best_model_at_end else None),
            eval_delay=1,
            report_to=[],
            run_name=args.wandb_run_name,
            remove_unused_columns=False,
            disable_tqdm=True,
        ),
        tokenizer=tokenizer,
        data_collator=collator,
        metrics_path=args.training_metrics,
        formal_epochs=(args.schedule_total_epochs if args.schedule_total_epochs > 0 else args.epochs),
        pcsc_max_factor=args.pcsc_max_factor,
        pcsc_schedule_type=args.pcsc_schedule_type,
        callbacks=callbacks,
    )
    model.config.use_cache = False
    trainer.train(resume_from_checkpoint=(args.resume_from_checkpoint or None))
    trainer.save_state()
    trainer.save_model(args.output_dir)
    if args.use_wandb:
        try:
            import wandb
            if wandb.run is not None:
                wandb.finish()
        except Exception:
            pass
    Path(args.run_summary).write_text(
        json.dumps(
            {
                "mode": "hard_only_static_intersection",
                "seed": args.seed,
                "data_seed": data_seed,
                "full_determinism": args.full_determinism,
                "determinism_warn_only": args.determinism_warn_only,
                "dataloader_num_workers": args.dataloader_num_workers,
                "dataloader_persistent_workers": (
                    args.dataloader_persistent_workers and args.dataloader_num_workers > 0
                ),
                "dataloader_pin_memory": (not args.disable_dataloader_pin_memory),
                "formal_epochs": args.epochs,
                "schedule_total_epochs": (args.schedule_total_epochs if args.schedule_total_epochs > 0 else args.epochs),
                "resume_from_checkpoint": args.resume_from_checkpoint,
                "stop_after_epoch": args.stop_after_epoch,
                "completed_epochs": trainer.state.epoch,
                "global_step": trainer.state.global_step,
                "best_checkpoint": trainer.state.best_model_checkpoint,
                "best_metric": trainer.state.best_metric,
                "early_stopping": False,
                "hard_only": True,
                "soft_curriculum": False,
                "codebook_embedding_injection": False,
                "tokenizer_checkpoint_loaded": False,
                "rqvae_checkpoint_loaded": False,
                "pcsc_aux": args.pcsc_aux,
                "pcsc_alignment": args.pcsc_alignment,
                "sid_component_order": args.sid_component_order,
                "pcsc_h12_mode": args.pcsc_h12_mode,
                "pcsc_contract_version": contract["version"],
                "pcsc_contract": contract["contract"],
                "pcsc_max_factor": args.pcsc_max_factor,
                "pcsc_schedule_type": args.pcsc_schedule_type,
                "lambda_cf": args.lambda_cf,
                "lambda_cfres": args.lambda_cfres,
                "lambda_base": args.lambda_base,
                "lambda_res": args.lambda_res,
                "lambda_comp": args.lambda_comp,
                "shared_emb": args.shared_emb,
                "use_wandb": args.use_wandb,
                "wandb_project": args.wandb_project,
                "wandb_run_name": args.wandb_run_name,
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
    parser.add_argument("--index", required=True)
    parser.add_argument("--item_order", required=True)
    parser.add_argument("--cf_emb", required=True)
    parser.add_argument("--sem_emb", required=True)
    parser.add_argument("--cf_res", required=True)
    parser.add_argument("--sem_base", required=True)
    parser.add_argument("--sem_res_raw", required=True)
    parser.add_argument("--shared_emb", required=True)
    parser.add_argument("--pcsc_aux", action="store_true")
    parser.add_argument("--sid_component_order", default="shared,cfres,semres")
    parser.add_argument("--pcsc_max_factor", type=float, default=1.0)
    parser.add_argument("--pcsc_schedule_type", choices=["warmup_hold", "warmup_hold_decay"], default="warmup_hold_decay")
    parser.add_argument("--pcsc_h12_mode", choices=["mean", "sum", "h2"], default="mean")
    parser.add_argument("--pcsc_alignment", choices=["component", "positional"], default="component")
    parser.add_argument("--lambda_cf", type=float, default=1.0)
    parser.add_argument("--lambda_cfres", type=float, default=1.0)
    parser.add_argument("--lambda_base", type=float, default=1.0)
    parser.add_argument("--lambda_res", type=float, default=1.0)
    parser.add_argument("--lambda_comp", type=float, default=1.0)
    parser.add_argument("--training_metrics", required=True)
    parser.add_argument("--run_summary", required=True)
    parser.add_argument("--schedule_total_epochs", type=int, default=0)
    parser.add_argument("--stop_after_epoch", type=float, default=0.0)
    parser.add_argument("--full_determinism", action="store_true")
    parser.add_argument("--determinism_warn_only", action="store_true")
    parser.add_argument("--data_seed", type=int, default=-1)
    parser.add_argument("--dataloader_num_workers", type=int, default=0)
    parser.add_argument("--dataloader_persistent_workers", action="store_true")
    parser.add_argument("--disable_train_eval", action="store_true")
    parser.add_argument("--allow_unversioned_legacy_resume", action="store_true")
    parser.add_argument("--disable_dataloader_pin_memory", action="store_true")
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", default="pls-sd128-dpos-pcsc")
    parser.add_argument("--wandb_entity", default="")
    parser.add_argument("--wandb_mode", default=os.environ.get("WANDB_MODE", "online"))
    parser.add_argument("--wandb_dir", default=os.environ.get("WANDB_DIR", ""))
    parser.add_argument("--load_best_model_at_end", action="store_true")
    parser.add_argument("--metric_for_best_model", default="eval_loss")
    parser.add_argument("--greater_is_better", type=lambda x: str(x).lower() in {"1", "true", "yes", "y"}, default=False)
    parser.add_argument("--save_total_limit", type=int, default=5)
    parser.add_argument(
        "--save_epochs",
        default="",
        help="Comma-separated 1-based epochs to save; overrides epoch saves at all other epochs.",
    )
    main(parser.parse_args())
