#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD


SEQ_FIELDS = ("item_ids", "items", "item_seq", "item_list", "sequence")
ITEM_FIELDS = ("item_id", "item", "asin", "business_id", "id")
USER_FIELDS = ("user_id", "user", "uid")
TIME_FIELDS = ("timestamp", "time", "ts", "unixReviewTime")
TRAIN_FIELDS = ("train", "train_items", "train_item_ids", "train_seq", "train_sequence")
VALID_FIELDS = ("valid", "validation", "val")
TEST_FIELDS = ("test",)
TEXT_FIELDS = ("title", "brand", "category", "categories", "description")


def read_json_flexible(path: Path) -> Any:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"empty json file: {path}")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        records = []
        for lineno, line in enumerate(text.splitlines(), 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON/JSONL at {path}:{lineno}: {exc}") from exc
        return records


def write_json(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def first_present(record: dict[str, Any], names: tuple[str, ...]) -> Any:
    for name in names:
        if name in record:
            return record[name]
    return None


def normalize_item_id(value: Any) -> str:
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def sort_item_ids(items: list[str]) -> list[str]:
    unique = list(dict.fromkeys(normalize_item_id(x) for x in items))
    if all(str(x).lstrip("-").isdigit() for x in unique):
        return sorted(unique, key=lambda x: int(x))
    return sorted(unique, key=str)


def detect_sequence_record(record: dict[str, Any]) -> list[Any] | None:
    for field in SEQ_FIELDS:
        value = record.get(field)
        if isinstance(value, list):
            return value
    return None


def explicit_train_from_record(record: dict[str, Any]) -> list[Any] | None:
    for field in TRAIN_FIELDS:
        value = record.get(field)
        if isinstance(value, list):
            return value
    split = record.get("split")
    if isinstance(split, dict):
        for field in TRAIN_FIELDS:
            value = split.get(field)
            if isinstance(value, list):
                return value
    return None


def target_items_from_record(record: dict[str, Any]) -> list[Any]:
    out: list[Any] = []
    for field in VALID_FIELDS + TEST_FIELDS:
        value = record.get(field)
        if isinstance(value, list):
            out.extend(value)
        elif value is not None and not isinstance(value, dict):
            out.append(value)
    split = record.get("split")
    if isinstance(split, dict):
        for field in VALID_FIELDS + TEST_FIELDS:
            value = split.get(field)
            if isinstance(value, list):
                out.extend(value)
            elif value is not None:
                out.append(value)
    return out


def leave_one_out_train(seq: list[Any]) -> list[Any]:
    if len(seq) >= 3:
        return seq[:-2]
    if len(seq) == 2:
        return seq[:1]
    return []


def parse_interactions(inter_json: Path) -> tuple[dict[str, list[str]], list[str], str, dict[str, Any]]:
    obj = read_json_flexible(inter_json)
    train_by_user: dict[str, list[str]] = {}
    all_items: list[str] = []
    explicit = False
    record_count = 0

    def add_sequence(user: Any, seq: list[Any], train_seq: list[Any] | None, targets: list[Any] | None = None) -> None:
        nonlocal explicit, record_count
        record_count += 1
        user_id = str(user)
        seq_items = [normalize_item_id(x) for x in seq if x is not None]
        all_items.extend(seq_items)
        if targets:
            all_items.extend(normalize_item_id(x) for x in targets if x is not None)
        if train_seq is not None:
            explicit = True
            train = [normalize_item_id(x) for x in train_seq if x is not None]
        else:
            train = [normalize_item_id(x) for x in leave_one_out_train(seq) if x is not None]
        train_by_user[user_id] = train

    if isinstance(obj, dict):
        # Common compact format: {"user": [item, item, ...], ...}
        if all(isinstance(v, list) for v in obj.values()):
            for user, seq in obj.items():
                add_sequence(user, seq, None)
        elif "users" in obj and isinstance(obj["users"], list):
            for idx, record in enumerate(obj["users"]):
                if not isinstance(record, dict):
                    continue
                user = first_present(record, USER_FIELDS) or idx
                seq = detect_sequence_record(record)
                train_seq = explicit_train_from_record(record)
                if seq is None:
                    seq = as_list(first_present(record, ITEM_FIELDS))
                add_sequence(user, seq, train_seq, target_items_from_record(record))
        else:
            # Dict of richer per-user records.
            for user, record in obj.items():
                if isinstance(record, dict):
                    seq = detect_sequence_record(record)
                    train_seq = explicit_train_from_record(record)
                    if seq is None:
                        seq = train_seq or as_list(first_present(record, ITEM_FIELDS))
                    add_sequence(record.get("user_id", user), seq, train_seq, target_items_from_record(record))
                elif isinstance(record, list):
                    add_sequence(user, record, None)
    elif isinstance(obj, list):
        if obj and all(isinstance(x, dict) for x in obj):
            has_seq = any(detect_sequence_record(x) is not None or explicit_train_from_record(x) is not None for x in obj)
            if has_seq:
                for idx, record in enumerate(obj):
                    user = first_present(record, USER_FIELDS) or idx
                    seq = detect_sequence_record(record)
                    train_seq = explicit_train_from_record(record)
                    if seq is None:
                        seq = train_seq or as_list(first_present(record, ITEM_FIELDS))
                    add_sequence(user, seq, train_seq, target_items_from_record(record))
            else:
                grouped: dict[str, list[tuple[float, Any]]] = defaultdict(list)
                for record in obj:
                    user = first_present(record, USER_FIELDS)
                    item = first_present(record, ITEM_FIELDS)
                    if user is None or item is None:
                        continue
                    time = first_present(record, TIME_FIELDS)
                    try:
                        sort_time = float(time) if time is not None else math.inf
                    except (TypeError, ValueError):
                        sort_time = math.inf
                    grouped[str(user)].append((sort_time, item))
                for user, events in grouped.items():
                    events.sort(key=lambda x: x[0])
                    add_sequence(user, [item for _, item in events], None)
        else:
            raise ValueError(f"unsupported .inter list format: {inter_json}")
    else:
        raise ValueError(f"unsupported .inter format: {inter_json}")

    all_items.extend(item for seq in train_by_user.values() for item in seq)
    split_rule = "explicit_train_split" if explicit else "leave_one_out"
    meta = {
        "inter_json": str(inter_json),
        "record_count": record_count,
        "num_users": len(train_by_user),
        "split_rule": "explicit_train_split_or_leave_one_out",
        "detected_split_rule": split_rule,
        "train_only": True,
        "no_validation_test_items_in_cf_resource": True,
    }
    return train_by_user, sort_item_ids(all_items), split_rule, meta


def read_item_metadata(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    obj = read_json_flexible(path)
    out: dict[str, dict[str, Any]] = {}
    if isinstance(obj, dict):
        for key, value in obj.items():
            if isinstance(value, dict):
                item = first_present(value, ITEM_FIELDS) or key
                out[normalize_item_id(item)] = value
    elif isinstance(obj, list):
        for value in obj:
            if not isinstance(value, dict):
                continue
            item = first_present(value, ITEM_FIELDS)
            if item is not None:
                out[normalize_item_id(item)] = value
    else:
        raise ValueError(f"unsupported .item format: {path}")
    return out


def flatten_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return " ".join(flatten_text(v) for v in value if v is not None).strip()
    if isinstance(value, dict):
        return " ".join(flatten_text(v) for v in value.values() if v is not None).strip()
    return str(value).strip()


def item_text(item: str, metadata: dict[str, Any]) -> tuple[str, bool]:
    parts = []
    has_text = False
    for field in TEXT_FIELDS:
        value = flatten_text(metadata.get(field))
        if value:
            has_text = True
            label = "category" if field == "categories" else field
            parts.append(f"{label}: {value}")
    if not parts:
        return item, True
    return ". ".join(parts), not has_text


def build_item_order(resource_dir: Path, dataset: str, item_json: Path | None, inter_items: list[str], force: bool) -> tuple[list[str], str]:
    order_path = resource_dir / f"{dataset}_item_id_order.json"
    if order_path.exists() and not force:
        return [normalize_item_id(x) for x in read_json_flexible(order_path)], "existing_resource_order"
    if item_json and item_json.exists():
        metadata = read_item_metadata(item_json)
        order = sort_item_ids(list(metadata.keys()))
        source = "item_json"
    else:
        order = sort_item_ids(inter_items)
        source = "inter_json_items"
    write_json(order, order_path)
    return order, source


def build_cf_svd(
    dataset: str,
    resource_dir: Path,
    item_order: list[str],
    train_by_user: dict[str, list[str]],
    svd_dim: int,
    svd_random_state: int,
    weight: str,
) -> dict[str, Any]:
    if weight != "binary":
        raise ValueError(f"only --svd_weight binary is currently supported, got: {weight}")
    item_to_row = {item: idx for idx, item in enumerate(item_order)}
    user_ids = sorted(train_by_user.keys(), key=lambda x: int(x) if x.lstrip("-").isdigit() else x)
    user_to_col = {user: idx for idx, user in enumerate(user_ids)}
    coords = set()
    for user, seq in train_by_user.items():
        col = user_to_col[user]
        for item in seq:
            row = item_to_row.get(item)
            if row is not None:
                coords.add((row, col))
    if coords:
        rows, cols = zip(*sorted(coords))
        data = np.ones(len(rows), dtype=np.float32)
    else:
        rows, cols, data = [], [], np.asarray([], dtype=np.float32)
    mat = csr_matrix((data, (rows, cols)), shape=(len(item_order), len(user_ids)), dtype=np.float32)
    max_components = max(1, min(mat.shape) - 1)
    actual_dim = min(svd_dim, max_components)
    svd = TruncatedSVD(n_components=actual_dim, random_state=svd_random_state, algorithm="randomized")
    emb = svd.fit_transform(mat).astype("float32")
    if emb.shape[1] < svd_dim:
        emb = np.pad(emb, ((0, 0), (0, svd_dim - emb.shape[1])), mode="constant")
    out = resource_dir / f"{dataset}_trainonly_cf_svd.npy"
    np.save(out, emb.astype("float32"))
    meta = {
        "dataset": dataset,
        "num_items": len(item_order),
        "num_users": len(user_ids),
        "num_train_interactions": int(len(coords)),
        "svd_dim": svd_dim,
        "actual_svd_dim": actual_dim,
        "svd_random_state": svd_random_state,
        "weight": weight,
        "train_only": True,
        "no_validation_test_items_in_cf_resource": True,
        "output": str(out),
        "shape": list(emb.shape),
    }
    write_json(meta, resource_dir / f"{dataset}_cf_svd_meta.json")
    return meta


def build_st5(
    dataset: str,
    item_order: list[str],
    item_json: Path,
    primary_out_dir: Path,
    legacy_out_dir: Path,
    model_name: str,
    batch_size: int,
    max_seq_length: int,
    device: str | None,
    force: bool,
) -> dict[str, Any]:
    emb_path = primary_out_dir / f"{dataset}_st5_rqvae_input_embeddings.npy"
    meta_path = primary_out_dir / f"{dataset}_st5_rqvae_embedding_meta.json"
    order_path = primary_out_dir / f"{dataset}_st5_rqvae_item_id_order.json"
    legacy_emb_path = legacy_out_dir / emb_path.name
    legacy_meta_path = legacy_out_dir / meta_path.name
    legacy_order_path = legacy_out_dir / order_path.name

    if emb_path.exists() and meta_path.exists() and not force:
        meta = read_json_flexible(meta_path)
    else:
        metadata = read_item_metadata(item_json)
        texts: list[str] = []
        missing = 0
        for item in item_order:
            text, missing_text = item_text(item, metadata.get(item, {}))
            texts.append(text)
            missing += int(missing_text)
        backend = "sentence_transformers"
        try:
            from sentence_transformers import SentenceTransformer

            model = SentenceTransformer(model_name, device=device)
            model.max_seq_length = max_seq_length
            emb = model.encode(
                texts,
                batch_size=batch_size,
                convert_to_numpy=True,
                normalize_embeddings=False,
                show_progress_bar=True,
            ).astype("float32")
        except ModuleNotFoundError as exc:
            if exc.name != "sentence_transformers":
                raise
            backend = "transformers_t5_encoder_mean_pooling"
            import torch
            from tqdm.auto import tqdm
            from transformers import AutoTokenizer, T5EncoderModel

            torch_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = T5EncoderModel.from_pretrained(model_name).to(torch_device)
            model.eval()
            chunks: list[np.ndarray] = []
            with torch.inference_mode():
                for start in tqdm(range(0, len(texts), batch_size), desc=f"{dataset} ST5 encode"):
                    batch = texts[start : start + batch_size]
                    encoded = tokenizer(
                        batch,
                        padding=True,
                        truncation=True,
                        max_length=max_seq_length,
                        return_tensors="pt",
                    )
                    encoded = {k: v.to(torch_device) for k, v in encoded.items()}
                    hidden = model(**encoded).last_hidden_state
                    mask = encoded["attention_mask"].unsqueeze(-1).to(hidden.dtype)
                    pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
                    chunks.append(pooled.detach().cpu().numpy().astype("float32"))
            emb = np.concatenate(chunks, axis=0).astype("float32")
        primary_out_dir.mkdir(parents=True, exist_ok=True)
        np.save(emb_path, emb)
        write_json(item_order, order_path)
        meta = {
            "dataset": dataset,
            "model": model_name,
            "backend": backend,
            "device": str(device) if device else None,
            "max_seq_length": max_seq_length,
            "batch_size": batch_size,
            "normalize_embeddings": False,
            "num_items": len(item_order),
            "dim": int(emb.shape[1]),
            "shape": list(emb.shape),
            "item_order": str(Path("results/resources") / dataset / f"{dataset}_item_id_order.json"),
            "st5_item_order": str(order_path),
            "item_json": str(item_json),
            "missing_text_count": int(missing),
            "missing_text_ratio": float(missing / max(len(item_order), 1)),
            "warning": "missing_text_count exceeds 20%" if missing / max(len(item_order), 1) > 0.2 else None,
            "output": str(emb_path),
        }
        write_json(meta, meta_path)

    legacy_out_dir.mkdir(parents=True, exist_ok=True)
    for src, dst in [(emb_path, legacy_emb_path), (meta_path, legacy_meta_path), (order_path, legacy_order_path)]:
        if src.resolve() != dst.resolve():
            shutil.copy2(src, dst)
    meta["legacy_output"] = str(legacy_emb_path)
    write_json(meta, meta_path)
    write_json(meta, legacy_meta_path)
    return meta


def main() -> None:
    parser = argparse.ArgumentParser(description="Build CHORD resources from .inter and optional item metadata.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--inter_json", required=True, type=Path)
    parser.add_argument("--item_json", type=Path, default=None)
    parser.add_argument("--project", type=Path, default=Path("/home/huangxin/llmNrec/pls_sd128_dpos_pcsc_pipeline"))
    parser.add_argument("--letter_root", type=Path, default=Path("/home/huangxin/llmNrec/LETTER-master"))
    parser.add_argument("--svd_dim", type=int, default=128)
    parser.add_argument("--svd_random_state", type=int, default=42)
    parser.add_argument("--svd_weight", default="binary")
    parser.add_argument("--st5_model", default="sentence-transformers/sentence-t5-base")
    parser.add_argument("--st5_batch_size", type=int, default=32)
    parser.add_argument("--st5_max_seq_length", type=int, default=256)
    parser.add_argument("--st5_device", default=None)
    parser.add_argument("--skip_st5", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if not args.inter_json.exists():
        raise FileNotFoundError(args.inter_json)
    if args.item_json is not None and not args.item_json.exists():
        raise FileNotFoundError(args.item_json)

    resource_dir = args.project / "results/resources" / args.dataset
    resource_dir.mkdir(parents=True, exist_ok=True)
    train_by_user, inter_items, split_rule, inter_meta = parse_interactions(args.inter_json)
    item_order, order_source = build_item_order(resource_dir, args.dataset, args.item_json, inter_items, args.force)
    cf_meta = build_cf_svd(
        dataset=args.dataset,
        resource_dir=resource_dir,
        item_order=item_order,
        train_by_user=train_by_user,
        svd_dim=args.svd_dim,
        svd_random_state=args.svd_random_state,
        weight=args.svd_weight,
    )

    st5_meta: dict[str, Any] | None = None
    st5_status = "skipped_by_flag" if args.skip_st5 else "not_built"
    if args.skip_st5:
        print("[st5] skipped by --skip_st5")
    elif args.item_json is None:
        st5_status = "missing_item_json"
        print("Cannot build meaningful ST5 embeddings from .inter only. Need item metadata or precomputed ST5 embeddings.")
    else:
        primary_st5_dir = Path("/home/huangxin/llmNrec/plain_st5_rqvae/input")
        legacy_st5_dir = args.letter_root / "component_relation_sid/rqvae_supervision/results/plain_st5_rqvae/input"
        st5_meta = build_st5(
            dataset=args.dataset,
            item_order=item_order,
            item_json=args.item_json,
            primary_out_dir=primary_st5_dir,
            legacy_out_dir=legacy_st5_dir,
            model_name=args.st5_model,
            batch_size=args.st5_batch_size,
            max_seq_length=args.st5_max_seq_length,
            device=args.st5_device,
            force=args.force,
        )
        st5_status = "built_or_reused"

    build_meta = {
        "dataset": args.dataset,
        "inter_json": str(args.inter_json),
        "item_json": str(args.item_json) if args.item_json else None,
        "item_order_source": order_source,
        "item_order": str(resource_dir / f"{args.dataset}_item_id_order.json"),
        "num_items": len(item_order),
        "split_rule": "explicit_train_split_or_leave_one_out",
        "detected_split_rule": split_rule,
        "train_only": True,
        "no_validation_test_items_in_cf_resource": True,
        "cf_svd": cf_meta,
        "st5_status": st5_status,
        "st5": st5_meta,
        "inter_parse": inter_meta,
    }
    write_json(build_meta, resource_dir / f"{args.dataset}_resource_build_meta.json")
    print(json.dumps(build_meta, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
