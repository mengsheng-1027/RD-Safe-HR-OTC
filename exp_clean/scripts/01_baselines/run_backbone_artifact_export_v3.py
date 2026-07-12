# exp_clean/scripts/01_baselines/run_backbone_artifact_export_v3.py
# Robust export runner for OTC-required backbone artifacts.
# Main fix over v2:
#   1) delete expected old artifact files before each run;
#   2) require EXP_CLEAN_ARTIFACT_JSON in stdout;
#   3) require metadata patch_version/checkpoint_selection markers;
#   4) write separate v3 logs/reports to avoid false success from old artifacts.

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[3]
BASELINE_ROOT = PROJECT_ROOT / "OpenSiteRec" / "baseline"
PATCHED_MAIN = BASELINE_ROOT / "main_exp_clean_export_artifacts_v2.py"
OFFICIAL_SPLIT_ROOT = PROJECT_ROOT / "OpenSiteRec"
LOG_DIR = PROJECT_ROOT / "exp_clean" / "logs" / "baselines" / "artifacts_v3"
RAW_DIR = PROJECT_ROOT / "exp_clean" / "results" / "raw"
REPORT_DIR = PROJECT_ROOT / "exp_clean" / "results" / "reports"
CONFIG_DIR = PROJECT_ROOT / "exp_clean" / "configs"
CKPT_DIR = PROJECT_ROOT / "exp_clean" / "checkpoints" / "baselines"
SCORE_DIR = PROJECT_ROOT / "exp_clean" / "scores" / "local_scores"

CITIES = ["Chicago", "NYC", "Singapore", "Tokyo"]
DEFAULT_MODELS = ["VanillaMF", "LightGCN"]
EXPECTED_PATCH_VERSION = "best_ndcg_export_v2_2026_06_08_a"
EXPECTED_CHECKPOINT_SELECTION = "best_ndcg"


def md5(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def split_hashes() -> dict[str, str]:
    out: dict[str, str] = {}
    for city in CITIES:
        for split in ["train", "test"]:
            path = OFFICIAL_SPLIT_ROOT / city / "split" / f"{split}.pkl"
            out[f"{city}_{split}"] = md5(path) if path.exists() else "MISSING"
    return out


def parse_best_metrics(text: str) -> tuple[float | None, float | None]:
    m = re.search(r"Best Results:\s*\nRecall@\d+:\s*([0-9.]+)\s*\nnDCG@\d+:\s*([0-9.]+)", text)
    if m:
        return float(m.group(1)), float(m.group(2))
    recalls = re.findall(r"Recall@\d+:\s*([0-9.]+)", text)
    ndcgs = re.findall(r"nDCG@\d+:\s*([0-9.]+)", text)
    if recalls and ndcgs:
        return float(recalls[-1]), float(ndcgs[-1])
    return None, None


def parse_artifact_json(text: str) -> dict[str, Any]:
    matches = re.findall(r"EXP_CLEAN_ARTIFACT_JSON=(\{.*\})", text)
    if not matches:
        return {}
    try:
        return json.loads(matches[-1])
    except json.JSONDecodeError:
        return {"_json_parse_error": True, "_raw_tail": matches[-1][-500:]}


def expected_artifact_paths(city: str, model: str, seed: int, epochs: int, dim: int) -> dict[str, Path]:
    tag = f"{city}_{model}_seed{seed}_epochs{epochs}_dim{dim}"
    return {
        "local_score_path": SCORE_DIR / f"{tag}_local_scores.npy",
        "brand_embedding_path": CKPT_DIR / f"{tag}_brand_embeddings.npy",
        "region_embedding_path": CKPT_DIR / f"{tag}_region_embeddings.npy",
        "checkpoint_path": CKPT_DIR / f"{tag}_checkpoint.pt",
        "metadata_path": CKPT_DIR / f"{tag}_metadata.json",
    }


def remove_old_artifacts(paths: dict[str, Path]) -> list[str]:
    removed: list[str] = []
    for p in paths.values():
        if p.exists():
            p.unlink()
            removed.append(str(p))
    return removed


def artifacts_complete(paths: dict[str, Path]) -> bool:
    return all(path.exists() and path.stat().st_size > 0 for path in paths.values())


def load_metadata(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"_metadata_parse_error": f"{type(exc).__name__}: {exc}"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--cities", nargs="+", default=CITIES)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--eval_freq", type=int, default=10)
    parser.add_argument("--cuda", type=int, default=-1)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=0.0001)
    parser.add_argument("--keep_old_artifacts", action="store_true", help="Do not delete old artifact files before running. Not recommended.")
    args = parser.parse_args()

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    SCORE_DIR.mkdir(parents=True, exist_ok=True)

    if not PATCHED_MAIN.exists():
        raise FileNotFoundError(f"Missing patched export main: {PATCHED_MAIN}. Run prepare_official_backbone_artifact_patch_v2.py first.")

    patched_text = PATCHED_MAIN.read_text(encoding="utf-8", errors="replace")
    patched_ok = (
        EXPECTED_PATCH_VERSION in patched_text
        and "EXP_CLEAN_ARTIFACT_JSON=" in patched_text
        and "exp_clean_export_artifacts_v2()" in patched_text
        and "best_state_dict" in patched_text
    )
    if not patched_ok:
        raise RuntimeError(
            "Patched main does not contain required v2 export markers. "
            "Re-run prepare_official_backbone_artifact_patch_v2.py."
        )

    before_hashes = split_hashes()
    rows: list[dict[str, Any]] = []

    for city in args.cities:
        for model in args.models:
            paths = expected_artifact_paths(city, model, args.seed, args.epochs, args.dim)
            log_path = LOG_DIR / f"{city}_{model}_seed{args.seed}_epochs{args.epochs}_artifact_v3.log"
            removed_files = [] if args.keep_old_artifacts else remove_old_artifacts(paths)

            cmd = [
                sys.executable,
                str(PATCHED_MAIN),
                "--city", city,
                "--model", model,
                "--seed", str(args.seed),
                "--epochs", str(args.epochs),
                "--eval_freq", str(args.eval_freq),
                "--cuda", str(args.cuda),
                "--batch_size", str(args.batch_size),
                "--dim", str(args.dim),
                "--lr", str(args.lr),
                "--weight_decay", str(args.weight_decay),
                "--save", "1",
            ]

            print(f"[RUN_V3] {city} {model}")
            started = datetime.now().isoformat(timespec="seconds")
            proc = subprocess.run(
                cmd,
                cwd=str(BASELINE_ROOT),
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
            ended = datetime.now().isoformat(timespec="seconds")
            log_path.write_text(proc.stdout, encoding="utf-8", errors="replace")

            rec, ndcg = parse_best_metrics(proc.stdout)
            artifact_meta = parse_artifact_json(proc.stdout)
            metadata = load_metadata(paths["metadata_path"])
            complete = artifacts_complete(paths)

            artifact_json_found = bool(artifact_meta) and not artifact_meta.get("_json_parse_error")
            metadata_patch_version = metadata.get("patch_version", "")
            metadata_checkpoint_selection = metadata.get("checkpoint_selection", "")
            restored_best_state = metadata.get("restored_best_state_before_export", None)
            selected_epoch = metadata.get("selected_epoch", "")

            v2_metadata_ok = (
                metadata_patch_version == EXPECTED_PATCH_VERSION
                and metadata_checkpoint_selection == EXPECTED_CHECKPOINT_SELECTION
                and restored_best_state is True
            )
            status = "success" if (
                proc.returncode == 0 and complete and artifact_json_found and v2_metadata_ok
            ) else "failed"
            if proc.returncode == 0 and not complete:
                status = "failed_missing_artifacts"
            elif proc.returncode == 0 and complete and not artifact_json_found:
                status = "failed_no_artifact_json"
            elif proc.returncode == 0 and complete and artifact_json_found and not v2_metadata_ok:
                status = "failed_metadata_not_v2_best"

            row: dict[str, Any] = {
                "city": city,
                "model": model,
                "seed": args.seed,
                "epochs": args.epochs,
                "eval_freq": args.eval_freq,
                "batch_size": args.batch_size,
                "dim": args.dim,
                "lr": args.lr,
                "weight_decay": args.weight_decay,
                "status": status,
                "returncode": proc.returncode,
                "recall@20": rec if rec is not None else "",
                "ndcg@20": ndcg if ndcg is not None else "",
                "artifact_json_found": artifact_json_found,
                "metadata_patch_version": metadata_patch_version,
                "metadata_checkpoint_selection": metadata_checkpoint_selection,
                "restored_best_state_before_export": restored_best_state,
                "selected_epoch": selected_epoch,
                "started": started,
                "ended": ended,
                "removed_old_artifact_count": len(removed_files),
                "log_path": str(log_path),
                **{k: str(v) for k, v in paths.items()},
            }
            rows.append(row)
            print(
                f"[{status.upper()}] {city} {model} "
                f"R@20={row['recall@20']} N@20={row['ndcg@20']} "
                f"json={artifact_json_found} patch={metadata_patch_version} "
                f"restored={restored_best_state} epoch={selected_epoch} log={log_path}"
            )

    after_hashes = split_hashes()
    split_hash_unchanged = before_hashes == after_hashes

    csv_path = RAW_DIR / "backbone_artifact_export_v3.csv"
    fieldnames = [
        "city", "model", "seed", "epochs", "eval_freq", "batch_size", "dim", "lr", "weight_decay",
        "status", "returncode", "recall@20", "ndcg@20", "artifact_json_found",
        "metadata_patch_version", "metadata_checkpoint_selection", "restored_best_state_before_export", "selected_epoch",
        "started", "ended", "removed_old_artifact_count", "log_path",
        "local_score_path", "brand_embedding_path", "region_embedding_path", "checkpoint_path", "metadata_path",
    ]
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    success_count = sum(1 for r in rows if r["status"] == "success")
    fail_count = sum(1 for r in rows if r["status"] != "success")

    report_path = REPORT_DIR / "backbone_artifact_export_v3_report.txt"
    report_lines = [
        "# Backbone Artifact Export V3 Report",
        "",
        f"patched_main: {PATCHED_MAIN}",
        f"expected_patch_version: {EXPECTED_PATCH_VERSION}",
        f"cities: {args.cities}",
        f"models: {args.models}",
        f"seed: {args.seed}",
        f"epochs: {args.epochs}",
        f"eval_freq: {args.eval_freq}",
        f"success_count: {success_count}",
        f"fail_count: {fail_count}",
        f"split_hash_unchanged: {split_hash_unchanged}",
        f"csv: {csv_path}",
        "",
    ]
    for r in rows:
        report_lines.append(
            f"{r['city']} | {r['model']} | {r['status']} | R@20={r['recall@20']} | N@20={r['ndcg@20']} | "
            f"artifact_json_found={r['artifact_json_found']} | patch={r['metadata_patch_version']} | "
            f"selection={r['metadata_checkpoint_selection']} | restored={r['restored_best_state_before_export']} | "
            f"selected_epoch={r['selected_epoch']} | log={r['log_path']}"
        )
    report_path.write_text("\n".join(report_lines), encoding="utf-8")

    config_path = CONFIG_DIR / "backbone_artifact_export_v3_config.json"
    config = {
        "script": "exp_clean/scripts/01_baselines/run_backbone_artifact_export_v3.py",
        "patched_main": str(PATCHED_MAIN),
        "expected_patch_version": EXPECTED_PATCH_VERSION,
        "expected_checkpoint_selection": EXPECTED_CHECKPOINT_SELECTION,
        "cities": args.cities,
        "models": args.models,
        "seed": args.seed,
        "epochs": args.epochs,
        "eval_freq": args.eval_freq,
        "cuda": args.cuda,
        "batch_size": args.batch_size,
        "dim": args.dim,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "keep_old_artifacts": args.keep_old_artifacts,
        "split_hash_unchanged": split_hash_unchanged,
        "before_split_hashes": before_hashes,
        "after_split_hashes": after_hashes,
        "note": "V3 deletes old artifacts before each run and requires v2 best-nDCG metadata markers. Official main.py is not modified.",
    }
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[OK] csv: {csv_path}")
    print(f"[OK] report: {report_path}")
    print(f"[OK] config: {config_path}")
    if not split_hash_unchanged:
        raise RuntimeError("Split hashes changed during artifact export. Stop and inspect immediately.")
    if fail_count > 0:
        raise RuntimeError(f"{fail_count} artifact export job(s) failed. Inspect report/logs before continuing.")


if __name__ == "__main__":
    main()
