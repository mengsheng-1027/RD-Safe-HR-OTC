# exp_clean/scripts/01_baselines/run_backbone_artifact_export_multiseed.py
# Purpose:
#   Run backbone artifact export for multiple seeds.
#
# It calls:
#   exp_clean/scripts/01_baselines/run_backbone_artifact_export_v3.py
#
# Required before running:
#   1. prepare_official_backbone_artifact_patch_v2.py has been run successfully.
#   2. run_backbone_artifact_export_v3.py exists in exp_clean/scripts/01_baselines/.
#
# Outputs from each called run_backbone_artifact_export_v3.py:
#   exp_clean/scores/local_scores/*_local_scores.npy
#   exp_clean/checkpoints/baselines/*_brand_embeddings.npy
#   exp_clean/checkpoints/baselines/*_region_embeddings.npy
#   exp_clean/results/raw/backbone_artifact_export_v3.csv
#   exp_clean/results/reports/backbone_artifact_export_v3_report.txt
#
# This wrapper additionally saves per-seed copies:
#   exp_clean/results/raw/backbone_artifact_export_seed{seed}.csv
#   exp_clean/results/reports/backbone_artifact_export_seed{seed}_report.txt
#   exp_clean/results/reports/backbone_artifact_export_multiseed_report.txt
#   exp_clean/configs/backbone_artifact_export_multiseed_config.json

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd


CITIES = ["Chicago", "NYC", "Singapore", "Tokyo"]
MODELS = ["VanillaMF", "LightGCN"]


def project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", default=MODELS, choices=MODELS)
    p.add_argument("--cities", nargs="+", default=CITIES, choices=CITIES)
    p.add_argument("--seeds", nargs="+", type=int, default=[2025, 2026])
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--eval_freq", type=int, default=10)
    p.add_argument("--cuda", type=int, default=-1)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--dim", type=int, default=64)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--weight_decay", type=float, default=0.0001)
    p.add_argument("--keep_old_artifacts", action="store_true")
    p.add_argument("--continue-on-error", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    root = project_root()

    runner = root / "exp_clean" / "scripts" / "01_baselines" / "run_backbone_artifact_export_v3.py"
    if not runner.exists():
        raise FileNotFoundError(
            f"Missing {runner}. Put run_backbone_artifact_export_v3.py there first."
        )

    raw_dir = root / "exp_clean" / "results" / "raw"
    report_dir = root / "exp_clean" / "results" / "reports"
    config_dir = root / "exp_clean" / "configs"
    log_dir = root / "exp_clean" / "logs" / "baselines" / "artifacts_multiseed"
    for d in [raw_dir, report_dir, config_dir, log_dir]:
        ensure_dir(d)

    all_frames = []
    report_lines = [
        "# Backbone Artifact Export Multi-seed Wrapper Report",
        f"models: {args.models}",
        f"cities: {args.cities}",
        f"seeds: {args.seeds}",
        f"epochs: {args.epochs}",
        f"dim: {args.dim}",
        "",
    ]

    success_count = 0
    fail_count = 0

    for seed in args.seeds:
        log_path = log_dir / f"backbone_artifact_export_seed{seed}.log"

        cmd = [
            sys.executable,
            str(runner),
            "--models", *args.models,
            "--cities", *args.cities,
            "--seed", str(seed),
            "--epochs", str(args.epochs),
            "--eval_freq", str(args.eval_freq),
            "--cuda", str(args.cuda),
            "--batch_size", str(args.batch_size),
            "--dim", str(args.dim),
            "--lr", str(args.lr),
            "--weight_decay", str(args.weight_decay),
        ]
        if args.keep_old_artifacts:
            cmd.append("--keep_old_artifacts")

        report_lines.append(f"[RUN] seed={seed}")
        report_lines.append(f"      log={log_path}")

        with open(log_path, "w", encoding="utf-8") as f:
            proc = subprocess.run(
                cmd,
                cwd=str(root),
                stdout=f,
                stderr=subprocess.STDOUT,
                text=True,
            )

        if proc.returncode != 0:
            fail_count += 1
            report_lines.append(f"[FAIL] seed={seed}, returncode={proc.returncode}")
            if not args.continue_on_error:
                raise RuntimeError(f"Backbone artifact export failed for seed={seed}. See {log_path}")
            continue

        seed_csv_src = raw_dir / "backbone_artifact_export_v3.csv"
        seed_report_src = report_dir / "backbone_artifact_export_v3_report.txt"

        seed_csv_dst = raw_dir / f"backbone_artifact_export_seed{seed}.csv"
        seed_report_dst = report_dir / f"backbone_artifact_export_seed{seed}_report.txt"

        if seed_csv_src.exists():
            shutil.copy2(seed_csv_src, seed_csv_dst)
            df = pd.read_csv(seed_csv_dst)
            df.insert(0, "wrapper_seed", seed)
            df.to_csv(seed_csv_dst, index=False, encoding="utf-8-sig")
            all_frames.append(df)
        else:
            fail_count += 1
            report_lines.append(f"[FAIL] seed={seed}: missing {seed_csv_src}")
            if not args.continue_on_error:
                raise FileNotFoundError(seed_csv_src)
            continue

        if seed_report_src.exists():
            shutil.copy2(seed_report_src, seed_report_dst)

        success_count += 1
        report_lines.append(f"[OK] seed={seed}, csv={seed_csv_dst}, report={seed_report_dst}")

    if all_frames:
        merged = pd.concat(all_frames, ignore_index=True)
    else:
        merged = pd.DataFrame()

    merged_path = raw_dir / "backbone_artifact_export_multiseed.csv"
    merged.to_csv(merged_path, index=False, encoding="utf-8-sig")

    report_lines.append("")
    report_lines.append(f"success_count: {success_count}")
    report_lines.append(f"fail_count: {fail_count}")
    report_lines.append(f"merged_csv: {merged_path}")

    report_path = report_dir / "backbone_artifact_export_multiseed_report.txt"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")

    config = {
        "script": "exp_clean/scripts/01_baselines/run_backbone_artifact_export_multiseed.py",
        "runner": str(runner),
        "models": args.models,
        "cities": args.cities,
        "seeds": args.seeds,
        "epochs": args.epochs,
        "eval_freq": args.eval_freq,
        "cuda": args.cuda,
        "batch_size": args.batch_size,
        "dim": args.dim,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "keep_old_artifacts": args.keep_old_artifacts,
        "success_count": success_count,
        "fail_count": fail_count,
        "outputs": {
            "merged_csv": str(merged_path),
            "report": str(report_path),
            "log_dir": str(log_dir),
        },
        "note": "This wrapper only exports local scores and embeddings. It does not compute OTC-GW transfer scores.",
    }
    config_path = config_dir / "backbone_artifact_export_multiseed_config.json"
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[OK] {merged_path}")
    print(f"[OK] {report_path}")
    print(f"[OK] {config_path}")

    if fail_count > 0:
        raise RuntimeError(f"{fail_count} seed run(s) failed. Inspect {report_path} and logs.")


if __name__ == "__main__":
    main()
