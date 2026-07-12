# exp_clean/scripts/02_otc_reproduction/run_otc_gw_multiseed.py
# Purpose:
#   Run OTC-GW reproduction for multiple seeds after multi-seed backbone artifacts exist.
#
# It calls:
#   exp_clean/scripts/02_otc_reproduction/run_otc_gw_reproduction.py
#
# Recommended use:
#   seed 2025/2026 only, because seed 2024 was already computed.
#
# Outputs:
#   exp_clean/results/raw/otc_gw_multiseed.csv
#   exp_clean/results/reports/otc_gw_multiseed_report.txt
#   exp_clean/configs/otc_gw_multiseed_config.json
#   exp_clean/results/raw/otc_gw_seed{seed}_{model}_{target}.csv
#   exp_clean/results/reports/otc_gw_seed{seed}_{model}_{target}_report.txt
#   exp_clean/logs/otc/gw_multiseed/*.log
#
# Note:
#   This script computes or reuses GW transfer scores:
#   exp_clean/scores/transfer_scores/{source}_to_{target}_{model}_seed{seed}_gw_transfer.npy

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
    p.add_argument("--target-cities", nargs="+", default=CITIES, choices=CITIES)
    p.add_argument("--source-cities", nargs="+", default=CITIES, choices=CITIES)
    p.add_argument("--seeds", nargs="+", type=int, default=[2025, 2026])
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--dim", type=int, default=64)
    p.add_argument("--k", type=int, default=20)
    p.add_argument("--gammas", nargs="+", default=["0.5", "1.0", "1.5", "2.0", "2.5", "3.0", "3.5", "4.0", "4.5", "5.0"])
    p.add_argument("--gw-method", default="entropic_gw", choices=["entropic_gw", "gw"])
    p.add_argument("--gw-epsilon", default="0.01")
    p.add_argument("--gw-max-iter", default="80")
    p.add_argument("--score-norm", default="none", choices=["none", "global", "row"])
    p.add_argument("--normalize-emb", action="store_true")
    p.add_argument("--mask-train", action="store_true")
    p.add_argument("--no-row-weighted", action="store_true")
    p.add_argument("--save-plans", action="store_true")
    p.add_argument("--continue-on-error", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    root = project_root()

    runner = root / "exp_clean" / "scripts" / "02_otc_reproduction" / "run_otc_gw_reproduction.py"
    if not runner.exists():
        raise FileNotFoundError(
            f"Missing {runner}. Put run_otc_gw_reproduction.py there first."
        )

    raw_dir = root / "exp_clean" / "results" / "raw"
    report_dir = root / "exp_clean" / "results" / "reports"
    config_dir = root / "exp_clean" / "configs"
    log_dir = root / "exp_clean" / "logs" / "otc" / "gw_multiseed"
    for d in [raw_dir, report_dir, config_dir, log_dir]:
        ensure_dir(d)

    all_frames = []
    report_lines = [
        "# OTC-GW Multi-seed Wrapper Report",
        f"models: {args.models}",
        f"target_cities: {args.target_cities}",
        f"source_cities: {args.source_cities}",
        f"seeds: {args.seeds}",
        f"epochs: {args.epochs}",
        f"dim: {args.dim}",
        f"gammas: {args.gammas}",
        f"gw_method: {args.gw_method}",
        f"gw_epsilon: {args.gw_epsilon}",
        f"gw_max_iter: {args.gw_max_iter}",
        f"score_norm: {args.score_norm}",
        "",
    ]

    success_count = 0
    fail_count = 0

    for seed in args.seeds:
        for model in args.models:
            for target in args.target_cities:
                source_list = [s for s in args.source_cities if s != target]
                tag = f"seed{seed}_{model}_{target}"
                log_path = log_dir / f"otc_gw_{tag}.log"

                cmd = [
                    sys.executable,
                    str(runner),
                    "--models", model,
                    "--target-cities", target,
                    "--source-cities", *source_list,
                    "--seed", str(seed),
                    "--epochs", str(args.epochs),
                    "--dim", str(args.dim),
                    "--k", str(args.k),
                    "--gammas", *args.gammas,
                    "--gw-method", args.gw_method,
                    "--gw-epsilon", str(args.gw_epsilon),
                    "--gw-max-iter", str(args.gw_max_iter),
                    "--score-norm", args.score_norm,
                    "--reuse-cache",
                ]
                if args.normalize_emb:
                    cmd.append("--normalize-emb")
                if args.mask_train:
                    cmd.append("--mask-train")
                if args.no_row_weighted:
                    cmd.append("--no-row-weighted")
                if args.save_plans:
                    cmd.append("--save-plans")

                report_lines.append(f"[RUN] {tag}")
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
                    report_lines.append(f"[FAIL] {tag}, returncode={proc.returncode}")
                    if not args.continue_on_error:
                        raise RuntimeError(f"OTC-GW failed for {tag}. See {log_path}")
                    continue

                csv_src = raw_dir / "otc_gw_reproduction.csv"
                rep_src = report_dir / "otc_gw_reproduction_report.txt"
                cfg_src = config_dir / "otc_gw_reproduction_config.json"

                csv_dst = raw_dir / f"otc_gw_{tag}.csv"
                rep_dst = report_dir / f"otc_gw_{tag}_report.txt"
                cfg_dst = config_dir / f"otc_gw_{tag}_config.json"

                if not csv_src.exists():
                    fail_count += 1
                    report_lines.append(f"[FAIL] {tag}: missing {csv_src}")
                    if not args.continue_on_error:
                        raise FileNotFoundError(csv_src)
                    continue

                shutil.copy2(csv_src, csv_dst)
                if rep_src.exists():
                    shutil.copy2(rep_src, rep_dst)
                if cfg_src.exists():
                    shutil.copy2(cfg_src, cfg_dst)

                df = pd.read_csv(csv_dst)
                df.insert(0, "wrapper_seed", seed)
                df.insert(1, "wrapper_model", model)
                df.insert(2, "wrapper_target_city", target)
                df.to_csv(csv_dst, index=False, encoding="utf-8-sig")
                all_frames.append(df)

                success_count += 1
                report_lines.append(f"[OK] {tag}, csv={csv_dst}, report={rep_dst}")

    merged = pd.concat(all_frames, ignore_index=True) if all_frames else pd.DataFrame()
    merged_path = raw_dir / "otc_gw_multiseed.csv"
    merged.to_csv(merged_path, index=False, encoding="utf-8-sig")

    report_lines.append("")
    report_lines.append(f"success_count: {success_count}")
    report_lines.append(f"fail_count: {fail_count}")
    report_lines.append(f"merged_csv: {merged_path}")

    report_path = report_dir / "otc_gw_multiseed_report.txt"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")

    config = {
        "script": "exp_clean/scripts/02_otc_reproduction/run_otc_gw_multiseed.py",
        "runner": str(runner),
        "models": args.models,
        "target_cities": args.target_cities,
        "source_cities": args.source_cities,
        "seeds": args.seeds,
        "epochs": args.epochs,
        "dim": args.dim,
        "k": args.k,
        "gammas": args.gammas,
        "gw_method": args.gw_method,
        "gw_epsilon": args.gw_epsilon,
        "gw_max_iter": args.gw_max_iter,
        "score_norm": args.score_norm,
        "normalize_emb": args.normalize_emb,
        "mask_train": args.mask_train,
        "row_weighted": not args.no_row_weighted,
        "save_plans": args.save_plans,
        "success_count": success_count,
        "fail_count": fail_count,
        "outputs": {
            "merged_csv": str(merged_path),
            "report": str(report_path),
            "log_dir": str(log_dir),
        },
        "note": "No-fallback OTC-GW multi-seed run. Gamma grid does not include 0 by default.",
    }
    config_path = config_dir / "otc_gw_multiseed_config.json"
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[OK] {merged_path}")
    print(f"[OK] {report_path}")
    print(f"[OK] {config_path}")

    if fail_count > 0:
        raise RuntimeError(f"{fail_count} run(s) failed. Inspect {report_path} and logs.")


if __name__ == "__main__":
    main()
