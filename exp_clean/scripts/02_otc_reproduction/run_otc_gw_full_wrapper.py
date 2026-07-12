# exp_clean/scripts/02_otc_reproduction/run_otc_gw_full_wrapper.py
# Purpose:
#   Run OTC-GW full experiments model-by-model and target-by-target.
#
# Why wrapper:
#   run_otc_gw_reproduction.py can be slow/memory-heavy for LightGCN.
#   This wrapper runs one (model, target_city) at a time, immediately saves each chunk,
#   and merges all chunks into one master CSV/report.
#
# Requirements:
#   - exp_clean/scripts/02_otc_reproduction/run_otc_gw_reproduction.py must exist.
#   - POT package must be installed: pip install POT
#
# Outputs:
#   exp_clean/results/raw/otc_gw_full.csv
#   exp_clean/results/reports/otc_gw_full_report.txt
#   exp_clean/configs/otc_gw_full_config.json
#   exp_clean/logs/otc/gw_full/*.log

import argparse
import json
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


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", default=MODELS, choices=MODELS)
    p.add_argument("--target-cities", nargs="+", default=CITIES, choices=CITIES)
    p.add_argument("--source-cities", nargs="+", default=CITIES, choices=CITIES)
    p.add_argument("--gammas", nargs="+", default=["0", "0.001", "0.002", "0.005", "0.01", "0.02", "0.05", "0.1", "0.2", "0.5", "1", "2", "5"])
    p.add_argument("--gw-method", default="entropic_gw", choices=["entropic_gw", "gw"])
    p.add_argument("--gw-epsilon", default="0.01")
    p.add_argument("--gw-max-iter", default="80")
    p.add_argument("--score-norm", default="none", choices=["none", "global", "row"])
    p.add_argument("--normalize-emb", action="store_true")
    p.add_argument("--mask-train", action="store_true")
    p.add_argument("--no-row-weighted", action="store_true")
    p.add_argument("--reuse-cache", action="store_true", default=True)
    p.add_argument("--save-plans", action="store_true")
    p.add_argument("--continue-on-error", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    root = project_root()

    runner = root / "exp_clean" / "scripts" / "02_otc_reproduction" / "run_otc_gw_reproduction.py"
    if not runner.exists():
        raise FileNotFoundError(runner)

    raw_dir = root / "exp_clean" / "results" / "raw"
    report_dir = root / "exp_clean" / "results" / "reports"
    config_dir = root / "exp_clean" / "configs"
    log_dir = root / "exp_clean" / "logs" / "otc" / "gw_full"
    chunk_dir = raw_dir / "otc_gw_full_chunks"

    for d in [raw_dir, report_dir, config_dir, log_dir, chunk_dir]:
        ensure_dir(d)

    all_frames = []
    report_lines = [
        "# OTC-GW Full Wrapper Report",
        f"models: {args.models}",
        f"target_cities: {args.target_cities}",
        f"source_cities: {args.source_cities}",
        f"gammas: {args.gammas}",
        f"gw_method: {args.gw_method}",
        f"gw_epsilon: {args.gw_epsilon}",
        f"gw_max_iter: {args.gw_max_iter}",
        f"score_norm: {args.score_norm}",
        "",
    ]

    success_count = 0
    fail_count = 0

    for model in args.models:
        for target in args.target_cities:
            sources = [s for s in args.source_cities if s != target]
            if not sources:
                report_lines.append(f"[SKIP] {model} target={target}: no valid sources")
                continue

            log_path = log_dir / f"{model}_{target}_otc_gw.log"

            cmd = [
                sys.executable,
                str(runner),
                "--models", model,
                "--target-cities", target,
                "--source-cities", *args.source_cities,
                "--gammas", *args.gammas,
                "--gw-method", args.gw_method,
                "--gw-epsilon", args.gw_epsilon,
                "--gw-max-iter", args.gw_max_iter,
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

            report_lines.append(f"[RUN] model={model}, target={target}, sources={sources}")
            report_lines.append(f"      log={log_path}")

            with open(log_path, "w", encoding="utf-8") as log_f:
                proc = subprocess.run(
                    cmd,
                    cwd=str(root),
                    stdout=log_f,
                    stderr=subprocess.STDOUT,
                    text=True,
                )

            if proc.returncode != 0:
                fail_count += 1
                report_lines.append(f"[FAIL] model={model}, target={target}, returncode={proc.returncode}")
                if not args.continue_on_error:
                    raise RuntimeError(f"Failed: model={model}, target={target}. See log: {log_path}")
                continue

            chunk_src = raw_dir / "otc_gw_reproduction.csv"
            if not chunk_src.exists():
                fail_count += 1
                msg = f"Missing chunk output after success: {chunk_src}"
                report_lines.append(f"[FAIL] model={model}, target={target}: {msg}")
                if not args.continue_on_error:
                    raise FileNotFoundError(msg)
                continue

            chunk_df = pd.read_csv(chunk_src)
            chunk_df.insert(0, "chunk_model", model)
            chunk_df.insert(1, "chunk_target_city", target)
            chunk_path = chunk_dir / f"otc_gw_{model}_{target}.csv"
            chunk_df.to_csv(chunk_path, index=False, encoding="utf-8-sig")
            all_frames.append(chunk_df)

            success_count += 1
            report_lines.append(f"[OK] model={model}, target={target}, rows={len(chunk_df)}, chunk={chunk_path}")

    if all_frames:
        merged = pd.concat(all_frames, ignore_index=True)
    else:
        merged = pd.DataFrame()

    out_csv = raw_dir / "otc_gw_full.csv"
    merged.to_csv(out_csv, index=False, encoding="utf-8-sig")

    report_lines.append("")
    report_lines.append(f"success_count: {success_count}")
    report_lines.append(f"fail_count: {fail_count}")
    report_lines.append(f"csv: {out_csv}")

    report_path = report_dir / "otc_gw_full_report.txt"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")

    config = {
        "script": "exp_clean/scripts/02_otc_reproduction/run_otc_gw_full_wrapper.py",
        "models": args.models,
        "target_cities": args.target_cities,
        "source_cities": args.source_cities,
        "gammas": args.gammas,
        "gw_method": args.gw_method,
        "gw_epsilon": args.gw_epsilon,
        "gw_max_iter": args.gw_max_iter,
        "score_norm": args.score_norm,
        "normalize_emb": args.normalize_emb,
        "mask_train": args.mask_train,
        "row_weighted": not args.no_row_weighted,
        "reuse_cache": args.reuse_cache,
        "save_plans": args.save_plans,
        "success_count": success_count,
        "fail_count": fail_count,
        "official_files_modified": False,
        "outputs": {
            "csv": str(out_csv),
            "report": str(report_path),
            "chunk_dir": str(chunk_dir),
            "log_dir": str(log_dir),
        },
    }
    config_path = config_dir / "otc_gw_full_config.json"
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[OK] {out_csv}")
    print(f"[OK] {report_path}")
    print(f"[OK] {config_path}")


if __name__ == "__main__":
    main()
