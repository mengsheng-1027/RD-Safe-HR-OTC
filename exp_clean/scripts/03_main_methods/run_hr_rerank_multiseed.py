# exp_clean/scripts/03_main_methods/run_hr_rerank_multiseed.py
# Purpose:
#   Run final RD-Safe + HR-Rerank experiments for multiple seeds and summarize mean/std.
#
# It calls:
#   exp_clean/scripts/03_main_methods/run_otc_gw_rd_safe_hr_rerank.py
#
# Required before running:
#   1. local scores for every seed/model/city exist.
#   2. OTC-GW transfer scores for every seed/model/source-target exist.
#
# Outputs:
#   exp_clean/results/raw/hr_rerank_multiseed.csv
#   exp_clean/results/tables/hr_rerank_multiseed_mean_std.csv
#   exp_clean/results/reports/hr_rerank_multiseed_report.txt
#   exp_clean/configs/hr_rerank_multiseed_config.json
#   per-seed files:
#     exp_clean/results/raw/hr_rerank_seed{seed}.csv
#     exp_clean/results/raw/hr_rerank_params_seed{seed}.csv
#     exp_clean/results/reports/hr_rerank_seed{seed}_report.txt

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
    p.add_argument("--seeds", nargs="+", type=int, default=[2024, 2025, 2026])
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--dim", type=int, default=64)
    p.add_argument("--k", type=int, default=20)

    p.add_argument("--otc-gammas", nargs="+", default=["0.5", "1.0", "1.5", "2.0", "2.5", "3.0", "3.5", "4.0", "4.5", "5.0"])
    p.add_argument("--rd-betas", nargs="+", default=["0.0", "0.001", "0.002", "0.005", "0.01", "0.02", "0.05", "0.1", "0.2", "0.5", "1.0"])
    p.add_argument("--score-norm", default="none", choices=["none", "global", "row"])

    p.add_argument("--source-min-ndcg-gain", default="0.0005")
    p.add_argument("--method-min-ndcg-gain", default="0.0005")
    p.add_argument("--min-reliable-sources", default="1")

    p.add_argument("--hr-lambdas", nargs="+", default=["0.0", "0.02", "0.05", "0.1", "0.2", "0.5", "1.0"])
    p.add_argument("--hr-min-ndcg-gain", default="0.0001")
    p.add_argument("--recall-tolerance", default="0.0")

    p.add_argument("--mask-train", action="store_true")
    p.add_argument("--no-row-weighted", action="store_true")
    p.add_argument("--continue-on-error", action="store_true")
    return p.parse_args()


def add_improvement_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add relative improvements vs Target-only and Original OTC for the same seed/model/city/split."""
    target = (
        df[df["method"] == "Target-only"][
            ["seed", "model", "target_city", "eval_split", "recall@20", "ndcg@20"]
        ]
        .rename(columns={"recall@20": "target_recall@20", "ndcg@20": "target_ndcg@20"})
    )
    otc = (
        df[df["method"] == "Original-OTC-GW-NoFallback"][
            ["seed", "model", "target_city", "eval_split", "recall@20", "ndcg@20"]
        ]
        .rename(columns={"recall@20": "otc_recall@20", "ndcg@20": "otc_ndcg@20"})
    )
    out = df.merge(target, on=["seed", "model", "target_city", "eval_split"], how="left")
    out = out.merge(otc, on=["seed", "model", "target_city", "eval_split"], how="left")

    out["delta_R_vs_target"] = out["recall@20"] - out["target_recall@20"]
    out["delta_N_vs_target"] = out["ndcg@20"] - out["target_ndcg@20"]
    out["pct_R_vs_target"] = out["delta_R_vs_target"] / out["target_recall@20"] * 100.0
    out["pct_N_vs_target"] = out["delta_N_vs_target"] / out["target_ndcg@20"] * 100.0

    out["delta_R_vs_otc"] = out["recall@20"] - out["otc_recall@20"]
    out["delta_N_vs_otc"] = out["ndcg@20"] - out["otc_ndcg@20"]
    out["pct_R_vs_otc"] = out["delta_R_vs_otc"] / out["otc_recall@20"] * 100.0
    out["pct_N_vs_otc"] = out["delta_N_vs_otc"] / out["otc_ndcg@20"] * 100.0
    return out


def make_summary(df: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "recall@20", "ndcg@20",
        "delta_R_vs_target", "delta_N_vs_target", "pct_R_vs_target", "pct_N_vs_target",
        "delta_R_vs_otc", "delta_N_vs_otc", "pct_R_vs_otc", "pct_N_vs_otc",
    ]
    grouped = df.groupby(["model", "method", "target_city", "eval_split"], as_index=False)
    mean = grouped[metrics].mean()
    std = grouped[metrics].std(ddof=1)

    mean = mean.rename(columns={m: f"{m}_mean" for m in metrics})
    std = std.rename(columns={m: f"{m}_std" for m in metrics})
    out = mean.merge(std, on=["model", "method", "target_city", "eval_split"], how="left")

    # make compact printable fields
    out["recall@20_mean±std"] = out.apply(
        lambda r: f"{r['recall@20_mean']:.4f}±{0.0 if pd.isna(r['recall@20_std']) else r['recall@20_std']:.4f}",
        axis=1,
    )
    out["ndcg@20_mean±std"] = out.apply(
        lambda r: f"{r['ndcg@20_mean']:.4f}±{0.0 if pd.isna(r['ndcg@20_std']) else r['ndcg@20_std']:.4f}",
        axis=1,
    )
    return out


def main() -> None:
    args = parse_args()
    root = project_root()

    runner = root / "exp_clean" / "scripts" / "03_main_methods" / "run_otc_gw_rd_safe_hr_rerank.py"
    if not runner.exists():
        raise FileNotFoundError(
            f"Missing {runner}. Put run_otc_gw_rd_safe_hr_rerank.py there first."
        )

    raw_dir = root / "exp_clean" / "results" / "raw"
    table_dir = root / "exp_clean" / "results" / "tables"
    report_dir = root / "exp_clean" / "results" / "reports"
    config_dir = root / "exp_clean" / "configs"
    log_dir = root / "exp_clean" / "logs" / "ours" / "hr_rerank_multiseed"
    for d in [raw_dir, table_dir, report_dir, config_dir, log_dir]:
        ensure_dir(d)

    all_frames = []
    all_param_frames = []
    report_lines = [
        "# HR-Rerank Multi-seed Wrapper Report",
        f"models: {args.models}",
        f"target_cities: {args.target_cities}",
        f"source_cities: {args.source_cities}",
        f"seeds: {args.seeds}",
        f"epochs: {args.epochs}",
        f"dim: {args.dim}",
        f"otc_gammas: {args.otc_gammas}",
        f"hr_lambdas: {args.hr_lambdas}",
        "",
    ]

    success_count = 0
    fail_count = 0

    for seed in args.seeds:
        log_path = log_dir / f"hr_rerank_seed{seed}.log"
        cmd = [
            sys.executable,
            str(runner),
            "--models", *args.models,
            "--target-cities", *args.target_cities,
            "--source-cities", *args.source_cities,
            "--seed", str(seed),
            "--epochs", str(args.epochs),
            "--dim", str(args.dim),
            "--k", str(args.k),
            "--otc-gammas", *args.otc_gammas,
            "--rd-betas", *args.rd_betas,
            "--score-norm", args.score_norm,
            "--source-min-ndcg-gain", str(args.source_min_ndcg_gain),
            "--method-min-ndcg-gain", str(args.method_min_ndcg_gain),
            "--min-reliable-sources", str(args.min_reliable_sources),
            "--hr-lambdas", *args.hr_lambdas,
            "--hr-min-ndcg-gain", str(args.hr_min_ndcg_gain),
            "--recall-tolerance", str(args.recall_tolerance),
        ]
        if args.mask_train:
            cmd.append("--mask-train")
        if args.no_row_weighted:
            cmd.append("--no-row-weighted")

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
                raise RuntimeError(f"HR-Rerank final run failed for seed={seed}. See {log_path}")
            continue

        csv_src = raw_dir / "otc_gw_rd_safe_hr_rerank.csv"
        param_src = raw_dir / "otc_gw_hr_params.csv"
        rep_src = report_dir / "otc_gw_rd_safe_hr_rerank_report.txt"
        cfg_src = config_dir / "otc_gw_rd_safe_hr_rerank_config.json"

        csv_dst = raw_dir / f"hr_rerank_seed{seed}.csv"
        param_dst = raw_dir / f"hr_rerank_params_seed{seed}.csv"
        rep_dst = report_dir / f"hr_rerank_seed{seed}_report.txt"
        cfg_dst = config_dir / f"hr_rerank_seed{seed}_config.json"

        if not csv_src.exists():
            fail_count += 1
            report_lines.append(f"[FAIL] seed={seed}: missing {csv_src}")
            if not args.continue_on_error:
                raise FileNotFoundError(csv_src)
            continue

        shutil.copy2(csv_src, csv_dst)
        if param_src.exists():
            shutil.copy2(param_src, param_dst)
        if rep_src.exists():
            shutil.copy2(rep_src, rep_dst)
        if cfg_src.exists():
            shutil.copy2(cfg_src, cfg_dst)

        df = pd.read_csv(csv_dst)
        df.insert(0, "seed", seed)
        df.to_csv(csv_dst, index=False, encoding="utf-8-sig")
        all_frames.append(df)

        if param_dst.exists():
            pdf = pd.read_csv(param_dst)
            pdf.insert(0, "seed", seed)
            pdf.to_csv(param_dst, index=False, encoding="utf-8-sig")
            all_param_frames.append(pdf)

        success_count += 1
        report_lines.append(f"[OK] seed={seed}, csv={csv_dst}, report={rep_dst}")

    if all_frames:
        merged = pd.concat(all_frames, ignore_index=True)
        merged = add_improvement_columns(merged)
        summary = make_summary(merged)
    else:
        merged = pd.DataFrame()
        summary = pd.DataFrame()

    merged_path = raw_dir / "hr_rerank_multiseed.csv"
    summary_path = table_dir / "hr_rerank_multiseed_mean_std.csv"
    merged.to_csv(merged_path, index=False, encoding="utf-8-sig")
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")

    if all_param_frames:
        params = pd.concat(all_param_frames, ignore_index=True)
    else:
        params = pd.DataFrame()
    params_path = raw_dir / "hr_rerank_params_multiseed.csv"
    params.to_csv(params_path, index=False, encoding="utf-8-sig")

    # Official test compact summary for quick reading.
    compact_cols = [
        "model", "method", "target_city", "eval_split",
        "recall@20_mean±std", "ndcg@20_mean±std",
        "pct_R_vs_target_mean", "pct_R_vs_target_std",
        "pct_N_vs_target_mean", "pct_N_vs_target_std",
        "pct_R_vs_otc_mean", "pct_R_vs_otc_std",
        "pct_N_vs_otc_mean", "pct_N_vs_otc_std",
    ]
    compact = summary[compact_cols].copy() if not summary.empty else pd.DataFrame(columns=compact_cols)
    compact_path = table_dir / "hr_rerank_multiseed_compact.csv"
    compact.to_csv(compact_path, index=False, encoding="utf-8-sig")

    report_lines.append("")
    report_lines.append(f"success_count: {success_count}")
    report_lines.append(f"fail_count: {fail_count}")
    report_lines.append(f"merged_csv: {merged_path}")
    report_lines.append(f"summary_csv: {summary_path}")
    report_lines.append(f"compact_csv: {compact_path}")
    report_lines.append(f"params_csv: {params_path}")

    report_path = report_dir / "hr_rerank_multiseed_report.txt"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")

    config = {
        "script": "exp_clean/scripts/03_main_methods/run_hr_rerank_multiseed.py",
        "runner": str(runner),
        "models": args.models,
        "target_cities": args.target_cities,
        "source_cities": args.source_cities,
        "seeds": args.seeds,
        "epochs": args.epochs,
        "dim": args.dim,
        "k": args.k,
        "otc_gammas": args.otc_gammas,
        "rd_betas": args.rd_betas,
        "score_norm": args.score_norm,
        "source_min_ndcg_gain": args.source_min_ndcg_gain,
        "method_min_ndcg_gain": args.method_min_ndcg_gain,
        "min_reliable_sources": args.min_reliable_sources,
        "hr_lambdas": args.hr_lambdas,
        "hr_min_ndcg_gain": args.hr_min_ndcg_gain,
        "recall_tolerance": args.recall_tolerance,
        "mask_train": args.mask_train,
        "row_weighted": not args.no_row_weighted,
        "success_count": success_count,
        "fail_count": fail_count,
        "outputs": {
            "merged_csv": str(merged_path),
            "summary_csv": str(summary_path),
            "compact_csv": str(compact_path),
            "params_csv": str(params_path),
            "report": str(report_path),
            "log_dir": str(log_dir),
        },
        "note": "Final multi-seed RD-Safe + HR-Rerank experiment. Summary reports mean/std over seeds.",
    }
    config_path = config_dir / "hr_rerank_multiseed_config.json"
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[OK] {merged_path}")
    print(f"[OK] {summary_path}")
    print(f"[OK] {compact_path}")
    print(f"[OK] {params_path}")
    print(f"[OK] {report_path}")
    print(f"[OK] {config_path}")

    if fail_count > 0:
        raise RuntimeError(f"{fail_count} seed run(s) failed. Inspect {report_path} and logs.")


if __name__ == "__main__":
    main()
