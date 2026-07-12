# exp_clean/scripts/04_ablation/run_complete_ablation_multiseed.py
# Purpose:
#   Run a more complete ablation study for RD-Safe + HR-Rerank.
#
# This script reuses the implementation in:
#   exp_clean/scripts/03_main_methods/run_otc_gw_rd_safe_hr_rerank.py
#
# Ablation groups:
#   A. Main component ablation
#      - Target-only
#      - Original-OTC-GW-NoFallback
#      - RD-Safe-OTC-GW
#      - HR-Rerank-OTC-GW
#      - RD-Safe+HR-Rerank-OTC-GW
#
#   B. HR signal ablation on top of RD-Safe
#      - RD-Safe+Cat
#      - RD-Safe+RR
#      - RD-Safe+Pop
#      - RD-Safe+HR-w/o-Cat
#      - RD-Safe+HR-w/o-RR
#      - RD-Safe+HR-w/o-Pop
#
# Notes:
#   - All tuning is done on internal valid.
#   - Official test and strict unseen test are evaluation-only.
#   - No official split file is modified.
#
# Outputs:
#   exp_clean/results/raw/complete_ablation_multiseed.csv
#   exp_clean/results/tables/complete_ablation_multiseed_mean_std.csv
#   exp_clean/results/tables/complete_ablation_official_avg.csv
#   exp_clean/results/reports/complete_ablation_multiseed_report.txt
#   exp_clean/configs/complete_ablation_multiseed_config.json

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


CITIES = ["Chicago", "NYC", "Singapore", "Tokyo"]
MODELS = ["VanillaMF", "LightGCN"]


def project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def load_final_module(root: Path):
    module_path = root / "exp_clean" / "scripts" / "03_main_methods" / "run_otc_gw_rd_safe_hr_rerank.py"
    if not module_path.exists():
        raise FileNotFoundError(f"Missing dependency: {module_path}")
    spec = importlib.util.spec_from_file_location("final_method", module_path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", default=MODELS, choices=MODELS)
    p.add_argument("--target-cities", nargs="+", default=CITIES, choices=CITIES)
    p.add_argument("--source-cities", nargs="+", default=CITIES, choices=CITIES)
    p.add_argument("--seeds", nargs="+", type=int, default=[2024, 2025, 2026])
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--dim", type=int, default=64)
    p.add_argument("--k", type=int, default=20)

    p.add_argument("--otc-gammas", nargs="+", type=float, default=[0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0])
    p.add_argument("--rd-betas", nargs="+", type=float, default=[0.0, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0])
    p.add_argument("--score-norm", default="none", choices=["none", "global", "row"])

    p.add_argument("--source-min-ndcg-gain", type=float, default=0.0005)
    p.add_argument("--method-min-ndcg-gain", type=float, default=0.0005)
    p.add_argument("--min-reliable-sources", type=int, default=1)

    p.add_argument("--hr-lambdas", nargs="+", type=float, default=[0.0, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0])
    p.add_argument("--hr-min-ndcg-gain", type=float, default=0.0001)
    p.add_argument("--recall-tolerance", type=float, default=0.0)

    p.add_argument("--mask-train", action="store_true")
    p.add_argument("--no-row-weighted", action="store_true")
    return p.parse_args()


def graph_variant(graphs: Dict[str, np.ndarray], variant: str) -> Dict[str, np.ndarray]:
    """Return graph scores with only selected HR signals enabled."""
    z = np.zeros_like(graphs["cat"], dtype=np.float32)

    enabled = {
        "full": ("cat", "rr", "pop"),
        "cat_only": ("cat",),
        "rr_only": ("rr",),
        "pop_only": ("pop",),
        "without_cat": ("rr", "pop"),
        "without_rr": ("cat", "pop"),
        "without_pop": ("cat", "rr"),
    }[variant]

    return {
        "cat": graphs["cat"] if "cat" in enabled else z,
        "rr": graphs["rr"] if "rr" in enabled else z,
        "pop": graphs["pop"] if "pop" in enabled else z,
    }


def add_eval_rows(mod, rows: List[dict], *, seed: int, method: str, group: str, model: str, target: str,
                  score: np.ndarray, train_df: pd.DataFrame, official_test_df: pd.DataFrame,
                  strict_df: pd.DataFrame | None, args: argparse.Namespace, params: dict, notes: str) -> None:
    for split_name, eval_df in [("official_test", official_test_df), ("strict_unseen_test", strict_df)]:
        if eval_df is None:
            continue
        recall, ndcg = mod.evaluate_scores(
            score, train_df, eval_df, args.k, args.mask_train, not args.no_row_weighted
        )
        row = {
            "seed": seed,
            "group": group,
            "method": method,
            "model": model,
            "target_city": target,
            "eval_split": split_name,
            "recall@20": recall,
            "ndcg@20": ndcg,
            "status": "success",
            "notes": notes,
        }
        row.update(params)
        rows.append(row)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    metric_cols = ["recall@20", "ndcg@20"]
    g = df.groupby(["group", "method", "model", "target_city", "eval_split"], as_index=False)
    mean = g[metric_cols].mean().rename(columns={c: f"{c}_mean" for c in metric_cols})
    std = g[metric_cols].std(ddof=1).rename(columns={c: f"{c}_std" for c in metric_cols})
    out = mean.merge(std, on=["group", "method", "model", "target_city", "eval_split"], how="left")
    out["recall@20_mean±std"] = out.apply(
        lambda r: f"{r['recall@20_mean']:.4f}±{0.0 if pd.isna(r['recall@20_std']) else r['recall@20_std']:.4f}",
        axis=1,
    )
    out["ndcg@20_mean±std"] = out.apply(
        lambda r: f"{r['ndcg@20_mean']:.4f}±{0.0 if pd.isna(r['ndcg@20_std']) else r['ndcg@20_std']:.4f}",
        axis=1,
    )
    return out


def average_official(df: pd.DataFrame) -> pd.DataFrame:
    official = df[df["eval_split"] == "official_test"].copy()
    seed_avg = official.groupby(["seed", "group", "method", "model"], as_index=False)[["recall@20", "ndcg@20"]].mean()
    out = seed_avg.groupby(["group", "method", "model"], as_index=False).agg(
        recall_mean=("recall@20", "mean"),
        recall_std=("recall@20", "std"),
        ndcg_mean=("ndcg@20", "mean"),
        ndcg_std=("ndcg@20", "std"),
    )
    out["recall@20_mean±std"] = out.apply(lambda r: f"{r['recall_mean']:.4f}±{r['recall_std']:.4f}", axis=1)
    out["ndcg@20_mean±std"] = out.apply(lambda r: f"{r['ndcg_mean']:.4f}±{r['ndcg_std']:.4f}", axis=1)
    return out


def main() -> None:
    args = parse_args()
    root = project_root()
    mod = load_final_module(root)

    official_root = root / "exp_clean" / "data_splits" / "official"
    internal_root = root / "exp_clean" / "data_splits" / "internal_valid"
    strict_root = root / "exp_clean" / "data_splits" / "strict_unseen"

    raw_dir = root / "exp_clean" / "results" / "raw"
    table_dir = root / "exp_clean" / "results" / "tables"
    report_dir = root / "exp_clean" / "results" / "reports"
    config_dir = root / "exp_clean" / "configs"
    for d in [raw_dir, table_dir, report_dir, config_dir]:
        ensure_dir(d)

    rows = []
    param_rows = []
    report = [
        "# Complete Ablation Multi-seed Report",
        f"models: {args.models}",
        f"target_cities: {args.target_cities}",
        f"source_cities: {args.source_cities}",
        f"seeds: {args.seeds}",
        f"otc_gammas: {args.otc_gammas}",
        f"rd_betas: {args.rd_betas}",
        f"hr_lambdas: {args.hr_lambdas}",
        "",
    ]

    hr_variants: List[Tuple[str, str, str]] = [
        ("RD-Safe+Cat", "cat_only", "HR signal ablation: only category-region affinity on RD-Safe."),
        ("RD-Safe+RR", "rr_only", "HR signal ablation: only region-region diffusion on RD-Safe."),
        ("RD-Safe+Pop", "pop_only", "HR signal ablation: only region popularity prior on RD-Safe."),
        ("RD-Safe+HR-w/o-Cat", "without_cat", "HR leave-one-out ablation: remove category-region affinity."),
        ("RD-Safe+HR-w/o-RR", "without_rr", "HR leave-one-out ablation: remove region-region diffusion."),
        ("RD-Safe+HR-w/o-Pop", "without_pop", "HR leave-one-out ablation: remove region popularity prior."),
        ("RD-Safe+HR-Full", "full", "Full HR reranking on RD-Safe base."),
    ]

    for seed in args.seeds:
        for model in args.models:
            for target in args.target_cities:
                print(f"[RUN] seed={seed} model={model} target={target}", flush=True)
                sources = [s for s in args.source_cities if s != target]

                train_df = mod.load_pkl(official_root / target / "train.pkl")
                test_df = mod.load_pkl(official_root / target / "test.pkl")
                train_core = mod.load_pkl(internal_root / target / "train_core.pkl")
                valid_df = mod.load_pkl(internal_root / target / "valid.pkl")
                strict_path = strict_root / target / "test_strict_unseen.pkl"
                strict_df = mod.load_pkl(strict_path) if strict_path.exists() else None

                local_path = mod.local_score_path(root, target, model, seed, args.epochs, args.dim)
                mod.require_file(local_path)
                local_raw = np.load(local_path).astype(np.float32, copy=False)
                n_brand, n_region = local_raw.shape

                transfers_raw = {}
                for src in sources:
                    tpath = mod.transfer_path(root, src, target, model, seed)
                    mod.require_file(tpath)
                    transfers_raw[src] = np.load(tpath).astype(np.float32, copy=False)

                local = mod.normalize_score(local_raw, args.score_norm)
                transfers = {src: mod.normalize_score(t, args.score_norm) for src, t in transfers_raw.items()}

                graphs_valid = mod.build_graph_scores(train_core, n_brand, n_region)
                graphs_test = mod.build_graph_scores(train_df, n_brand, n_region)

                local_r, local_n = mod.evaluate_scores(local, train_core, valid_df, args.k, args.mask_train, not args.no_row_weighted)
                local_rec = {
                    "score": local,
                    "valid_recall@20": local_r,
                    "valid_ndcg@20": local_n,
                    "alpha": 0.0,
                    "beta": 0.0,
                    "selected_sources": "",
                    "source_weights": "",
                    "gate": "local",
                }

                otc_rec = mod.tune_original_otc(local, transfers, train_core, valid_df, args)
                rd_rec = mod.tune_rd_safe(local, transfers, train_core, valid_df, args)
                hr_otc = mod.tune_hr(otc_rec["score"], otc_rec["score"], graphs_valid, graphs_test, train_core, valid_df, args)

                main_methods = [
                    ("Target-only", "main_component", local_rec["score"], {
                        "alpha": 0.0, "beta": 0.0, "lambda_cat": 0.0, "lambda_rr": 0.0, "lambda_pop": 0.0,
                        "valid_recall@20": local_rec["valid_recall@20"], "valid_ndcg@20": local_rec["valid_ndcg@20"],
                        "selected_sources": "", "source_weights": "", "gate": "local"
                    }, "Local target-city backbone only."),
                    ("Original-OTC-GW-NoFallback", "main_component", otc_rec["score"], {
                        "alpha": otc_rec["alpha"], "beta": 0.0, "lambda_cat": 0.0, "lambda_rr": 0.0, "lambda_pop": 0.0,
                        "valid_recall@20": otc_rec["valid_recall@20"], "valid_ndcg@20": otc_rec["valid_ndcg@20"],
                        "selected_sources": otc_rec["selected_sources"], "source_weights": "equal", "gate": otc_rec["gate"]
                    }, "Original no-fallback OTC-GW."),
                    ("RD-Safe-OTC-GW", "main_component", rd_rec["score"], {
                        "alpha": rd_rec["alpha"], "beta": rd_rec["beta"], "lambda_cat": 0.0, "lambda_rr": 0.0, "lambda_pop": 0.0,
                        "valid_recall@20": rd_rec["valid_recall@20"], "valid_ndcg@20": rd_rec["valid_ndcg@20"],
                        "selected_sources": rd_rec["selected_sources"], "source_weights": rd_rec.get("source_weights", ""), "gate": rd_rec["gate"]
                    }, "RD-Safe reliability-aware safe transfer."),
                    ("HR-Rerank-OTC-GW", "main_component", hr_otc["score_test"], {
                        "alpha": otc_rec["alpha"], "beta": 0.0,
                        "lambda_cat": hr_otc["lambda_cat"], "lambda_rr": hr_otc["lambda_rr"], "lambda_pop": hr_otc["lambda_pop"],
                        "valid_recall@20": hr_otc["valid_recall@20"], "valid_ndcg@20": hr_otc["valid_ndcg@20"],
                        "selected_sources": otc_rec["selected_sources"], "source_weights": "equal", "gate": hr_otc["gate"]
                    }, "Full HR reranking on Original OTC-GW."),
                ]

                for method, group, score, params, notes in main_methods:
                    add_eval_rows(
                        mod, rows, seed=seed, method=method, group=group, model=model, target=target,
                        score=score, train_df=train_df, official_test_df=test_df, strict_df=strict_df,
                        args=args, params={**params, "score_norm": args.score_norm}, notes=notes
                    )
                    param_rows.append({"seed": seed, "group": group, "method": method, "model": model, "target_city": target, **params})

                # HR signal variants on top of RD-Safe.
                for method_name, variant, notes in hr_variants:
                    gv = graph_variant(graphs_valid, variant)
                    gt = graph_variant(graphs_test, variant)
                    hr = mod.tune_hr(rd_rec["score"], rd_rec["score"], gv, gt, train_core, valid_df, args)
                    add_eval_rows(
                        mod, rows, seed=seed, method=method_name, group="hr_signal_ablation", model=model, target=target,
                        score=hr["score_test"], train_df=train_df, official_test_df=test_df, strict_df=strict_df,
                        args=args,
                        params={
                            "alpha": rd_rec["alpha"],
                            "beta": rd_rec["beta"],
                            "lambda_cat": hr["lambda_cat"],
                            "lambda_rr": hr["lambda_rr"],
                            "lambda_pop": hr["lambda_pop"],
                            "valid_recall@20": hr["valid_recall@20"],
                            "valid_ndcg@20": hr["valid_ndcg@20"],
                            "selected_sources": rd_rec["selected_sources"],
                            "source_weights": rd_rec.get("source_weights", ""),
                            "gate": hr["gate"],
                            "score_norm": args.score_norm,
                        },
                        notes=notes
                    )
                    param_rows.append({
                        "seed": seed, "group": "hr_signal_ablation", "method": method_name, "model": model, "target_city": target,
                        "variant": variant,
                        "alpha": rd_rec["alpha"], "beta": rd_rec["beta"],
                        "lambda_cat": hr["lambda_cat"], "lambda_rr": hr["lambda_rr"], "lambda_pop": hr["lambda_pop"],
                        "valid_recall@20": hr["valid_recall@20"], "valid_ndcg@20": hr["valid_ndcg@20"],
                        "selected_sources": rd_rec["selected_sources"],
                        "source_weights": rd_rec.get("source_weights", ""),
                        "gate": hr["gate"],
                    })

                report.append(
                    f"[OK] seed={seed} model={model} target={target} "
                    f"local_N={local_n:.6f} otc_N={otc_rec['valid_ndcg@20']:.6f} rd_N={rd_rec['valid_ndcg@20']:.6f}"
                )

    df = pd.DataFrame(rows)
    params_df = pd.DataFrame(param_rows)
    summary = summarize(df)
    avg = average_official(df)

    raw_path = raw_dir / "complete_ablation_multiseed.csv"
    params_path = raw_dir / "complete_ablation_params_multiseed.csv"
    summary_path = table_dir / "complete_ablation_multiseed_mean_std.csv"
    avg_path = table_dir / "complete_ablation_official_avg.csv"

    df.to_csv(raw_path, index=False, encoding="utf-8-sig")
    params_df.to_csv(params_path, index=False, encoding="utf-8-sig")
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    avg.to_csv(avg_path, index=False, encoding="utf-8-sig")

    report.append("")
    report.append(f"raw_csv: {raw_path}")
    report.append(f"params_csv: {params_path}")
    report.append(f"summary_csv: {summary_path}")
    report.append(f"official_avg_csv: {avg_path}")

    report_path = report_dir / "complete_ablation_multiseed_report.txt"
    report_path.write_text("\n".join(report), encoding="utf-8")

    config = {
        "script": "exp_clean/scripts/04_ablation/run_complete_ablation_multiseed.py",
        "dependency": "exp_clean/scripts/03_main_methods/run_otc_gw_rd_safe_hr_rerank.py",
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
        "official_files_modified": False,
        "outputs": {
            "raw_csv": str(raw_path),
            "params_csv": str(params_path),
            "summary_csv": str(summary_path),
            "official_avg_csv": str(avg_path),
            "report": str(report_path),
        },
        "ablation_methods": [
            "Target-only",
            "Original-OTC-GW-NoFallback",
            "RD-Safe-OTC-GW",
            "HR-Rerank-OTC-GW",
            "RD-Safe+Cat",
            "RD-Safe+RR",
            "RD-Safe+Pop",
            "RD-Safe+HR-w/o-Cat",
            "RD-Safe+HR-w/o-RR",
            "RD-Safe+HR-w/o-Pop",
            "RD-Safe+HR-Full",
        ],
    }
    config_path = config_dir / "complete_ablation_multiseed_config.json"
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[OK] {raw_path}")
    print(f"[OK] {params_path}")
    print(f"[OK] {summary_path}")
    print(f"[OK] {avg_path}")
    print(f"[OK] {report_path}")
    print(f"[OK] {config_path}")


if __name__ == "__main__":
    main()
