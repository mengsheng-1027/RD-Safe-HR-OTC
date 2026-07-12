# exp_clean/scripts/03_main_methods/run_otc_gw_rd_safe_hr_rerank.py
# Purpose:
#   Add heterogeneous-graph reranking on top of OTC-GW / RD-Safe OTC-GW.
#
# Final methods:
#   1. Target-only
#   2. Original-OTC-GW-NoFallback
#   3. RD-Safe-OTC-GW
#   4. HR-Rerank-OTC-GW
#   5. RD-Safe+HR-Rerank-OTC-GW
#
# Heterogeneous graph signals:
#   - category-region affinity: brands with similar category prefer similar regions
#   - region-region diffusion: candidate regions related to the brand's known regions
#   - region popularity prior: globally popular regions in the target city
#
# Safety:
#   HR reranking is selected by internal valid.
#   If graph reranking does not improve valid nDCG enough or reduces valid Recall,
#   it falls back to the corresponding base score.
#
# Outputs:
#   exp_clean/results/raw/otc_gw_rd_safe_hr_rerank.csv
#   exp_clean/results/raw/otc_gw_hr_params.csv
#   exp_clean/results/reports/otc_gw_rd_safe_hr_rerank_report.txt
#   exp_clean/configs/otc_gw_rd_safe_hr_rerank_config.json

import argparse
import json
import math
import pickle
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


CITIES = ["Chicago", "NYC", "Singapore", "Tokyo"]
MODELS = ["VanillaMF", "LightGCN"]


def project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def load_pkl(path: Path) -> pd.DataFrame:
    with open(path, "rb") as f:
        obj = pickle.load(f)
    if not isinstance(obj, pd.DataFrame):
        raise TypeError(f"{path} is not a pandas DataFrame")
    return obj.copy()


def local_score_path(root: Path, city: str, model: str, seed: int, epochs: int, dim: int) -> Path:
    return root / "exp_clean" / "scores" / "local_scores" / f"{city}_{model}_seed{seed}_epochs{epochs}_dim{dim}_local_scores.npy"


def transfer_path(root: Path, source: str, target: str, model: str, seed: int) -> Path:
    return root / "exp_clean" / "scores" / "transfer_scores" / f"{source}_to_{target}_{model}_seed{seed}_gw_transfer.npy"


def require_file(path: Path):
    if not path.exists():
        raise FileNotFoundError(path)


def zscore_global(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return ((x - float(x.mean())) / max(float(x.std()), eps)).astype(np.float32, copy=False)


def zscore_rows(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    mean = x.mean(axis=1, keepdims=True)
    std = np.maximum(x.std(axis=1, keepdims=True), eps)
    return ((x - mean) / std).astype(np.float32, copy=False)


def normalize_score(x: np.ndarray, mode: str) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if mode == "none":
        return x.copy()
    if mode == "global":
        return zscore_global(x)
    if mode == "row":
        return zscore_rows(x)
    raise ValueError(mode)


def topk_indices(scores: np.ndarray, k: int) -> np.ndarray:
    k_eff = min(k, scores.size)
    idx = np.argpartition(-scores, kth=k_eff - 1)[:k_eff]
    return idx[np.argsort(-scores[idx])]


def evaluate_scores(
    score: np.ndarray,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    k: int,
    mask_train: bool,
    row_weighted: bool,
):
    if row_weighted:
        grouped = test_df.groupby("Brand_ID")["Region_ID"].apply(list).to_dict()
    else:
        grouped = (
            test_df.drop_duplicates(["Brand_ID", "Region_ID"])
            .groupby("Brand_ID")["Region_ID"]
            .apply(list)
            .to_dict()
        )

    if mask_train:
        train_by_brand = train_df.groupby("Brand_ID")["Region_ID"].apply(list).to_dict()
    else:
        train_by_brand = {}

    recalls, ndcgs = [], []
    n_brand, n_region = score.shape

    for b_raw, regs_raw in grouped.items():
        b = int(b_raw)
        if b < 0 or b >= n_brand:
            continue

        positives = [int(r) for r in regs_raw if 0 <= int(r) < n_region]
        if not positives:
            continue

        s = score[b].astype(np.float64, copy=True)
        if mask_train:
            for r_raw in train_by_brand.get(b, []):
                r = int(r_raw)
                if 0 <= r < n_region:
                    s[r] = -np.inf

        pred = topk_indices(s, k)
        pred_set = set(int(x) for x in pred)
        pred_pos = {int(r): i for i, r in enumerate(pred)}

        hits = 0
        hit_positions = []
        for r in positives:
            if r in pred_set:
                hits += 1
                hit_positions.append(pred_pos[r])

        recall = hits / len(positives)
        ideal_len = min(len(positives), k)
        ideal_dcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_len))
        dcg = sum(1.0 / math.log2(pos + 2) for pos in sorted(hit_positions))
        ndcg = dcg / ideal_dcg if ideal_dcg > 0 else 0.0

        recalls.append(recall)
        ndcgs.append(ndcg)

    if not recalls:
        return float("nan"), float("nan")
    return float(np.mean(recalls)), float(np.mean(ndcgs))


def mode_value(series: pd.Series):
    s = series.dropna()
    if len(s) == 0:
        return None
    return s.mode().iloc[0]


def build_graph_scores(
    train_df: pd.DataFrame,
    n_brand: int,
    n_region: int,
    category_cols: List[str] = ["Cate3_ID", "Cate2_ID", "Cate1_ID"],
):
    # Clean bounds.
    df = train_df.copy()
    df["Brand_ID"] = df["Brand_ID"].astype(int)
    df["Region_ID"] = df["Region_ID"].astype(int)
    df = df[(df["Brand_ID"] >= 0) & (df["Brand_ID"] < n_brand) & (df["Region_ID"] >= 0) & (df["Region_ID"] < n_region)]

    # Incidence: brand-region.
    incidence = np.zeros((n_brand, n_region), dtype=np.float32)
    for b, r in df[["Brand_ID", "Region_ID"]].itertuples(index=False):
        incidence[int(b), int(r)] = 1.0

    # Region popularity.
    pop = np.zeros(n_region, dtype=np.float32)
    reg_counts = df["Region_ID"].value_counts()
    for r, c in reg_counts.items():
        if 0 <= int(r) < n_region:
            pop[int(r)] = float(c)
    pop = np.log1p(pop)
    pop_mat = np.tile(pop.reshape(1, -1), (n_brand, 1))
    pop_mat = zscore_rows(pop_mat)

    # Category-region affinity.
    cat_score = np.zeros((n_brand, n_region), dtype=np.float32)
    level_weights = {"Cate3_ID": 1.0, "Cate2_ID": 0.6, "Cate1_ID": 0.3}

    for col in category_cols:
        if col not in df.columns:
            continue

        brand_cat = df.groupby("Brand_ID")[col].agg(mode_value).to_dict()
        cat_region = {}
        for (cat, r), c in df.groupby([col, "Region_ID"]).size().items():
            if pd.isna(cat):
                continue
            cat_region.setdefault(cat, np.zeros(n_region, dtype=np.float32))
            cat_region[cat][int(r)] = float(c)

        for cat, vec in cat_region.items():
            # log + row-like normalization for each category distribution.
            vec[:] = np.log1p(vec)
            std = float(vec.std())
            if std > 1e-12:
                vec[:] = (vec - float(vec.mean())) / std

        w = level_weights.get(col, 0.3)
        for b in range(n_brand):
            cat = brand_cat.get(b, None)
            if cat in cat_region:
                cat_score[b] += w * cat_region[cat]

    cat_score = zscore_rows(cat_score)

    # Region-region diffusion.
    # R-R graph from brand-region incidence: regions connected if shared by brands.
    rr = incidence.T @ incidence
    np.fill_diagonal(rr, 0.0)
    rr = np.log1p(rr).astype(np.float32, copy=False)
    row_sum = rr.sum(axis=1, keepdims=True)
    rr_norm = rr / np.maximum(row_sum, 1e-12)

    rr_score = incidence @ rr_norm
    rr_score = zscore_rows(rr_score)

    return {
        "cat": cat_score.astype(np.float32, copy=False),
        "rr": rr_score.astype(np.float32, copy=False),
        "pop": pop_mat.astype(np.float32, copy=False),
    }


def append_eval_rows(
    rows: List[dict],
    *,
    method: str,
    model: str,
    target: str,
    score: np.ndarray,
    train_df: pd.DataFrame,
    official_test_df: pd.DataFrame,
    strict_df: pd.DataFrame,
    k: int,
    mask_train: bool,
    row_weighted: bool,
    params: dict,
    notes: str,
):
    for split_name, eval_df in [("official_test", official_test_df), ("strict_unseen_test", strict_df)]:
        if eval_df is None:
            continue
        recall, ndcg = evaluate_scores(score, train_df, eval_df, k, mask_train, row_weighted)
        row = {
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


def best_by_ndcg(candidates: List[dict]):
    return max(candidates, key=lambda x: x["valid_ndcg@20"])


def tune_original_otc(local: np.ndarray, transfers: Dict[str, np.ndarray], train_core, valid_df, args):
    equal = np.mean(np.stack(list(transfers.values()), axis=0), axis=0).astype(np.float32, copy=False)
    candidates = []
    for alpha in args.otc_gammas:
        score = (local + alpha * equal).astype(np.float32, copy=False)
        r, n = evaluate_scores(score, train_core, valid_df, args.k, args.mask_train, not args.no_row_weighted)
        candidates.append({
            "score": score,
            "alpha": alpha,
            "beta": 0.0,
            "valid_recall@20": r,
            "valid_ndcg@20": n,
            "selected_sources": "|".join(transfers.keys()),
            "source_weights": "equal",
            "gate": "transfer",
        })
    return best_by_ndcg(candidates)


def tune_rd_safe(local: np.ndarray, transfers: Dict[str, np.ndarray], train_core, valid_df, args):
    local_r, local_n = evaluate_scores(local, train_core, valid_df, args.k, args.mask_train, not args.no_row_weighted)
    local_rec = {
        "score": local,
        "alpha": 0.0,
        "beta": 0.0,
        "valid_recall@20": local_r,
        "valid_ndcg@20": local_n,
        "selected_sources": "",
        "source_weights": "",
        "gate": "local",
    }

    # Original OTC candidate uses no fallback gamma grid.
    otc_rec = tune_original_otc(local, transfers, train_core, valid_df, args)

    # Reliable sources by single-source improvement over local.
    reliable = []
    source_gains = {}
    for src, t in transfers.items():
        best_src = None
        for beta in args.rd_betas:
            score = (local + beta * t).astype(np.float32, copy=False)
            r, n = evaluate_scores(score, train_core, valid_df, args.k, args.mask_train, not args.no_row_weighted)
            rec = {
                "beta": beta,
                "valid_recall@20": r,
                "valid_ndcg@20": n,
                "valid_recall_gain": r - local_r,
                "valid_ndcg_gain": n - local_n,
            }
            if best_src is None or rec["valid_ndcg@20"] > best_src["valid_ndcg@20"]:
                best_src = rec
        source_gains[src] = best_src["valid_ndcg_gain"]
        if best_src["valid_ndcg_gain"] >= args.source_min_ndcg_gain and best_src["valid_recall_gain"] >= -args.recall_tolerance:
            reliable.append((src, max(best_src["valid_ndcg_gain"], 0.0)))

    rd_rec = None
    if len(reliable) >= args.min_reliable_sources:
        vals = np.array([g for _, g in reliable], dtype=np.float32)
        vals = vals / max(float(vals.sum()), 1e-12)
        weights = {src: float(w) for (src, _), w in zip(reliable, vals)}

        rd_transfer = np.zeros_like(local, dtype=np.float32)
        for src, w in weights.items():
            rd_transfer += float(w) * transfers[src]

        candidates = []
        for alpha in [0.0] + list(args.otc_gammas):
            base = local if alpha == 0.0 else local + alpha * np.mean(np.stack(list(transfers.values()), axis=0), axis=0)
            base = base.astype(np.float32, copy=False)
            for beta in args.rd_betas:
                score = (base + beta * rd_transfer).astype(np.float32, copy=False)
                r, n = evaluate_scores(score, train_core, valid_df, args.k, args.mask_train, not args.no_row_weighted)
                candidates.append({
                    "score": score,
                    "alpha": alpha,
                    "beta": beta,
                    "valid_recall@20": r,
                    "valid_ndcg@20": n,
                    "selected_sources": "|".join(weights.keys()),
                    "source_weights": json.dumps(weights, ensure_ascii=False),
                    "gate": "transfer",
                })
        rd_rec = best_by_ndcg(candidates)

    # Safe choose among Local, Original OTC, and RD.
    safe_base = best_by_ndcg([local_rec, otc_rec])
    if rd_rec is not None and rd_rec["valid_ndcg@20"] >= safe_base["valid_ndcg@20"] + args.method_min_ndcg_gain and rd_rec["valid_recall@20"] >= safe_base["valid_recall@20"] - args.recall_tolerance:
        rd_rec["gate"] = "safe_select:RD"
        return rd_rec

    safe_base = dict(safe_base)
    safe_base["gate"] = "safe_select:OTC" if safe_base is otc_rec else "safe_select:Local"
    return safe_base


def tune_hr(base_valid: np.ndarray, base_test: np.ndarray, graphs_valid: Dict[str, np.ndarray], graphs_test: Dict[str, np.ndarray],
            train_core, valid_df, args):
    base_r, base_n = evaluate_scores(base_valid, train_core, valid_df, args.k, args.mask_train, not args.no_row_weighted)
    base_valid_norm = zscore_rows(base_valid)
    base_test_norm = zscore_rows(base_test)

    best = {
        "score_valid": base_valid,
        "score_test": base_test,
        "lambda_cat": 0.0,
        "lambda_rr": 0.0,
        "lambda_pop": 0.0,
        "valid_recall@20": base_r,
        "valid_ndcg@20": base_n,
        "gate": "base",
    }

    for lc in args.hr_lambdas:
        for lr in args.hr_lambdas:
            for lp in args.hr_lambdas:
                if lc == 0.0 and lr == 0.0 and lp == 0.0:
                    continue
                score_v = (
                    base_valid_norm
                    + lc * graphs_valid["cat"]
                    + lr * graphs_valid["rr"]
                    + lp * graphs_valid["pop"]
                ).astype(np.float32, copy=False)
                r, n = evaluate_scores(score_v, train_core, valid_df, args.k, args.mask_train, not args.no_row_weighted)
                if n > best["valid_ndcg@20"]:
                    score_t = (
                        base_test_norm
                        + lc * graphs_test["cat"]
                        + lr * graphs_test["rr"]
                        + lp * graphs_test["pop"]
                    ).astype(np.float32, copy=False)
                    best = {
                        "score_valid": score_v,
                        "score_test": score_t,
                        "lambda_cat": lc,
                        "lambda_rr": lr,
                        "lambda_pop": lp,
                        "valid_recall@20": r,
                        "valid_ndcg@20": n,
                        "gate": "graph_rerank",
                    }

    # Conservative gate.
    if best["valid_ndcg@20"] >= base_n + args.hr_min_ndcg_gain and best["valid_recall@20"] >= base_r - args.recall_tolerance:
        return best

    return {
        "score_valid": base_valid,
        "score_test": base_test,
        "lambda_cat": 0.0,
        "lambda_rr": 0.0,
        "lambda_pop": 0.0,
        "valid_recall@20": base_r,
        "valid_ndcg@20": base_n,
        "gate": "fallback_base",
    }


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", default=MODELS, choices=MODELS)
    p.add_argument("--target-cities", nargs="+", default=CITIES, choices=CITIES)
    p.add_argument("--source-cities", nargs="+", default=CITIES, choices=CITIES)
    p.add_argument("--seed", type=int, default=2024)
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


def main():
    args = parse_args()
    root = project_root()

    official_root = root / "exp_clean" / "data_splits" / "official"
    internal_root = root / "exp_clean" / "data_splits" / "internal_valid"
    strict_root = root / "exp_clean" / "data_splits" / "strict_unseen"

    raw_dir = root / "exp_clean" / "results" / "raw"
    report_dir = root / "exp_clean" / "results" / "reports"
    config_dir = root / "exp_clean" / "configs"
    for d in [raw_dir, report_dir, config_dir]:
        ensure_dir(d)

    rows = []
    param_rows = []
    report = [
        "# OTC-GW RD-Safe HR-Rerank Report",
        f"models: {args.models}",
        f"target_cities: {args.target_cities}",
        f"otc_gammas: {args.otc_gammas}",
        f"hr_lambdas: {args.hr_lambdas}",
        "",
    ]

    for model in args.models:
        for target in args.target_cities:
            print(f"[RUN] model={model}, target={target}", flush=True)
            sources = [s for s in args.source_cities if s != target]

            train_df = load_pkl(official_root / target / "train.pkl")
            test_df = load_pkl(official_root / target / "test.pkl")
            train_core = load_pkl(internal_root / target / "train_core.pkl")
            valid_df = load_pkl(internal_root / target / "valid.pkl")
            strict_path = strict_root / target / "test_strict_unseen.pkl"
            strict_df = load_pkl(strict_path) if strict_path.exists() else None

            local_path = local_score_path(root, target, model, args.seed, args.epochs, args.dim)
            require_file(local_path)
            local_raw = np.load(local_path).astype(np.float32, copy=False)
            n_brand, n_region = local_raw.shape

            transfer_raw = {}
            for src in sources:
                tpath = transfer_path(root, src, target, model, args.seed)
                require_file(tpath)
                transfer_raw[src] = np.load(tpath).astype(np.float32, copy=False)

            local = normalize_score(local_raw, args.score_norm)
            transfers = {src: normalize_score(t, args.score_norm) for src, t in transfer_raw.items()}

            # Graph scores: valid uses train_core; test uses official train.
            graphs_valid = build_graph_scores(train_core, n_brand, n_region)
            graphs_test = build_graph_scores(train_df, n_brand, n_region)

            # Base 1: Local.
            local_r, local_n = evaluate_scores(local, train_core, valid_df, args.k, args.mask_train, not args.no_row_weighted)
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

            # Base 2: Original OTC no fallback.
            otc_rec = tune_original_otc(local, transfers, train_core, valid_df, args)

            # Base 3: RD-safe.
            rd_rec = tune_rd_safe(local, transfers, train_core, valid_df, args)

            # HR over OTC and RD.
            hr_otc = tune_hr(otc_rec["score"], otc_rec["score"], graphs_valid, graphs_test, train_core, valid_df, args)
            hr_rd = tune_hr(rd_rec["score"], rd_rec["score"], graphs_valid, graphs_test, train_core, valid_df, args)

            methods = [
                ("Target-only", local_rec["score"], {
                    "alpha": 0.0, "beta": 0.0, "lambda_cat": 0.0, "lambda_rr": 0.0, "lambda_pop": 0.0,
                    "valid_recall@20": local_rec["valid_recall@20"], "valid_ndcg@20": local_rec["valid_ndcg@20"],
                    "selected_sources": "", "gate": "local"
                }),
                ("Original-OTC-GW-NoFallback", otc_rec["score"], {
                    "alpha": otc_rec["alpha"], "beta": 0.0, "lambda_cat": 0.0, "lambda_rr": 0.0, "lambda_pop": 0.0,
                    "valid_recall@20": otc_rec["valid_recall@20"], "valid_ndcg@20": otc_rec["valid_ndcg@20"],
                    "selected_sources": otc_rec["selected_sources"], "gate": otc_rec["gate"]
                }),
                ("RD-Safe-OTC-GW", rd_rec["score"], {
                    "alpha": rd_rec["alpha"], "beta": rd_rec["beta"], "lambda_cat": 0.0, "lambda_rr": 0.0, "lambda_pop": 0.0,
                    "valid_recall@20": rd_rec["valid_recall@20"], "valid_ndcg@20": rd_rec["valid_ndcg@20"],
                    "selected_sources": rd_rec["selected_sources"], "gate": rd_rec["gate"]
                }),
                ("HR-Rerank-OTC-GW", hr_otc["score_test"], {
                    "alpha": otc_rec["alpha"], "beta": 0.0,
                    "lambda_cat": hr_otc["lambda_cat"], "lambda_rr": hr_otc["lambda_rr"], "lambda_pop": hr_otc["lambda_pop"],
                    "valid_recall@20": hr_otc["valid_recall@20"], "valid_ndcg@20": hr_otc["valid_ndcg@20"],
                    "selected_sources": otc_rec["selected_sources"], "gate": hr_otc["gate"]
                }),
                ("RD-Safe+HR-Rerank-OTC-GW", hr_rd["score_test"], {
                    "alpha": rd_rec["alpha"], "beta": rd_rec["beta"],
                    "lambda_cat": hr_rd["lambda_cat"], "lambda_rr": hr_rd["lambda_rr"], "lambda_pop": hr_rd["lambda_pop"],
                    "valid_recall@20": hr_rd["valid_recall@20"], "valid_ndcg@20": hr_rd["valid_ndcg@20"],
                    "selected_sources": rd_rec["selected_sources"], "gate": hr_rd["gate"]
                }),
            ]

            for method, score, params in methods:
                append_eval_rows(
                    rows,
                    method=method,
                    model=model,
                    target=target,
                    score=score,
                    train_df=train_df,
                    official_test_df=test_df,
                    strict_df=strict_df,
                    k=args.k,
                    mask_train=args.mask_train,
                    row_weighted=not args.no_row_weighted,
                    params={
                        **params,
                        "score_norm": args.score_norm,
                    },
                    notes="Heterogeneous graph reranking with category-region, region-region diffusion, and region popularity.",
                )
                param_rows.append({
                    "model": model,
                    "target_city": target,
                    "method": method,
                    **params,
                    "score_norm": args.score_norm,
                })

            report.append(f"{target} | {model}")
            for method, _, params in methods:
                report.append(
                    f"  {method}: valid_R={params['valid_recall@20']:.6f} valid_N={params['valid_ndcg@20']:.6f} "
                    f"alpha={params['alpha']} beta={params['beta']} "
                    f"l_cat={params['lambda_cat']} l_rr={params['lambda_rr']} l_pop={params['lambda_pop']} "
                    f"gate={params['gate']} sources={params['selected_sources']}"
                )
            report.append("")

    out_csv = raw_dir / "otc_gw_rd_safe_hr_rerank.csv"
    param_csv = raw_dir / "otc_gw_hr_params.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False, encoding="utf-8-sig")
    pd.DataFrame(param_rows).to_csv(param_csv, index=False, encoding="utf-8-sig")

    report.append(f"csv: {out_csv}")
    report.append(f"param_csv: {param_csv}")
    report_path = report_dir / "otc_gw_rd_safe_hr_rerank_report.txt"
    report_path.write_text("\n".join(report), encoding="utf-8")

    config = {
        "script": "exp_clean/scripts/03_main_methods/run_otc_gw_rd_safe_hr_rerank.py",
        "seed": args.seed,
        "epochs": args.epochs,
        "dim": args.dim,
        "k": args.k,
        "models": args.models,
        "target_cities": args.target_cities,
        "source_cities": args.source_cities,
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
            "csv": str(out_csv),
            "param_csv": str(param_csv),
            "report": str(report_path),
        },
        "note": "Adds heterogeneous graph reranking on top of no-fallback OTC-GW and RD-Safe OTC-GW.",
    }
    config_path = config_dir / "otc_gw_rd_safe_hr_rerank_config.json"
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[OK] {out_csv}")
    print(f"[OK] {param_csv}")
    print(f"[OK] {report_path}")
    print(f"[OK] {config_path}")


if __name__ == "__main__":
    main()
