# exp_clean/scripts/02_otc_reproduction/run_otc_gw_reproduction.py
# Purpose:
#   A closer OTC reproduction using Gromov-Wasserstein OT plans.
#
# Key differences from OTC-fast:
#   1. Uses GW/entropic-GW transport over intra-city embedding distance matrices.
#   2. Projects source brand/region embeddings to target city via barycentric projection:
#          U_s_to_t = T_u.T @ U_s / colsum(T_u)
#          V_s_to_t = T_v.T @ V_s / colsum(T_v)
#      then computes transfer score = U_s_to_t @ V_s_to_t.T.
#   3. Selects gamma on internal valid, evaluates on official test and strict unseen test.
#
# Notes:
#   - Requires POT package: import ot
#     If missing: pip install POT
#   - This script does not modify official OpenSiteRec files.
#   - It still uses our external evaluator, so do not mix its values with official main.py
#     metrics without explanation.
#
# Outputs:
#   exp_clean/results/raw/otc_gw_reproduction.csv
#   exp_clean/results/reports/otc_gw_reproduction_report.txt
#   exp_clean/configs/otc_gw_reproduction_config.json
#   exp_clean/scores/transfer_scores/*_otc_gw_transfer.npy
#   exp_clean/scores/transfer_scores/*_gw_brand_plan.npy
#   exp_clean/scores/transfer_scores/*_gw_region_plan.npy

import argparse
import json
import math
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


CITIES = ["Chicago", "NYC", "Singapore", "Tokyo"]
MODELS = ["VanillaMF", "LightGCN"]


def project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_pkl(path: Path) -> pd.DataFrame:
    with open(path, "rb") as f:
        obj = pickle.load(f)
    if not isinstance(obj, pd.DataFrame):
        raise TypeError(f"{path} is not a pandas DataFrame, got {type(obj)}")
    for col in ["Brand_ID", "Region_ID"]:
        if col not in obj.columns:
            raise ValueError(f"{path} missing column {col}")
    return obj.copy()


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(path)


def artifact_paths(root: Path, city: str, model: str, seed: int, epochs: int, dim: int) -> Dict[str, Path]:
    prefix = f"{city}_{model}_seed{seed}_epochs{epochs}_dim{dim}"
    return {
        "local_score": root / "exp_clean" / "scores" / "local_scores" / f"{prefix}_local_scores.npy",
        "brand_emb": root / "exp_clean" / "checkpoints" / "baselines" / f"{prefix}_brand_embeddings.npy",
        "region_emb": root / "exp_clean" / "checkpoints" / "baselines" / f"{prefix}_region_embeddings.npy",
        "metadata": root / "exp_clean" / "checkpoints" / "baselines" / f"{prefix}_metadata.json",
    }


def topk_indices(scores: np.ndarray, k: int) -> np.ndarray:
    k_eff = min(k, scores.size)
    idx = np.argpartition(-scores, kth=k_eff - 1)[:k_eff]
    return idx[np.argsort(-scores[idx])]


def evaluate_scores(
    score: np.ndarray,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    k: int = 20,
    mask_train: bool = False,
    row_weighted: bool = True,
) -> Tuple[float, float, int, int]:
    score = np.asarray(score)
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
        train_regions_by_brand = train_df.groupby("Brand_ID")["Region_ID"].apply(list).to_dict()
    else:
        train_regions_by_brand = {}

    recalls, ndcgs = [], []
    evaluated_brands, positive_count = 0, 0
    num_brands, num_regions = score.shape

    for b_raw, regs_raw in grouped.items():
        b = int(b_raw)
        if b < 0 or b >= num_brands:
            continue

        positives = [int(r) for r in regs_raw if 0 <= int(r) < num_regions]
        if not positives:
            continue

        s = score[b].astype(np.float64).copy()
        if mask_train:
            for r_raw in train_regions_by_brand.get(b, []):
                r = int(r_raw)
                if 0 <= r < num_regions:
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
        evaluated_brands += 1
        positive_count += len(positives)

    if not recalls:
        return float("nan"), float("nan"), 0, 0
    return float(np.mean(recalls)), float(np.mean(ndcgs)), evaluated_brands, positive_count


def l2_normalize_rows(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), eps)


def intra_city_distance_matrix(emb: np.ndarray, normalize_emb: bool = True, eps: float = 1e-12) -> np.ndarray:
    x = np.asarray(emb, dtype=np.float64)
    if normalize_emb:
        x = l2_normalize_rows(x)
    sq = np.sum(x * x, axis=1, keepdims=True)
    dist = sq + sq.T - 2.0 * (x @ x.T)
    dist = np.maximum(dist, 0.0)
    # Paper says L2 distance; sqrt is closer to L2 than squared L2.
    dist = np.sqrt(dist + eps)
    maxv = float(np.nanmax(dist))
    if maxv > 0:
        dist = dist / maxv
    return dist.astype(np.float64)


def load_pot_module():
    try:
        import ot  # type: ignore
        return ot
    except Exception as e:
        raise ImportError(
            "POT package is required for GW OTC. Install it in your environment with: pip install POT"
        ) from e


def gw_transport_plan(
    source_emb: np.ndarray,
    target_emb: np.ndarray,
    epsilon: float,
    max_iter: int,
    normalize_emb: bool,
    method: str,
    random_state: int,
) -> np.ndarray:
    ot = load_pot_module()

    C1 = intra_city_distance_matrix(source_emb, normalize_emb=normalize_emb)
    C2 = intra_city_distance_matrix(target_emb, normalize_emb=normalize_emb)

    n, m = C1.shape[0], C2.shape[0]
    p = np.full(n, 1.0 / n, dtype=np.float64)
    q = np.full(m, 1.0 / m, dtype=np.float64)

    if method == "entropic_gw":
        # POT versions differ slightly; keep arguments conservative.
        T = ot.gromov.entropic_gromov_wasserstein(
            C1,
            C2,
            p,
            q,
            loss_fun="square_loss",
            epsilon=epsilon,
            max_iter=max_iter,
            verbose=False,
            log=False,
        )
    elif method == "gw":
        T = ot.gromov.gromov_wasserstein(
            C1,
            C2,
            p,
            q,
            loss_fun="square_loss",
            max_iter=max_iter,
            verbose=False,
            log=False,
        )
    else:
        raise ValueError(f"Unknown GW method: {method}")

    T = np.asarray(T, dtype=np.float64)
    s = float(T.sum())
    if s <= 0 or not np.isfinite(s):
        raise FloatingPointError("Invalid GW transport plan.")
    return T / s


def barycentric_project(source_emb: np.ndarray, T_source_to_target: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    # POT plan has column sums equal target marginal q_j=1/M_t.
    # Eq. (6)(7) says weighted sum; for a proper weighted average per target entity,
    # normalize each target column.
    colsum = np.sum(T_source_to_target, axis=0, keepdims=True).T
    return (T_source_to_target.T @ source_emb) / np.maximum(colsum, eps)


def zscore_global(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return (x - float(np.mean(x))) / max(float(np.std(x)), eps)


def zscore_rows(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return (x - np.mean(x, axis=1, keepdims=True)) / np.maximum(np.std(x, axis=1, keepdims=True), eps)


def normalize_score(x: np.ndarray, mode: str) -> np.ndarray:
    if mode == "none":
        return np.asarray(x, dtype=np.float64)
    if mode == "global":
        return zscore_global(x)
    if mode == "row":
        return zscore_rows(x)
    raise ValueError(f"Unknown score_norm: {mode}")


def compute_transfer_score_gw(
    source_brand_emb: np.ndarray,
    source_region_emb: np.ndarray,
    target_brand_emb: np.ndarray,
    target_region_emb: np.ndarray,
    epsilon: float,
    max_iter: int,
    normalize_emb: bool,
    method: str,
    random_state: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    Tb = gw_transport_plan(
        source_brand_emb,
        target_brand_emb,
        epsilon=epsilon,
        max_iter=max_iter,
        normalize_emb=normalize_emb,
        method=method,
        random_state=random_state,
    )
    Tr = gw_transport_plan(
        source_region_emb,
        target_region_emb,
        epsilon=epsilon,
        max_iter=max_iter,
        normalize_emb=normalize_emb,
        method=method,
        random_state=random_state,
    )

    U_proj = barycentric_project(source_brand_emb.astype(np.float64), Tb)
    V_proj = barycentric_project(source_region_emb.astype(np.float64), Tr)
    transfer_score = U_proj @ V_proj.T
    return transfer_score, Tb, Tr


def append_result(
    rows: List[dict],
    method: str,
    model: str,
    target: str,
    source_cities: List[str],
    gamma: float,
    split_name: str,
    score: np.ndarray,
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    k: int,
    mask_train: bool,
    row_weighted: bool,
    notes: str,
) -> None:
    recall, ndcg, eb, pc = evaluate_scores(
        score,
        train_df=train_df,
        test_df=eval_df,
        k=k,
        mask_train=mask_train,
        row_weighted=row_weighted,
    )
    rows.append({
        "method": method,
        "model": model,
        "target_city": target,
        "source_cities": "|".join(source_cities),
        "gamma": gamma,
        "eval_split": split_name,
        "recall@20": recall,
        "ndcg@20": ndcg,
        "evaluated_brands": eb,
        "positive_count": pc,
        "status": "success",
        "notes": notes,
    })


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", default=["VanillaMF"], choices=MODELS)
    p.add_argument("--target-cities", nargs="+", default=["Chicago"], choices=CITIES)
    p.add_argument("--source-cities", nargs="+", default=["NYC"], choices=CITIES)
    p.add_argument("--seed", type=int, default=2024)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--dim", type=int, default=64)
    p.add_argument("--k", type=int, default=20)
    p.add_argument("--gammas", nargs="+", type=float, default=[0.0, 0.1, 0.5, 1.0, 2.0, 5.0])
    p.add_argument("--gw-method", choices=["entropic_gw", "gw"], default="entropic_gw")
    p.add_argument("--gw-epsilon", type=float, default=0.01)
    p.add_argument("--gw-max-iter", type=int, default=80)
    p.add_argument("--score-norm", choices=["none", "global", "row"], default="none")
    p.add_argument("--normalize-emb", action="store_true")
    p.add_argument("--mask-train", action="store_true")
    p.add_argument("--no-row-weighted", action="store_true")
    p.add_argument("--save-plans", action="store_true")
    p.add_argument("--reuse-cache", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    root = project_root()

    official_root = root / "exp_clean" / "data_splits" / "official"
    internal_root = root / "exp_clean" / "data_splits" / "internal_valid"
    strict_root = root / "exp_clean" / "data_splits" / "strict_unseen"

    raw_dir = root / "exp_clean" / "results" / "raw"
    report_dir = root / "exp_clean" / "results" / "reports"
    config_dir = root / "exp_clean" / "configs"
    transfer_dir = root / "exp_clean" / "scores" / "transfer_scores"

    for d in [raw_dir, report_dir, config_dir, transfer_dir]:
        ensure_dir(d)

    # Fail fast if POT missing.
    load_pot_module()

    report = [
        "# OTC-GW Reproduction Report",
        f"models: {args.models}",
        f"target_cities: {args.target_cities}",
        f"source_cities: {args.source_cities}",
        f"gammas: {args.gammas}",
        f"gw_method: {args.gw_method}",
        f"gw_epsilon: {args.gw_epsilon}",
        f"gw_max_iter: {args.gw_max_iter}",
        f"score_norm: {args.score_norm}",
        f"normalize_emb: {args.normalize_emb}",
        f"mask_train: {args.mask_train}",
        f"row_weighted: {not args.no_row_weighted}",
        "",
    ]

    rows = []

    for model in args.models:
        cache = {}
        for city in CITIES:
            paths = artifact_paths(root, city, model, args.seed, args.epochs, args.dim)
            for p in paths.values():
                require_file(p)
            cache[city] = {
                "local_score": np.load(paths["local_score"]),
                "brand_emb": np.load(paths["brand_emb"]),
                "region_emb": np.load(paths["region_emb"]),
            }

        for target in args.target_cities:
            source_list = [s for s in args.source_cities if s != target]
            if not source_list:
                report.append(f"[WARN] target={target}, model={model}: no valid source city after removing target.")
                continue

            official_train = load_pkl(official_root / target / "train.pkl")
            official_test = load_pkl(official_root / target / "test.pkl")
            train_core = load_pkl(internal_root / target / "train_core.pkl")
            valid_df = load_pkl(internal_root / target / "valid.pkl")
            strict_path = strict_root / target / "test_strict_unseen.pkl"
            strict_df = load_pkl(strict_path) if strict_path.exists() else None

            local_score = cache[target]["local_score"].astype(np.float64)
            local_for_fusion = normalize_score(local_score, args.score_norm)

            append_result(
                rows,
                method="Local",
                model=model,
                target=target,
                source_cities=[],
                gamma=0.0,
                split_name="official_test",
                score=local_for_fusion,
                train_df=official_train,
                eval_df=official_test,
                k=args.k,
                mask_train=args.mask_train,
                row_weighted=not args.no_row_weighted,
                notes="External evaluator on local score under same score_norm as OTC-GW.",
            )
            if strict_df is not None:
                append_result(
                    rows,
                    method="Local",
                    model=model,
                    target=target,
                    source_cities=[],
                    gamma=0.0,
                    split_name="strict_unseen_test",
                    score=local_for_fusion,
                    train_df=official_train,
                    eval_df=strict_df,
                    k=args.k,
                    mask_train=args.mask_train,
                    row_weighted=not args.no_row_weighted,
                    notes="Strict unseen-pair evaluation.",
                )

            transfer_scores = []
            for source in source_list:
                prefix = f"{source}_to_{target}_{model}_seed{args.seed}_gw"
                transfer_path = transfer_dir / f"{prefix}_transfer.npy"
                brand_plan_path = transfer_dir / f"{prefix}_brand_plan.npy"
                region_plan_path = transfer_dir / f"{prefix}_region_plan.npy"

                if args.reuse_cache and transfer_path.exists():
                    transfer_score = np.load(transfer_path)
                    report.append(f"[CACHE] {source}->{target} {model}: {transfer_path}")
                else:
                    report.append(f"[COMPUTE] {source}->{target} {model}: GW brand/region plans")
                    transfer_score, Tb, Tr = compute_transfer_score_gw(
                        source_brand_emb=cache[source]["brand_emb"],
                        source_region_emb=cache[source]["region_emb"],
                        target_brand_emb=cache[target]["brand_emb"],
                        target_region_emb=cache[target]["region_emb"],
                        epsilon=args.gw_epsilon,
                        max_iter=args.gw_max_iter,
                        normalize_emb=args.normalize_emb,
                        method=args.gw_method,
                        random_state=args.seed,
                    )
                    np.save(transfer_path, transfer_score.astype(np.float32))
                    if args.save_plans:
                        np.save(brand_plan_path, Tb.astype(np.float32))
                        np.save(region_plan_path, Tr.astype(np.float32))

                transfer_scores.append(normalize_score(transfer_score, args.score_norm))

            transfer_mean = np.mean(np.stack(transfer_scores, axis=0), axis=0)

            # Select gamma on internal valid.
            best = None
            for gamma in args.gammas:
                final_score = local_for_fusion + gamma * transfer_mean
                valid_recall, valid_ndcg, _, _ = evaluate_scores(
                    final_score,
                    train_df=train_core,
                    test_df=valid_df,
                    k=args.k,
                    mask_train=args.mask_train,
                    row_weighted=not args.no_row_weighted,
                )
                if best is None or valid_ndcg > best["valid_ndcg"]:
                    best = {
                        "gamma": gamma,
                        "valid_recall": valid_recall,
                        "valid_ndcg": valid_ndcg,
                        "final_score": final_score,
                    }

            append_result(
                rows,
                method="OTC-GW",
                model=model,
                target=target,
                source_cities=source_list,
                gamma=float(best["gamma"]),
                split_name="official_test",
                score=best["final_score"],
                train_df=official_train,
                eval_df=official_test,
                k=args.k,
                mask_train=args.mask_train,
                row_weighted=not args.no_row_weighted,
                notes=f"GW OTC; gamma selected on internal valid; valid_ndcg={best['valid_ndcg']:.6f}.",
            )
            if strict_df is not None:
                append_result(
                    rows,
                    method="OTC-GW",
                    model=model,
                    target=target,
                    source_cities=source_list,
                    gamma=float(best["gamma"]),
                    split_name="strict_unseen_test",
                    score=best["final_score"],
                    train_df=official_train,
                    eval_df=strict_df,
                    k=args.k,
                    mask_train=args.mask_train,
                    row_weighted=not args.no_row_weighted,
                    notes=f"Strict unseen-pair; valid_ndcg={best['valid_ndcg']:.6f}.",
                )

            report.append(
                f"{target} | {model} | sources={source_list} | best_gamma={best['gamma']} | valid_R={best['valid_recall']:.4f} | valid_N={best['valid_ndcg']:.4f}"
            )

    out_csv = raw_dir / "otc_gw_reproduction.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False, encoding="utf-8-sig")

    report.append("")
    report.append(f"csv: {out_csv}")
    report.append("Note: This is closer to paper OTC than OTC-fast, but still uses current exported baseline artifacts and external evaluator.")
    report_path = report_dir / "otc_gw_reproduction_report.txt"
    report_path.write_text("\n".join(report), encoding="utf-8")

    config = {
        "script": "exp_clean/scripts/02_otc_reproduction/run_otc_gw_reproduction.py",
        "seed": args.seed,
        "epochs": args.epochs,
        "dim": args.dim,
        "k": args.k,
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
        "save_plans": args.save_plans,
        "reuse_cache": args.reuse_cache,
        "official_files_modified": False,
        "outputs": {
            "csv": str(out_csv),
            "report": str(report_path),
            "transfer_dir": str(transfer_dir),
        },
        "note": "GW OTC with barycentric projection and valid-selected gamma.",
    }
    config_path = config_dir / "otc_gw_reproduction_config.json"
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[OK] Wrote {out_csv}")
    print(f"[OK] Wrote {report_path}")
    print(f"[OK] Wrote {config_path}")


if __name__ == "__main__":
    main()
