# exp_clean/scripts/00_data_checks/create_internal_valid_splits.py
# Purpose:
#   Create an internal validation split from the official OpenSiteRec train.pkl files.
#   The official test.pkl files are NOT modified.
#
# Protocol:
#   official train/test is treated as the outer split, approximately 8:2.
#   internal valid is sampled from official train by Brand_ID.
#   valid_ratio_in_official_train = 0.125, so the overall split is approximately 7:1:2.
#
# Outputs:
#   exp_clean/data_splits/internal_valid/{City}/train_core.pkl
#   exp_clean/data_splits/internal_valid/{City}/valid.pkl
#   exp_clean/results/raw/internal_valid_split_stats.csv
#   exp_clean/results/reports/internal_valid_split_report.txt
#   exp_clean/configs/internal_valid_split_config.json

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
OFFICIAL_ROOT = PROJECT_ROOT / "exp_clean" / "data_splits" / "official"
INTERNAL_ROOT = PROJECT_ROOT / "exp_clean" / "data_splits" / "internal_valid"
RAW_OUT_DIR = PROJECT_ROOT / "exp_clean" / "results" / "raw"
REPORT_OUT_DIR = PROJECT_ROOT / "exp_clean" / "results" / "reports"
CONFIG_OUT_DIR = PROJECT_ROOT / "exp_clean" / "configs"

CITIES = ["Chicago", "NYC", "Singapore", "Tokyo"]
REQUIRED_COLUMNS = ["Brand_ID", "Cate1_ID", "Cate2_ID", "Cate3_ID", "Region_ID"]
SEED = 2024
VALID_RATIO_IN_OFFICIAL_TRAIN = 0.125


def ensure_dirs() -> None:
    INTERNAL_ROOT.mkdir(parents=True, exist_ok=True)
    RAW_OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_OUT_DIR.mkdir(parents=True, exist_ok=True)
    CONFIG_OUT_DIR.mkdir(parents=True, exist_ok=True)
    for city in CITIES:
        (INTERNAL_ROOT / city).mkdir(parents=True, exist_ok=True)


def load_pickle(path: Path):
    with path.open("rb") as f:
        return pickle.load(f)


def save_pickle(obj, path: Path) -> None:
    with path.open("wb") as f:
        pickle.dump(obj, f)


def check_dataframe(df: pd.DataFrame, path: Path) -> None:
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"{path} is not a pandas DataFrame. Got: {type(df)}")
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{path} missing required columns: {missing}")


def pair_set(df: pd.DataFrame) -> set:
    return set(zip(df["Brand_ID"].tolist(), df["Region_ID"].tolist()))


def split_one_city(train_df: pd.DataFrame, city: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_indices = []
    valid_indices = []

    # Sort Brand_ID for deterministic traversal. Sampling itself is random but seed-controlled.
    for brand_id, group in train_df.groupby("Brand_ID", sort=True):
        indices = group.index.to_numpy()
        # Brand-specific seed keeps result stable even if group traversal changes.
        rng = np.random.default_rng(SEED + 1009 * CITIES.index(city) + int(brand_id))
        shuffled = indices.copy()
        rng.shuffle(shuffled)

        n = len(shuffled)
        valid_n = int(round(n * VALID_RATIO_IN_OFFICIAL_TRAIN))
        valid_n = max(1, valid_n)
        valid_n = min(valid_n, n - 1)  # keep at least one training row per brand

        valid_indices.extend(shuffled[:valid_n].tolist())
        train_indices.extend(shuffled[valid_n:].tolist())

    train_core = train_df.loc[sorted(train_indices)].copy()
    valid = train_df.loc[sorted(valid_indices)].copy()
    return train_core, valid


def summarize_city(city: str, official_train: pd.DataFrame, train_core: pd.DataFrame, valid: pd.DataFrame, official_test: pd.DataFrame) -> dict:
    train_core_pairs = pair_set(train_core)
    valid_pairs = pair_set(valid)
    test_pairs = pair_set(official_test)

    valid_overlap_pairs = train_core_pairs & valid_pairs
    test_overlap_pairs = pair_set(official_train) & test_pairs

    train_core_brand_counts = train_core.groupby("Brand_ID").size()
    valid_brand_counts = valid.groupby("Brand_ID").size()

    return {
        "city": city,
        "official_train_rows": len(official_train),
        "train_core_rows": len(train_core),
        "valid_rows": len(valid),
        "official_test_rows": len(official_test),
        "valid_ratio_in_official_train_actual": len(valid) / len(official_train),
        "official_train_unique_brands": official_train["Brand_ID"].nunique(),
        "train_core_unique_brands": train_core["Brand_ID"].nunique(),
        "valid_unique_brands": valid["Brand_ID"].nunique(),
        "official_test_unique_brands": official_test["Brand_ID"].nunique(),
        "train_core_unique_regions": train_core["Region_ID"].nunique(),
        "valid_unique_regions": valid["Region_ID"].nunique(),
        "official_test_unique_regions": official_test["Region_ID"].nunique(),
        "train_core_unique_pairs": len(train_core_pairs),
        "valid_unique_pairs": len(valid_pairs),
        "official_test_unique_pairs": len(test_pairs),
        "valid_pairs_seen_in_train_core": len(valid_overlap_pairs),
        "valid_pair_overlap_ratio": len(valid_overlap_pairs) / len(valid_pairs) if valid_pairs else 0.0,
        "official_test_pairs_seen_in_official_train": len(test_overlap_pairs),
        "official_test_pair_overlap_ratio": len(test_overlap_pairs) / len(test_pairs) if test_pairs else 0.0,
        "min_train_core_rows_per_brand": int(train_core_brand_counts.min()),
        "median_train_core_rows_per_brand": float(train_core_brand_counts.median()),
        "mean_train_core_rows_per_brand": float(train_core_brand_counts.mean()),
        "max_train_core_rows_per_brand": int(train_core_brand_counts.max()),
        "min_valid_rows_per_brand": int(valid_brand_counts.min()),
        "median_valid_rows_per_brand": float(valid_brand_counts.median()),
        "mean_valid_rows_per_brand": float(valid_brand_counts.mean()),
        "max_valid_rows_per_brand": int(valid_brand_counts.max()),
    }


def main() -> None:
    ensure_dirs()

    config = {
        "script": "exp_clean/scripts/00_data_checks/create_internal_valid_splits.py",
        "seed": SEED,
        "valid_ratio_in_official_train": VALID_RATIO_IN_OFFICIAL_TRAIN,
        "cities": CITIES,
        "rule": "For each Brand_ID, sample max(1, round(n_train_rows * ratio)) rows into valid and keep at least one row in train_core.",
        "note": "official test.pkl is not modified. This split is for validation/model selection/safe gate only.",
    }
    (CONFIG_OUT_DIR / "internal_valid_split_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    rows = []
    report_lines = []
    report_lines.append("# Internal Validation Split Report")
    report_lines.append("")
    report_lines.append(f"PROJECT_ROOT: {PROJECT_ROOT}")
    report_lines.append(f"OFFICIAL_ROOT: {OFFICIAL_ROOT}")
    report_lines.append(f"INTERNAL_ROOT: {INTERNAL_ROOT}")
    report_lines.append(f"SEED: {SEED}")
    report_lines.append(f"VALID_RATIO_IN_OFFICIAL_TRAIN: {VALID_RATIO_IN_OFFICIAL_TRAIN}")
    report_lines.append("")

    for city in CITIES:
        train_path = OFFICIAL_ROOT / city / "train.pkl"
        test_path = OFFICIAL_ROOT / city / "test.pkl"
        if not train_path.exists():
            raise FileNotFoundError(f"Missing official train file: {train_path}")
        if not test_path.exists():
            raise FileNotFoundError(f"Missing official test file: {test_path}")

        official_train = load_pickle(train_path)
        official_test = load_pickle(test_path)
        check_dataframe(official_train, train_path)
        check_dataframe(official_test, test_path)

        train_core, valid = split_one_city(official_train, city)

        city_out_dir = INTERNAL_ROOT / city
        save_pickle(train_core, city_out_dir / "train_core.pkl")
        save_pickle(valid, city_out_dir / "valid.pkl")

        summary = summarize_city(city, official_train, train_core, valid, official_test)
        rows.append(summary)

        report_lines.append("=" * 90)
        report_lines.append(f"CITY: {city}")
        for k, v in summary.items():
            report_lines.append(f"{k}: {v}")
        report_lines.append(f"saved_train_core: {city_out_dir / 'train_core.pkl'}")
        report_lines.append(f"saved_valid: {city_out_dir / 'valid.pkl'}")
        report_lines.append("")

    stats_df = pd.DataFrame(rows)
    stats_path = RAW_OUT_DIR / "internal_valid_split_stats.csv"
    report_path = REPORT_OUT_DIR / "internal_valid_split_report.txt"
    stats_df.to_csv(stats_path, index=False, encoding="utf-8-sig")
    report_path.write_text("\n".join(report_lines), encoding="utf-8")

    print(f"[OK] Internal validation splits written to: {INTERNAL_ROOT}")
    print(f"[OK] Stats written to: {stats_path}")
    print(f"[OK] Report written to: {report_path}")


if __name__ == "__main__":
    main()
