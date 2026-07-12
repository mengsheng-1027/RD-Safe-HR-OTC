# exp_clean/scripts/00_data_checks/create_strict_unseen_test_splits.py
# Purpose:
#   Create strict unseen-pair test files for robustness evaluation.
#   A test row is kept only if its (Brand_ID, Region_ID) pair never appears in the official train split.
#
# Important:
#   This script does NOT modify official train/test files.
#   The official full test remains the main reproduction evaluation set.
#   The strict unseen test is only for robustness/new-pair evaluation.

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
OFFICIAL_ROOT = PROJECT_ROOT / "exp_clean" / "data_splits" / "official"
STRICT_ROOT = PROJECT_ROOT / "exp_clean" / "data_splits" / "strict_unseen"
RAW_OUT_DIR = PROJECT_ROOT / "exp_clean" / "results" / "raw"
REPORT_OUT_DIR = PROJECT_ROOT / "exp_clean" / "results" / "reports"
CONFIG_OUT_DIR = PROJECT_ROOT / "exp_clean" / "configs"

CITIES = ["Chicago", "NYC", "Singapore", "Tokyo"]
REQUIRED_COLUMNS = ["Brand_ID", "Region_ID"]


def load_pickle(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)


def save_pickle(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def ensure_dataframe(obj, path: Path) -> pd.DataFrame:
    if not isinstance(obj, pd.DataFrame):
        raise TypeError(f"Expected pandas.DataFrame from {path}, got {type(obj)}")
    missing = [c for c in REQUIRED_COLUMNS if c not in obj.columns]
    if missing:
        raise ValueError(f"Missing required columns in {path}: {missing}")
    return obj.copy()


def pair_set(df: pd.DataFrame) -> set[Tuple[int, int]]:
    return set(zip(df["Brand_ID"].astype(int), df["Region_ID"].astype(int)))


def make_pair_series(df: pd.DataFrame) -> pd.Series:
    return list(zip(df["Brand_ID"].astype(int), df["Region_ID"].astype(int)))


def audit_city(city: str) -> Dict[str, object]:
    train_path = OFFICIAL_ROOT / city / "train.pkl"
    test_path = OFFICIAL_ROOT / city / "test.pkl"

    if not train_path.exists():
        raise FileNotFoundError(f"Missing official train file: {train_path}")
    if not test_path.exists():
        raise FileNotFoundError(f"Missing official test file: {test_path}")

    train_df = ensure_dataframe(load_pickle(train_path), train_path)
    test_df = ensure_dataframe(load_pickle(test_path), test_path)

    train_pairs = pair_set(train_df)
    test_pairs = pair_set(test_df)

    test_pair_values = make_pair_series(test_df)
    seen_mask = pd.Series([p in train_pairs for p in test_pair_values], index=test_df.index)
    strict_test_df = test_df.loc[~seen_mask].copy()
    seen_test_df = test_df.loc[seen_mask].copy()

    strict_dir = STRICT_ROOT / city
    strict_test_path = strict_dir / "test_strict_unseen.pkl"
    seen_test_path = strict_dir / "test_seen_pairs_removed.pkl"
    save_pickle(strict_test_df, strict_test_path)
    save_pickle(seen_test_df, seen_test_path)

    strict_pairs = pair_set(strict_test_df)
    seen_pairs = pair_set(seen_test_df)

    test_unique_pair_count = len(test_pairs)
    overlap_pair_count = len(train_pairs & test_pairs)
    strict_unique_pair_count = len(strict_pairs)
    seen_unique_pair_count = len(seen_pairs)

    test_rows = len(test_df)
    strict_rows = len(strict_test_df)
    seen_rows = len(seen_test_df)

    train_brands = set(train_df["Brand_ID"].astype(int).unique())
    test_brands = set(test_df["Brand_ID"].astype(int).unique())
    strict_brands = set(strict_test_df["Brand_ID"].astype(int).unique())

    brands_lost_in_strict = sorted(test_brands - strict_brands)

    return {
        "city": city,
        "official_train_rows": int(len(train_df)),
        "official_test_rows": int(test_rows),
        "strict_unseen_test_rows": int(strict_rows),
        "removed_seen_pair_rows": int(seen_rows),
        "strict_unseen_row_ratio": float(strict_rows / test_rows) if test_rows else 0.0,
        "removed_seen_pair_row_ratio": float(seen_rows / test_rows) if test_rows else 0.0,
        "official_train_unique_brands": int(len(train_brands)),
        "official_test_unique_brands": int(len(test_brands)),
        "strict_unseen_test_unique_brands": int(len(strict_brands)),
        "brands_lost_in_strict_count": int(len(brands_lost_in_strict)),
        "brands_lost_in_strict_sample": ",".join(map(str, brands_lost_in_strict[:20])),
        "official_train_unique_regions": int(train_df["Region_ID"].nunique()),
        "official_test_unique_regions": int(test_df["Region_ID"].nunique()),
        "strict_unseen_test_unique_regions": int(strict_test_df["Region_ID"].nunique()),
        "official_train_unique_pairs": int(len(train_pairs)),
        "official_test_unique_pairs": int(test_unique_pair_count),
        "overlap_unique_pairs": int(overlap_pair_count),
        "overlap_pair_ratio_in_test_unique_pairs": float(overlap_pair_count / test_unique_pair_count) if test_unique_pair_count else 0.0,
        "strict_unseen_unique_pairs": int(strict_unique_pair_count),
        "removed_seen_unique_pairs": int(seen_unique_pair_count),
        "strict_test_pairs_seen_in_train_check": int(len(train_pairs & strict_pairs)),
        "strict_test_path": str(strict_test_path),
        "removed_seen_pair_rows_path": str(seen_test_path),
    }


def main() -> None:
    STRICT_ROOT.mkdir(parents=True, exist_ok=True)
    RAW_OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_OUT_DIR.mkdir(parents=True, exist_ok=True)
    CONFIG_OUT_DIR.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, object]] = []
    report_lines: List[str] = []
    report_lines.append("# Strict Unseen-Pair Test Split Report")
    report_lines.append("")
    report_lines.append(f"PROJECT_ROOT: {PROJECT_ROOT}")
    report_lines.append(f"OFFICIAL_ROOT: {OFFICIAL_ROOT}")
    report_lines.append(f"STRICT_ROOT: {STRICT_ROOT}")
    report_lines.append("")
    report_lines.append("Rule: keep a test row only if its (Brand_ID, Region_ID) pair is absent from official train.pkl.")
    report_lines.append("Note: official train.pkl and official test.pkl are not modified.")
    report_lines.append("")

    for city in CITIES:
        row = audit_city(city)
        rows.append(row)

        report_lines.append("=" * 90)
        report_lines.append(f"CITY: {city}")
        for key, value in row.items():
            report_lines.append(f"{key}: {value}")
        report_lines.append("")

    stats_df = pd.DataFrame(rows)
    stats_path = RAW_OUT_DIR / "strict_unseen_test_split_stats.csv"
    stats_df.to_csv(stats_path, index=False, encoding="utf-8-sig")

    report_path = REPORT_OUT_DIR / "strict_unseen_test_split_report.txt"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")

    config = {
        "script": "exp_clean/scripts/00_data_checks/create_strict_unseen_test_splits.py",
        "cities": CITIES,
        "rule": "Keep test rows whose (Brand_ID, Region_ID) pair is absent from official train.pkl.",
        "main_evaluation": "Use official test.pkl for official-compatible reproduction results.",
        "robustness_evaluation": "Use test_strict_unseen.pkl for strict unseen-pair robustness evaluation.",
        "official_files_modified": False,
        "outputs": {
            "stats": str(stats_path),
            "report": str(report_path),
            "strict_root": str(STRICT_ROOT),
        },
    }
    config_path = CONFIG_OUT_DIR / "strict_unseen_split_config.json"
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[OK] Strict unseen split stats saved to: {stats_path}")
    print(f"[OK] Strict unseen split report saved to: {report_path}")
    print(f"[OK] Strict unseen split config saved to: {config_path}")


if __name__ == "__main__":
    main()
