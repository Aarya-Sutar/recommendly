from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict

os.environ["OPENBLAS_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]

SPLIT_DIR = BASE_DIR / "data" / "splits"
ARTIFACT_DIR = BASE_DIR / "data" / "processed" / "artifacts"
POPULARITY_DIR = ARTIFACT_DIR / "popularity"

TRAIN_FILE = SPLIT_DIR / "train.parquet"
USER_MAP_FILE = ARTIFACT_DIR / "user_map.json"
ITEM_MAP_FILE = ARTIFACT_DIR / "item_map.json"

ITEM_POPULARITY_FILE = POPULARITY_DIR / "item_popularity.json"
TOP_POPULAR_ITEMS_FILE = POPULARITY_DIR / "top_popular_items.json"
POPULARITY_SUMMARY_FILE = POPULARITY_DIR / "popularity_summary.json"


def ensure_directories() -> None:
    POPULARITY_DIR.mkdir(parents=True, exist_ok=True)


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def invert_mapping(mapping: Dict[str, int]) -> Dict[int, str]:
    return {int(v): str(k) for k, v in mapping.items()}


def load_train_df() -> pd.DataFrame:
    if not TRAIN_FILE.exists():
        raise FileNotFoundError(
            f"Missing train split file: {TRAIN_FILE}\n"
            "Run ml/src/preprocess.py first."
        )
    df = pd.read_parquet(TRAIN_FILE)

    required = {"item_idx", "interaction_score", "last_event_ts"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"train.parquet is missing columns: {sorted(missing)}")

    return df


def normalize_series(values: pd.Series) -> pd.Series:
    min_value = float(values.min())
    max_value = float(values.max())

    if abs(max_value - min_value) < 1e-12:
        return pd.Series(np.zeros(len(values), dtype=np.float32), index=values.index)

    return ((values - min_value) / (max_value - min_value)).astype(np.float32)


def build_popularity_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build item popularity using the training split only.

    We combine:
    - how many users touched the item
    - how much interaction strength the item received
    """
    popularity = (
        df.groupby("item_idx", as_index=False)
        .agg(
            interaction_count=("user_idx", "size"),
            total_strength=("interaction_score", "sum"),
            last_seen_ts=("last_event_ts", "max"),
        )
        .copy()
    )

    popularity["raw_score"] = np.log1p(popularity["interaction_count"]) + 0.5 * np.log1p(
        popularity["total_strength"]
    )
    popularity["popularity_score"] = normalize_series(popularity["raw_score"])

    popularity = popularity.sort_values(
        ["popularity_score", "interaction_count", "total_strength"],
        ascending=[False, False, False],
    ).reset_index(drop=True)

    return popularity


def save_json(data: object, path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def main() -> None:
    ensure_directories()

    print("Loading training split...")
    train_df = load_train_df()
    print(f"Train rows: {len(train_df):,}")

    print("Building popularity table...")
    popularity_df = build_popularity_table(train_df)

    print("Loading item mapping...")
    item_map = load_json(ITEM_MAP_FILE)
    inverse_item_map = invert_mapping(item_map)

    popularity_lookup = {
        str(int(row.item_idx)): float(row.popularity_score)
        for row in popularity_df.itertuples(index=False)
    }

    top_popular_items = []
    for row in popularity_df.head(500).itertuples(index=False):
        item_idx = int(row.item_idx)
        top_popular_items.append(
            {
                "item_idx": item_idx,
                "external_item_id": inverse_item_map.get(item_idx, str(item_idx)),
                "interaction_count": int(row.interaction_count),
                "total_strength": float(row.total_strength),
                "raw_score": float(row.raw_score),
                "popularity_score": float(row.popularity_score),
            }
        )

    summary = {
        "num_popular_items": int(len(popularity_df)),
        "top_item_idx": int(popularity_df.iloc[0]["item_idx"]) if not popularity_df.empty else None,
        "top_item_external_id": (
            inverse_item_map.get(int(popularity_df.iloc[0]["item_idx"]), str(int(popularity_df.iloc[0]["item_idx"])))
            if not popularity_df.empty
            else None
        ),
        "max_popularity_score": float(popularity_df["popularity_score"].max()) if not popularity_df.empty else 0.0,
        "min_popularity_score": float(popularity_df["popularity_score"].min()) if not popularity_df.empty else 0.0,
        "mean_popularity_score": float(popularity_df["popularity_score"].mean()) if not popularity_df.empty else 0.0,
    }

    save_json(popularity_lookup, ITEM_POPULARITY_FILE)
    save_json(top_popular_items, TOP_POPULAR_ITEMS_FILE)
    save_json(summary, POPULARITY_SUMMARY_FILE)

    print(f"Saved popularity lookup to: {ITEM_POPULARITY_FILE}")
    print(f"Saved top items to:        {TOP_POPULAR_ITEMS_FILE}")
    print(f"Saved summary to:          {POPULARITY_SUMMARY_FILE}")
    print("\nPopularity summary:")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()