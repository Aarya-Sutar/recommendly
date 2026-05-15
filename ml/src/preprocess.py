from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from scipy import sparse

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw" / "retailrocket"
PROCESSED_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"
SPLIT_DIR = Path(__file__).resolve().parents[1] / "data" / "splits"
ARTIFACT_DIR = PROCESSED_DIR / "artifacts"

RAW_EVENTS_FILE = RAW_DIR / "events.csv"

NORMALIZED_EVENTS_FILE = PROCESSED_DIR / "events_normalized.parquet"
AGGREGATED_INTERACTIONS_FILE = PROCESSED_DIR / "interactions_aggregated.parquet"

TRAIN_FILE = SPLIT_DIR / "train.parquet"
VALIDATION_FILE = SPLIT_DIR / "validation.parquet"
TEST_FILE = SPLIT_DIR / "test.parquet"

TRAIN_MATRIX_FILE = SPLIT_DIR / "train_matrix.npz"

USER_MAP_FILE = ARTIFACT_DIR / "user_map.json"
ITEM_MAP_FILE = ARTIFACT_DIR / "item_map.json"
STATS_FILE = ARTIFACT_DIR / "dataset_stats.json"

EVENT_WEIGHTS = {
    "view": 1.0,
    "addtocart": 3.0,
    "transaction": 5.0,
}

ALLOWED_EVENTS = set(EVENT_WEIGHTS.keys())


def ensure_directories() -> None:
    """Create all output directories if they do not already exist."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)


def load_raw_events() -> pd.DataFrame:
    """
    Load the raw RetailRocket interaction log.

    Expected columns:
    - timestamp
    - visitorid
    - event
    - itemid
    """
    if not RAW_EVENTS_FILE.exists():
        raise FileNotFoundError(
            f"Missing dataset file: {RAW_EVENTS_FILE}\n"
            "Put RetailRocket's events.csv into ml/data/raw/retailrocket/"
        )

    df = pd.read_csv(RAW_EVENTS_FILE)

    expected_columns = {"timestamp", "visitorid", "event", "itemid"}
    missing = expected_columns - set(df.columns)
    if missing:
        raise ValueError(f"events.csv is missing columns: {sorted(missing)}")

    return df


def normalize_events(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert the raw event log into a clean canonical interaction table.

    Output columns:
    - user_id
    - item_id
    - event_type
    - event_value
    - event_ts
    """
    out = df.copy()

    out["event"] = out["event"].astype(str).str.lower().str.strip()
    out = out[out["event"].isin(ALLOWED_EVENTS)].copy()

    out["event_ts"] = pd.to_datetime(out["timestamp"], unit="ms", utc=True)
    out["user_id"] = out["visitorid"].astype(str)
    out["item_id"] = out["itemid"].astype(str)
    out["event_type"] = out["event"]
    out["event_value"] = out["event_type"].map(EVENT_WEIGHTS).astype(float)

    out = out[
        [
            "user_id",
            "item_id",
            "event_type",
            "event_value",
            "event_ts",
        ]
    ].copy()

    out = out.dropna(subset=["user_id", "item_id", "event_ts"])
    out = out.drop_duplicates(subset=["user_id", "item_id", "event_type", "event_ts"])
    out = out.sort_values(["user_id", "event_ts", "item_id"]).reset_index(drop=True)

    return out


def aggregate_user_item_interactions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse repeated interactions for the same user-item pair.

    Why this matters:
    - The same user may view the same item many times.
    - Training and evaluation should work on unique user-item pairs.
    - We keep the strongest useful signal per pair.
    """
    aggregated = (
        df.groupby(["user_id", "item_id"], as_index=False)
        .agg(
            interaction_score=("event_value", "sum"),
            interaction_count=("event_value", "size"),
            first_event_ts=("event_ts", "min"),
            last_event_ts=("event_ts", "max"),
        )
        .sort_values(["user_id", "last_event_ts", "item_id"])
        .reset_index(drop=True)
    )

    return aggregated


def build_id_maps(df: pd.DataFrame) -> Tuple[Dict[str, int], Dict[str, int]]:
    """
    Build stable integer mappings for users and items.

    We sort by interaction frequency so popular entities get smaller indices.
    That is not required, but it is a clean choice.
    """
    user_counts = df["user_id"].value_counts()
    item_counts = df["item_id"].value_counts()

    user_map = {user_id: idx for idx, user_id in enumerate(user_counts.index.tolist())}
    item_map = {item_id: idx for idx, item_id in enumerate(item_counts.index.tolist())}

    return user_map, item_map


def attach_integer_ids(
    df: pd.DataFrame,
    user_map: Dict[str, int],
    item_map: Dict[str, int],
) -> pd.DataFrame:
    """Add integer indices that the recommender will use later."""
    out = df.copy()
    out["user_idx"] = out["user_id"].map(user_map)
    out["item_idx"] = out["item_id"].map(item_map)

    if out["user_idx"].isna().any() or out["item_idx"].isna().any():
        raise ValueError("Some user/item IDs could not be mapped to integer indices.")

    out["user_idx"] = out["user_idx"].astype(int)
    out["item_idx"] = out["item_idx"].astype(int)

    return out


def split_user_histories(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split each user's interaction history chronologically.

    Rules:
    - 1 interaction: train only
    - 2 interactions: train, test
    - 3+ interactions: train, validation, test

    This avoids leakage and gives us a realistic offline evaluation setup.
    """
    train_parts = []
    validation_parts = []
    test_parts = []

    for _, user_df in df.groupby("user_id", sort=False):
        user_df = user_df.sort_values("last_event_ts").reset_index(drop=True)
        n = len(user_df)

        if n == 1:
            train_parts.append(user_df)
        elif n == 2:
            train_parts.append(user_df.iloc[:1])
            test_parts.append(user_df.iloc[1:])
        else:
            train_parts.append(user_df.iloc[:-2])
            validation_parts.append(user_df.iloc[-2:-1])
            test_parts.append(user_df.iloc[-1:])

    train_df = pd.concat(train_parts, ignore_index=True) if train_parts else df.iloc[0:0].copy()
    validation_df = (
        pd.concat(validation_parts, ignore_index=True) if validation_parts else df.iloc[0:0].copy()
    )
    test_df = pd.concat(test_parts, ignore_index=True) if test_parts else df.iloc[0:0].copy()

    return train_df, validation_df, test_df


def build_sparse_matrix(df: pd.DataFrame, num_users: int, num_items: int) -> sparse.csr_matrix:
    """
    Build a sparse user-item matrix for training.

    Rows = users
    Columns = items
    Values = interaction scores
    """
    if df.empty:
        return sparse.csr_matrix((num_users, num_items), dtype=np.float32)

    rows = df["user_idx"].to_numpy(dtype=np.int32)
    cols = df["item_idx"].to_numpy(dtype=np.int32)
    data = df["interaction_score"].to_numpy(dtype=np.float32)

    matrix = sparse.csr_matrix((data, (rows, cols)), shape=(num_users, num_items), dtype=np.float32)
    matrix.sum_duplicates()
    return matrix


def save_json(data: dict, path: Path) -> None:
    """Small helper to save dicts as pretty JSON."""
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def save_dataframe(df: pd.DataFrame, path: Path) -> None:
    """Save a dataframe as parquet."""
    df.to_parquet(path, index=False)


def main() -> None:
    ensure_directories()

    print("Loading raw events...")
    raw_df = load_raw_events()
    print(f"Raw rows: {len(raw_df):,}")

    print("Normalizing events...")
    normalized_df = normalize_events(raw_df)
    save_dataframe(normalized_df, NORMALIZED_EVENTS_FILE)
    print(f"Normalized rows: {len(normalized_df):,}")

    print("Aggregating duplicate user-item interactions...")
    aggregated_df = aggregate_user_item_interactions(normalized_df)
    save_dataframe(aggregated_df, AGGREGATED_INTERACTIONS_FILE)
    print(f"Aggregated rows: {len(aggregated_df):,}")

    print("Building user/item mappings...")
    user_map, item_map = build_id_maps(aggregated_df)
    save_json(user_map, USER_MAP_FILE)
    save_json(item_map, ITEM_MAP_FILE)

    print("Attaching integer IDs...")
    indexed_df = attach_integer_ids(aggregated_df, user_map, item_map)

    print("Splitting user histories into train/validation/test...")
    train_df, validation_df, test_df = split_user_histories(indexed_df)

    save_dataframe(train_df, TRAIN_FILE)
    save_dataframe(validation_df, VALIDATION_FILE)
    save_dataframe(test_df, TEST_FILE)

    print("Building sparse train matrix...")
    train_matrix = build_sparse_matrix(
        train_df,
        num_users=len(user_map),
        num_items=len(item_map),
    )
    sparse.save_npz(TRAIN_MATRIX_FILE, train_matrix)

    stats = {
        "raw_rows": int(len(raw_df)),
        "normalized_rows": int(len(normalized_df)),
        "aggregated_rows": int(len(aggregated_df)),
        "num_users": int(aggregated_df["user_id"].nunique()),
        "num_items": int(aggregated_df["item_id"].nunique()),
        "train_rows": int(len(train_df)),
        "validation_rows": int(len(validation_df)),
        "test_rows": int(len(test_df)),
        "train_matrix_shape": [int(train_matrix.shape[0]), int(train_matrix.shape[1])],
        "train_matrix_nonzeros": int(train_matrix.nnz),
    }
    save_json(stats, STATS_FILE)

    print("\nDone.")
    print(f"Saved normalized data to: {NORMALIZED_EVENTS_FILE}")
    print(f"Saved aggregated data to:  {AGGREGATED_INTERACTIONS_FILE}")
    print(f"Saved train split to:      {TRAIN_FILE}")
    print(f"Saved validation split to: {VALIDATION_FILE}")
    print(f"Saved test split to:       {TEST_FILE}")
    print(f"Saved sparse matrix to:    {TRAIN_MATRIX_FILE}")
    print(f"Saved stats to:            {STATS_FILE}")
    print("\nDataset summary:")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()