from __future__ import annotations

from pathlib import Path

import pandas as pd

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw" / "retailrocket"
PROCESSED_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"

RAW_EVENTS_FILE = RAW_DIR / "events.csv"
PROCESSED_EVENTS_FILE = PROCESSED_DIR / "interactions.parquet"

# RetailRocket event types are usually view, addtocart, transaction.
# We normalize them into a common interaction schema.
EVENT_WEIGHTS = {
    "view": 1.0,
    "addtocart": 3.0,
    "transaction": 5.0,
}


def load_raw_events() -> pd.DataFrame:
    if not RAW_EVENTS_FILE.exists():
        raise FileNotFoundError(f"Missing raw file: {RAW_EVENTS_FILE}")

    df = pd.read_csv(RAW_EVENTS_FILE)
    expected_cols = {"timestamp", "visitorid", "event", "itemid"}
    missing = expected_cols - set(df.columns)
    if missing:
        raise ValueError(f"Raw events file is missing columns: {sorted(missing)}")

    return df


def normalize_events(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["event"] = out["event"].astype(str).str.lower().str.strip()
    out["event_ts"] = pd.to_datetime(out["timestamp"], unit="ms", utc=True)
    out["user_id"] = out["visitorid"].astype(str)
    out["item_id"] = out["itemid"].astype(str)
    out["event_type"] = out["event"]
    out["event_value"] = out["event_type"].map(EVENT_WEIGHTS).fillna(1.0)

    out = out[["user_id", "item_id", "event_type", "event_value", "event_ts"]]
    out = out.dropna(subset=["user_id", "item_id", "event_ts"])
    out = out.drop_duplicates()
    out = out.sort_values(["user_id", "event_ts"]).reset_index(drop=True)

    return out


def save_processed(df: pd.DataFrame) -> Path:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(PROCESSED_EVENTS_FILE, index=False)
    return PROCESSED_EVENTS_FILE


def main() -> None:
    raw_df = load_raw_events()
    processed_df = normalize_events(raw_df)
    output_path = save_processed(processed_df)
    print(f"Saved processed interactions to: {output_path}")
    print(f"Rows: {len(processed_df):,}")
    print(f"Users: {processed_df['user_id'].nunique():,}")
    print(f"Items: {processed_df['item_id'].nunique():,}")


if __name__ == "__main__":
    main()