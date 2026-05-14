from pathlib import Path

EXPECTED_FILES = [
    "events.csv",
    "item_properties_part1.csv",
    "item_properties_part2.csv",
]


def main() -> None:
    raw_dir = Path(__file__).resolve().parents[1] / "data" / "raw" / "retailrocket"
    raw_dir.mkdir(parents=True, exist_ok=True)

    missing = [name for name in EXPECTED_FILES if not (raw_dir / name).exists()]
    if missing:
        raise FileNotFoundError(
            "Missing raw RetailRocket files:\n"
            + "\n".join(f"- {name}" for name in missing)
            + "\n\nPut the dataset files in ml/data/raw/retailrocket/"
        )

    print("RetailRocket raw files found. Ready for preprocessing.")


if __name__ == "__main__":
    main()