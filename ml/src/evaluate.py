from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

os.environ["OPENBLAS_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd
from implicit.cpu.als import AlternatingLeastSquares
from scipy import sparse

BASE_DIR = Path(__file__).resolve().parents[1]

SPLIT_DIR = BASE_DIR / "data" / "splits"
ARTIFACT_DIR = BASE_DIR / "data" / "processed" / "artifacts"
MODEL_DIR = BASE_DIR / "models" / "als"
EVAL_DIR = MODEL_DIR / "evaluation"

TRAIN_MATRIX_FILE = SPLIT_DIR / "train_matrix.npz"
VALIDATION_FILE = SPLIT_DIR / "validation.parquet"
TEST_FILE = SPLIT_DIR / "test.parquet"

STATS_FILE = ARTIFACT_DIR / "dataset_stats.json"
USER_MAP_FILE = ARTIFACT_DIR / "user_map.json"
ITEM_MAP_FILE = ARTIFACT_DIR / "item_map.json"

MODEL_FILE = MODEL_DIR / "als_model.npz"
EVALUATION_REPORT_FILE = EVAL_DIR / "evaluation_report_k10.json"
VALIDATION_SAMPLES_FILE = EVAL_DIR / "validation_samples_k10.json"
TEST_SAMPLES_FILE = EVAL_DIR / "test_samples_k10.json"

TOP_K = 10
BATCH_SIZE = 1024


def ensure_directories() -> None:
    EVAL_DIR.mkdir(parents=True, exist_ok=True)


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def invert_mapping(mapping: Dict[str, int]) -> Dict[int, str]:
    return {int(v): str(k) for k, v in mapping.items()}


def load_train_matrix() -> sparse.csr_matrix:
    if not TRAIN_MATRIX_FILE.exists():
        raise FileNotFoundError(f"Missing train matrix: {TRAIN_MATRIX_FILE}")
    matrix = sparse.load_npz(TRAIN_MATRIX_FILE)
    if not sparse.isspmatrix_csr(matrix):
        matrix = matrix.tocsr()
    return matrix.astype(np.float32)


def load_split_df(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing split file: {path}")
    return pd.read_parquet(path)


def load_model() -> AlternatingLeastSquares:
    if not MODEL_FILE.exists():
        raise FileNotFoundError(
            f"Missing model file: {MODEL_FILE}\n"
            "Patch ml/src/train.py to save model.save(...) and rerun training."
        )
    return AlternatingLeastSquares.load(MODEL_FILE)


def build_truth_lookup(df: pd.DataFrame) -> Dict[int, set[int]]:
    """
    Build a user -> set(items) lookup from the held-out split.
    Each user usually has a single held-out item, but this supports multiple.
    """
    truth: Dict[int, set[int]] = {}
    for row in df.itertuples(index=False):
        user_idx = int(row.user_idx)
        item_idx = int(row.item_idx)
        truth.setdefault(user_idx, set()).add(item_idx)
    return truth


def compute_recall_at_k(recommended_items: List[int], relevant_items: set[int], k: int) -> float:
    if not relevant_items:
        return 0.0
    top_k_items = set(recommended_items[:k])
    return len(top_k_items.intersection(relevant_items)) / len(relevant_items)


def compute_ndcg_at_k(recommended_items: List[int], relevant_items: set[int], k: int) -> float:
    if not relevant_items:
        return 0.0

    dcg = 0.0
    for rank, item_idx in enumerate(recommended_items[:k]):
        if item_idx in relevant_items:
            dcg += 1.0 / np.log2(rank + 2)

    ideal_hits = min(len(relevant_items), k)
    idcg = sum(1.0 / np.log2(rank + 2) for rank in range(ideal_hits))

    if idcg == 0.0:
        return 0.0
    return dcg / idcg


def evaluate_split(
    model: AlternatingLeastSquares,
    train_matrix: sparse.csr_matrix,
    split_df: pd.DataFrame,
    inverse_user_map: Dict[int, str],
    inverse_item_map: Dict[int, str],
    split_name: str,
    k: int = TOP_K,
    batch_size: int = BATCH_SIZE,
) -> Tuple[dict, List[dict]]:
    truth_lookup = build_truth_lookup(split_df)
    eligible_users = np.array(sorted(truth_lookup.keys()), dtype=np.int32)

    if len(eligible_users) == 0:
        return {
            "split": split_name,
            "k": k,
            "users_evaluated": 0,
            "mean_recall_at_k": 0.0,
            "mean_ndcg_at_k": 0.0,
            "average_relevant_items_per_user": 0.0,
        }, []

    recalls: List[float] = []
    ndcgs: List[float] = []
    relevant_counts: List[int] = []
    samples: List[dict] = []

    for start in range(0, len(eligible_users), batch_size):
        batch_users = eligible_users[start : start + batch_size]
        batch_user_items = train_matrix[batch_users]

        recommended_item_ids, recommended_scores = model.recommend(
            batch_users,
            batch_user_items,
            N=k,
            filter_already_liked_items=True,
        )

        recommended_item_ids = np.asarray(recommended_item_ids)
        recommended_scores = np.asarray(recommended_scores)

        if recommended_item_ids.ndim == 1:
            recommended_item_ids = recommended_item_ids[np.newaxis, :]
            recommended_scores = recommended_scores[np.newaxis, :]

        for row_idx, user_idx in enumerate(batch_users):
            relevant_items = truth_lookup.get(int(user_idx), set())
            if not relevant_items:
                continue

            rec_items = [int(x) for x in recommended_item_ids[row_idx].tolist()]
            rec_scores = [float(x) for x in recommended_scores[row_idx].tolist()]

            recall = compute_recall_at_k(rec_items, relevant_items, k)
            ndcg = compute_ndcg_at_k(rec_items, relevant_items, k)

            recalls.append(recall)
            ndcgs.append(ndcg)
            relevant_counts.append(len(relevant_items))

            if len(samples) < 5:
                samples.append(
                    {
                        "user_idx": int(user_idx),
                        "external_user_id": inverse_user_map.get(int(user_idx), str(int(user_idx))),
                        "relevant_items": [
                            {
                                "item_idx": int(item_idx),
                                "external_item_id": inverse_item_map.get(int(item_idx), str(int(item_idx))),
                            }
                            for item_idx in sorted(relevant_items)
                        ],
                        "recommendations": [
                            {
                                "item_idx": int(item_idx),
                                "external_item_id": inverse_item_map.get(int(item_idx), str(int(item_idx))),
                                "score": float(score),
                            }
                            for item_idx, score in zip(rec_items, rec_scores)
                        ],
                        "recall_at_k": float(recall),
                        "ndcg_at_k": float(ndcg),
                    }
                )

    users_evaluated = len(recalls)
    metrics = {
        "split": split_name,
        "k": k,
        "users_evaluated": int(users_evaluated),
        "mean_recall_at_k": float(np.mean(recalls) if recalls else 0.0),
        "mean_ndcg_at_k": float(np.mean(ndcgs) if ndcgs else 0.0),
        "average_relevant_items_per_user": float(np.mean(relevant_counts) if relevant_counts else 0.0),
    }
    return metrics, samples


def main() -> None:
    ensure_directories()

    print("Loading stats and mappings...")
    stats = load_json(STATS_FILE) if STATS_FILE.exists() else {}
    user_map = load_json(USER_MAP_FILE)
    item_map = load_json(ITEM_MAP_FILE)
    inverse_user_map = invert_mapping(user_map)
    inverse_item_map = invert_mapping(item_map)

    print("Loading train matrix...")
    train_matrix = load_train_matrix()
    print(f"Train matrix shape: {train_matrix.shape[0]:,} users x {train_matrix.shape[1]:,} items")

    print("Loading validation and test splits...")
    validation_df = load_split_df(VALIDATION_FILE)
    test_df = load_split_df(TEST_FILE)

    print("Loading ALS model...")
    model = load_model()

    print("\nEvaluating validation split...")
    validation_metrics, validation_samples = evaluate_split(
        model=model,
        train_matrix=train_matrix,
        split_df=validation_df,
        inverse_user_map=inverse_user_map,
        inverse_item_map=inverse_item_map,
        split_name="validation",
        k=TOP_K,
        batch_size=BATCH_SIZE,
    )

    print("Evaluating test split...")
    test_metrics, test_samples = evaluate_split(
        model=model,
        train_matrix=train_matrix,
        split_df=test_df,
        inverse_user_map=inverse_user_map,
        inverse_item_map=inverse_item_map,
        split_name="test",
        k=TOP_K,
        batch_size=BATCH_SIZE,
    )

    report = {
        "model_type": "implicit_als",
        "k": TOP_K,
        "dataset_stats": stats,
        "validation": validation_metrics,
        "test": test_metrics,
    }

    with EVALUATION_REPORT_FILE.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    with VALIDATION_SAMPLES_FILE.open("w", encoding="utf-8") as f:
        json.dump(validation_samples, f, indent=2, ensure_ascii=False)

    with TEST_SAMPLES_FILE.open("w", encoding="utf-8") as f:
        json.dump(test_samples, f, indent=2, ensure_ascii=False)

    print("\nValidation metrics:")
    print(json.dumps(validation_metrics, indent=2, ensure_ascii=False))

    print("\nTest metrics:")
    print(json.dumps(test_metrics, indent=2, ensure_ascii=False))

    print(f"\nSaved evaluation report to: {EVALUATION_REPORT_FILE}")
    print(f"Saved validation samples to: {VALIDATION_SAMPLES_FILE}")
    print(f"Saved test samples to: {TEST_SAMPLES_FILE}")


if __name__ == "__main__":
    main()