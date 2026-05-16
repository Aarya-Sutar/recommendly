from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Dict, Tuple

os.environ["OPENBLAS_NUM_THREADS"] = "1"

import numpy as np
from implicit.als import AlternatingLeastSquares
from scipy import sparse

BASE_DIR = Path(__file__).resolve().parents[1]

SPLIT_DIR = BASE_DIR / "data" / "splits"
ARTIFACT_DIR = BASE_DIR / "data" / "processed" / "artifacts"
MODEL_DIR = BASE_DIR / "models" / "als"

TRAIN_MATRIX_FILE = SPLIT_DIR / "train_matrix.npz"
USER_MAP_FILE = ARTIFACT_DIR / "user_map.json"
ITEM_MAP_FILE = ARTIFACT_DIR / "item_map.json"
STATS_FILE = ARTIFACT_DIR / "dataset_stats.json"

MODEL_FILE = MODEL_DIR / "als_model.npz"
USER_FACTORS_FILE = MODEL_DIR / "user_factors.npy"
ITEM_FACTORS_FILE = MODEL_DIR / "item_factors.npy"
MODEL_META_FILE = MODEL_DIR / "model_metadata.json"
SAMPLE_RECS_FILE = MODEL_DIR / "sample_recommendations.json"

MODEL_PARAMS = {
    "factors": 32,
    "regularization": 0.08,
    "iterations": 15,
    "calculate_training_loss": True,
}


def ensure_directories() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_mappings() -> Tuple[Dict[str, int], Dict[str, int]]:
    if not USER_MAP_FILE.exists():
        raise FileNotFoundError(f"Missing file: {USER_MAP_FILE}")
    if not ITEM_MAP_FILE.exists():
        raise FileNotFoundError(f"Missing file: {ITEM_MAP_FILE}")

    user_map = load_json(USER_MAP_FILE)
    item_map = load_json(ITEM_MAP_FILE)
    return user_map, item_map


def invert_mapping(mapping: Dict[str, int]) -> Dict[int, str]:
    return {int(v): str(k) for k, v in mapping.items()}


def load_train_matrix() -> sparse.csr_matrix:
    if not TRAIN_MATRIX_FILE.exists():
        raise FileNotFoundError(
            f"Missing train matrix: {TRAIN_MATRIX_FILE}\n"
            "Run ml/src/preprocess.py first."
        )

    matrix = sparse.load_npz(TRAIN_MATRIX_FILE)
    if not sparse.isspmatrix_csr(matrix):
        matrix = matrix.tocsr()

    return matrix.astype(np.float32)


def train_als(train_matrix: sparse.csr_matrix) -> AlternatingLeastSquares:
    """
    Train an implicit ALS model on the user-item sparse matrix.
    """
    num_threads = max(1, (os.cpu_count() or 4) - 1)

    model = AlternatingLeastSquares(
        factors=MODEL_PARAMS["factors"],
        regularization=MODEL_PARAMS["regularization"],
        iterations=MODEL_PARAMS["iterations"],
        calculate_training_loss=MODEL_PARAMS["calculate_training_loss"],
        num_threads=num_threads,
    )

    model.fit(train_matrix)
    return model


def save_model_artifacts(
    model: AlternatingLeastSquares,
    train_matrix: sparse.csr_matrix,
    stats: dict,
) -> None:
    np.save(USER_FACTORS_FILE, model.user_factors.astype(np.float32))
    np.save(ITEM_FACTORS_FILE, model.item_factors.astype(np.float32))

    # Save the full ALS model in the format supported by implicit
    model.save(MODEL_FILE)

    metadata = {
        "model_type": "implicit_als",
        "parameters": MODEL_PARAMS,
        "num_users": int(train_matrix.shape[0]),
        "num_items": int(train_matrix.shape[1]),
        "non_zero_interactions": int(train_matrix.nnz),
        "user_factors_shape": list(model.user_factors.shape),
        "item_factors_shape": list(model.item_factors.shape),
        "source_dataset_stats": stats,
    }

    with MODEL_META_FILE.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

def recommend_for_sample_users(
    model: AlternatingLeastSquares,
    train_matrix: sparse.csr_matrix,
    inverse_user_map: Dict[int, str],
    inverse_item_map: Dict[int, str],
    top_k: int = 10,
    num_samples: int = 3,
) -> list[dict]:
    """
    Generate a few sample recommendations for sanity-checking the model.
    """
    active_users = np.flatnonzero(np.asarray(train_matrix.getnnz(axis=1)).ravel())

    if len(active_users) == 0:
        return []

    sample_users = active_users[:num_samples]
    sample_outputs = []

    for user_idx in sample_users:
        user_row = train_matrix[user_idx]

        item_ids, scores = model.recommend(
            userid=int(user_idx),
            user_items=user_row,
            N=top_k,
            filter_already_liked_items=True,
        )

        recs = []
        for item_idx, score in zip(item_ids.tolist(), scores.tolist()):
            recs.append(
                {
                    "item_idx": int(item_idx),
                    "external_item_id": inverse_item_map.get(int(item_idx), str(item_idx)),
                    "score": float(score),
                }
            )

        sample_outputs.append(
            {
                "user_idx": int(user_idx),
                "external_user_id": inverse_user_map.get(int(user_idx), str(user_idx)),
                "recommendations": recs,
            }
        )

    return sample_outputs


def main() -> None:
    ensure_directories()

    print("Loading dataset stats...")
    stats = load_json(STATS_FILE) if STATS_FILE.exists() else {}

    print("Loading user and item mappings...")
    user_map, item_map = load_mappings()
    inverse_user_map = invert_mapping(user_map)
    inverse_item_map = invert_mapping(item_map)

    print("Loading training matrix...")
    train_matrix = load_train_matrix()
    print(f"Train matrix shape: {train_matrix.shape[0]:,} users x {train_matrix.shape[1]:,} items")
    print(f"Non-zero interactions: {train_matrix.nnz:,}")

    print("Training implicit ALS model...")
    start_time = time.time()
    model = train_als(train_matrix)
    elapsed = time.time() - start_time
    print(f"Training finished in {elapsed:.2f} seconds")

    print("Saving learned embeddings and metadata...")
    save_model_artifacts(model, train_matrix, stats)

    print("Generating sample recommendations...")
    sample_recs = recommend_for_sample_users(
        model=model,
        train_matrix=train_matrix,
        inverse_user_map=inverse_user_map,
        inverse_item_map=inverse_item_map,
        top_k=10,
        num_samples=3,
    )

    with SAMPLE_RECS_FILE.open("w", encoding="utf-8") as f:
        json.dump(sample_recs, f, indent=2, ensure_ascii=False)

    print(f"Saved user factors to: {USER_FACTORS_FILE}")
    print(f"Saved item factors to: {ITEM_FACTORS_FILE}")
    print(f"Saved model metadata to: {MODEL_META_FILE}")
    print(f"Saved sample recommendations to: {SAMPLE_RECS_FILE}")
    print(f"Saved ALS model to: {MODEL_FILE}")

    print("\nSample recommendations:")
    print(json.dumps(sample_recs, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()