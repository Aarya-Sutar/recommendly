from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

os.environ["OPENBLAS_NUM_THREADS"] = "1"

import numpy as np
from implicit.cpu.als import AlternatingLeastSquares
from scipy import sparse

BASE_DIR = Path(__file__).resolve().parents[1]

SPLIT_DIR = BASE_DIR / "data" / "splits"
ARTIFACT_DIR = BASE_DIR / "data" / "processed" / "artifacts"
POPULARITY_DIR = ARTIFACT_DIR / "popularity"
MODEL_DIR = BASE_DIR / "models" / "als"

TRAIN_MATRIX_FILE = SPLIT_DIR / "train_matrix.npz"
USER_MAP_FILE = ARTIFACT_DIR / "user_map.json"
ITEM_MAP_FILE = ARTIFACT_DIR / "item_map.json"
ITEM_POPULARITY_FILE = POPULARITY_DIR / "item_popularity.json"
TOP_POPULAR_ITEMS_FILE = POPULARITY_DIR / "top_popular_items.json"

MODEL_FILE = MODEL_DIR / "als_model.npz"
HYBRID_SAMPLES_FILE = MODEL_DIR / "hybrid_sample_recommendations.json"

DEFAULT_TOP_K = 10
DEFAULT_CANDIDATE_K = 200
DEFAULT_ALPHA = 0.85


def ensure_directories() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def invert_mapping(mapping: Dict[str, int]) -> Dict[int, str]:
    return {int(v): str(k) for k, v in mapping.items()}


def load_model() -> AlternatingLeastSquares:
    if not MODEL_FILE.exists():
        raise FileNotFoundError(
            f"Missing model file: {MODEL_FILE}\n"
            "Run ml/src/train.py first."
        )
    return AlternatingLeastSquares.load(MODEL_FILE)


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


def load_mappings() -> Tuple[Dict[str, int], Dict[str, int]]:
    if not USER_MAP_FILE.exists():
        raise FileNotFoundError(f"Missing file: {USER_MAP_FILE}")
    if not ITEM_MAP_FILE.exists():
        raise FileNotFoundError(f"Missing file: {ITEM_MAP_FILE}")

    user_map = load_json(USER_MAP_FILE)
    item_map = load_json(ITEM_MAP_FILE)
    return user_map, item_map


def load_popularity_lookup() -> Dict[str, float]:
    if not ITEM_POPULARITY_FILE.exists():
        raise FileNotFoundError(
            f"Missing popularity file: {ITEM_POPULARITY_FILE}\n"
            "Run ml/src/popularity.py first."
        )
    raw = load_json(ITEM_POPULARITY_FILE)
    return {str(k): float(v) for k, v in raw.items()}


def load_top_popular_items() -> List[dict]:
    if not TOP_POPULAR_ITEMS_FILE.exists():
        raise FileNotFoundError(
            f"Missing top popular items file: {TOP_POPULAR_ITEMS_FILE}\n"
            "Run ml/src/popularity.py first."
        )
    return load_json(TOP_POPULAR_ITEMS_FILE)


def normalize_scores(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return values.astype(np.float32)

    min_value = float(values.min())
    max_value = float(values.max())

    if abs(max_value - min_value) < 1e-12:
        return np.zeros_like(values, dtype=np.float32)

    return ((values - min_value) / (max_value - min_value)).astype(np.float32)


def get_seen_items(train_matrix: sparse.csr_matrix, user_idx: int) -> set[int]:
    if user_idx < 0 or user_idx >= train_matrix.shape[0]:
        return set()
    return set(int(i) for i in train_matrix[user_idx].indices.tolist())


def fallback_popular_recommendations(
    top_popular_items: List[dict],
    seen_items: set[int],
    inverse_item_map: Dict[int, str],
    top_k: int,
) -> List[dict]:
    recommendations = []
    for row in top_popular_items:
        item_idx = int(row["item_idx"])
        if item_idx in seen_items:
            continue
        recommendations.append(
            {
                "item_idx": item_idx,
                "external_item_id": row.get("external_item_id", inverse_item_map.get(item_idx, str(item_idx))),
                "als_score": None,
                "popularity_score": float(row["popularity_score"]),
                "final_score": float(row["popularity_score"]),
            }
        )
        if len(recommendations) >= top_k:
            break

    return recommendations


def rerank_candidates(
    candidate_item_ids: np.ndarray,
    candidate_scores: np.ndarray,
    popularity_lookup: Dict[str, float],
    inverse_item_map: Dict[int, str],
    top_k: int,
    alpha: float,
) -> List[dict]:
    if candidate_item_ids.size == 0:
        return []

    candidate_item_ids = np.asarray(candidate_item_ids, dtype=np.int64)
    candidate_scores = np.asarray(candidate_scores, dtype=np.float32)

    als_norm = normalize_scores(candidate_scores)
    pop_scores = np.array(
        [float(popularity_lookup.get(str(int(item_idx)), 0.0)) for item_idx in candidate_item_ids],
        dtype=np.float32,
    )

    final_scores = alpha * als_norm + (1.0 - alpha) * pop_scores
    order = np.argsort(-final_scores)

    recommendations: List[dict] = []
    for idx in order[:top_k]:
        item_idx = int(candidate_item_ids[idx])
        recommendations.append(
            {
                "item_idx": item_idx,
                "external_item_id": inverse_item_map.get(item_idx, str(item_idx)),
                "als_score": float(candidate_scores[idx]),
                "popularity_score": float(pop_scores[idx]),
                "final_score": float(final_scores[idx]),
            }
        )

    return recommendations


def recommend_for_user_idx(
    model: AlternatingLeastSquares,
    train_matrix: sparse.csr_matrix,
    popularity_lookup: Dict[str, float],
    top_popular_items: List[dict],
    inverse_item_map: Dict[int, str],
    user_idx: int,
    top_k: int = DEFAULT_TOP_K,
    candidate_k: int = DEFAULT_CANDIDATE_K,
    alpha: float = DEFAULT_ALPHA,
) -> dict:
    if user_idx < 0 or user_idx >= train_matrix.shape[0]:
        raise ValueError(f"user_idx {user_idx} is out of bounds")

    seen_items = get_seen_items(train_matrix, user_idx)
    user_row = train_matrix[user_idx]

    if user_row.nnz == 0:
        recs = fallback_popular_recommendations(
            top_popular_items=top_popular_items,
            seen_items=seen_items,
            inverse_item_map=inverse_item_map,
            top_k=top_k,
        )
        return {
            "user_idx": int(user_idx),
            "recommendations": recs,
            "strategy": "popular_fallback",
            "seen_items_count": int(len(seen_items)),
        }

    candidate_k = min(candidate_k, train_matrix.shape[1])

    candidate_item_ids, candidate_scores = model.recommend(
        userid=int(user_idx),
        user_items=user_row,
        N=candidate_k,
        filter_already_liked_items=True,
    )

    candidate_item_ids = np.asarray(candidate_item_ids)
    candidate_scores = np.asarray(candidate_scores)

    recs = rerank_candidates(
        candidate_item_ids=candidate_item_ids,
        candidate_scores=candidate_scores,
        popularity_lookup=popularity_lookup,
        inverse_item_map=inverse_item_map,
        top_k=top_k,
        alpha=alpha,
    )

    return {
        "user_idx": int(user_idx),
        "recommendations": recs,
        "strategy": "hybrid_als_popularity",
        "seen_items_count": int(len(seen_items)),
    }


def recommend_for_external_user(
    external_user_id: str,
    user_map: Dict[str, int],
    model: AlternatingLeastSquares,
    train_matrix: sparse.csr_matrix,
    popularity_lookup: Dict[str, float],
    top_popular_items: List[dict],
    inverse_item_map: Dict[int, str],
    top_k: int = DEFAULT_TOP_K,
    candidate_k: int = DEFAULT_CANDIDATE_K,
    alpha: float = DEFAULT_ALPHA,
) -> dict:
    if external_user_id not in user_map:
        raise ValueError(f"Unknown external_user_id: {external_user_id}")

    user_idx = int(user_map[external_user_id])
    result = recommend_for_user_idx(
        model=model,
        train_matrix=train_matrix,
        popularity_lookup=popularity_lookup,
        top_popular_items=top_popular_items,
        inverse_item_map=inverse_item_map,
        user_idx=user_idx,
        top_k=top_k,
        candidate_k=candidate_k,
        alpha=alpha,
    )
    result["external_user_id"] = external_user_id
    return result


def save_json(data: object, path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def main() -> None:
    ensure_directories()

    parser = argparse.ArgumentParser(description="Generate hybrid ALS + popularity recommendations")
    parser.add_argument("--user_id", type=str, default=None, help="External user ID")
    parser.add_argument("--user_idx", type=int, default=None, help="Internal integer user index")
    parser.add_argument("--top_k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--candidate_k", type=int, default=DEFAULT_CANDIDATE_K)
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    args = parser.parse_args()

    print("Loading artifacts...")
    model = load_model()
    train_matrix = load_train_matrix()
    user_map, item_map = load_mappings()
    inverse_user_map = invert_mapping(user_map)
    inverse_item_map = invert_mapping(item_map)
    popularity_lookup = load_popularity_lookup()
    top_popular_items = load_top_popular_items()

    if args.user_id is not None and args.user_idx is not None:
        raise ValueError("Use either --user_id or --user_idx, not both.")

    if args.user_id is not None:
        result = recommend_for_external_user(
            external_user_id=args.user_id,
            user_map=user_map,
            model=model,
            train_matrix=train_matrix,
            popularity_lookup=popularity_lookup,
            top_popular_items=top_popular_items,
            inverse_item_map=inverse_item_map,
            top_k=args.top_k,
            candidate_k=args.candidate_k,
            alpha=args.alpha,
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return

    if args.user_idx is not None:
        result = recommend_for_user_idx(
            model=model,
            train_matrix=train_matrix,
            popularity_lookup=popularity_lookup,
            top_popular_items=top_popular_items,
            inverse_item_map=inverse_item_map,
            user_idx=args.user_idx,
            top_k=args.top_k,
            candidate_k=args.candidate_k,
            alpha=args.alpha,
        )
        result["external_user_id"] = inverse_user_map.get(args.user_idx, str(args.user_idx))
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return

    active_users = np.flatnonzero(np.asarray(train_matrix.getnnz(axis=1)).ravel())
    sample_users = active_users[:3].tolist()

    outputs = []
    for user_idx in sample_users:
        result = recommend_for_user_idx(
            model=model,
            train_matrix=train_matrix,
            popularity_lookup=popularity_lookup,
            top_popular_items=top_popular_items,
            inverse_item_map=inverse_item_map,
            user_idx=int(user_idx),
            top_k=args.top_k,
            candidate_k=args.candidate_k,
            alpha=args.alpha,
        )
        result["external_user_id"] = inverse_user_map.get(int(user_idx), str(int(user_idx)))
        outputs.append(result)

    sample_path = MODEL_DIR / "hybrid_sample_recommendations.json"
    save_json(outputs, sample_path)

    print(f"Saved hybrid sample recommendations to: {sample_path}")
    print("\nSample recommendations:")
    print(json.dumps(outputs, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()