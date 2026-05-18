from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Set

import numpy as np
from scipy import sparse

from app.core.config import Settings


class RecommenderService:
    def __init__(self, settings: Settings):
        self.settings = settings

        self.model_dir = settings.ml_models_dir
        self.splits_dir = settings.ml_splits_dir
        self.artifacts_dir = settings.ml_artifacts_dir
        self.popularity_dir = self.artifacts_dir / "popularity"

        self.user_factors = self._load_numpy_array(self.model_dir / "user_factors.npy")
        self.item_factors = self._load_numpy_array(self.model_dir / "item_factors.npy")
        self.train_matrix = self._load_sparse_matrix(self.splits_dir / "train_matrix.npz")

        self.user_map = self._load_json(self.artifacts_dir / "user_map.json")
        self.item_map = self._load_json(self.artifacts_dir / "item_map.json")
        self.inverse_user_map = {int(v): str(k) for k, v in self.user_map.items()}
        self.inverse_item_map = {int(v): str(k) for k, v in self.item_map.items()}

        self.popularity_lookup = self._load_popularity_lookup(self.popularity_dir / "item_popularity.json")
        self.top_popular_items = self._load_json(self.popularity_dir / "top_popular_items.json")

    def _load_json(self, path: Path) -> object:
        if not path.exists():
            raise FileNotFoundError(f"Missing artifact file: {path}")
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def _load_numpy_array(self, path: Path) -> np.ndarray:
        if not path.exists():
            raise FileNotFoundError(f"Missing numpy artifact: {path}")
        return np.load(path).astype(np.float32)

    def _load_sparse_matrix(self, path: Path) -> sparse.csr_matrix:
        if not path.exists():
            raise FileNotFoundError(f"Missing sparse matrix artifact: {path}")
        matrix = sparse.load_npz(path)
        if not sparse.isspmatrix_csr(matrix):
            matrix = matrix.tocsr()
        return matrix.astype(np.float32)

    def _load_popularity_lookup(self, path: Path) -> Dict[str, float]:
        raw = self._load_json(path)
        return {str(k): float(v) for k, v in raw.items()}

    @staticmethod
    def _normalize_scores(values: np.ndarray) -> np.ndarray:
        if values.size == 0:
            return values.astype(np.float32)

        min_value = float(values.min())
        max_value = float(values.max())

        if abs(max_value - min_value) < 1e-12:
            return np.zeros_like(values, dtype=np.float32)

        return ((values - min_value) / (max_value - min_value)).astype(np.float32)

    def _get_seen_items(self, user_idx: int) -> Set[int]:
        if user_idx < 0 or user_idx >= self.train_matrix.shape[0]:
            return set()
        return set(int(i) for i in self.train_matrix[user_idx].indices.tolist())

    def _format_item(
        self,
        item_idx: int,
        base_score: Optional[float],
        popularity_score: float,
        final_score: float,
    ) -> dict:
        return {
            "item_idx": int(item_idx),
            "external_item_id": self.inverse_item_map.get(int(item_idx), str(int(item_idx))),
            "base_score": None if base_score is None else float(base_score),
            "popularity_score": float(popularity_score),
            "final_score": float(final_score),
        }

    def _popular_fallback(
        self,
        blocked_items: Set[int],
        top_k: int,
    ) -> List[dict]:
        recommendations: List[dict] = []

        for row in self.top_popular_items:
            item_idx = int(row["item_idx"])
            if item_idx in blocked_items:
                continue

            recommendations.append(
                self._format_item(
                    item_idx=item_idx,
                    base_score=None,
                    popularity_score=float(row["popularity_score"]),
                    final_score=float(row["popularity_score"]),
                )
            )

            if len(recommendations) >= top_k:
                break

        return recommendations

    def _rerank_candidates(
        self,
        candidate_item_ids: np.ndarray,
        candidate_scores: np.ndarray,
        top_k: int,
        alpha: float,
    ) -> List[dict]:
        if candidate_item_ids.size == 0:
            return []

        candidate_item_ids = np.asarray(candidate_item_ids, dtype=np.int64)
        candidate_scores = np.asarray(candidate_scores, dtype=np.float32)

        als_norm = self._normalize_scores(candidate_scores)
        popularity_scores = np.array(
            [float(self.popularity_lookup.get(str(int(item_idx)), 0.0)) for item_idx in candidate_item_ids],
            dtype=np.float32,
        )

        final_scores = alpha * als_norm + (1.0 - alpha) * popularity_scores
        order = np.argsort(-final_scores)

        recommendations: List[dict] = []
        for idx in order[:top_k]:
            item_idx = int(candidate_item_ids[idx])
            recommendations.append(
                self._format_item(
                    item_idx=item_idx,
                    base_score=float(candidate_scores[idx]),
                    popularity_score=float(popularity_scores[idx]),
                    final_score=float(final_scores[idx]),
                )
            )

        return recommendations

    def recommend_for_user_idx(
        self,
        user_idx: int,
        top_k: int = 10,
        candidate_k: int = 200,
        alpha: float = 0.85,
    ) -> dict:
        if user_idx < 0 or user_idx >= self.train_matrix.shape[0]:
            raise ValueError(f"user_idx {user_idx} is out of bounds")

        seen_items = self._get_seen_items(user_idx)
        user_row = self.train_matrix[user_idx]

        if user_row.nnz == 0:
            recs = self._popular_fallback(blocked_items=seen_items, top_k=top_k)
            return {
                "user_idx": int(user_idx),
                "external_user_id": self.inverse_user_map.get(int(user_idx), str(int(user_idx))),
                "strategy": "popular_fallback",
                "seen_items_count": int(len(seen_items)),
                "recommendations": recs,
            }

        user_vector = self.user_factors[user_idx]
        scores = np.asarray(self.item_factors @ user_vector, dtype=np.float32).ravel()

        if seen_items:
            scores[list(seen_items)] = -np.inf

        available_indices = np.flatnonzero(np.isfinite(scores))
        if available_indices.size == 0:
            recs = self._popular_fallback(blocked_items=seen_items, top_k=top_k)
            return {
                "user_idx": int(user_idx),
                "external_user_id": self.inverse_user_map.get(int(user_idx), str(int(user_idx))),
                "strategy": "popular_fallback_no_candidates",
                "seen_items_count": int(len(seen_items)),
                "recommendations": recs,
            }

        candidate_k = min(candidate_k, available_indices.size)
        if candidate_k < available_indices.size:
            best_local = np.argpartition(-scores[available_indices], candidate_k - 1)[:candidate_k]
            candidate_indices = available_indices[best_local]
        else:
            candidate_indices = available_indices

        candidate_scores = scores[candidate_indices]
        recs = self._rerank_candidates(
            candidate_item_ids=candidate_indices,
            candidate_scores=candidate_scores,
            top_k=top_k,
            alpha=alpha,
        )

        if len(recs) < top_k:
            blocked_items = seen_items | {int(r["item_idx"]) for r in recs}
            recs.extend(self._popular_fallback(blocked_items=blocked_items, top_k=top_k - len(recs)))

        return {
            "user_idx": int(user_idx),
            "external_user_id": self.inverse_user_map.get(int(user_idx), str(int(user_idx))),
            "strategy": "hybrid_als_popularity",
            "seen_items_count": int(len(seen_items)),
            "recommendations": recs[:top_k],
        }

    def recommend_for_external_user_id(
        self,
        external_user_id: str,
        top_k: int = 10,
        candidate_k: int = 200,
        alpha: float = 0.85,
    ) -> dict:
        if external_user_id not in self.user_map:
            recs = self._popular_fallback(blocked_items=set(), top_k=top_k)
            return {
                "user_idx": None,
                "external_user_id": external_user_id,
                "strategy": "popular_fallback_unknown_user",
                "seen_items_count": 0,
                "recommendations": recs,
            }

        user_idx = int(self.user_map[external_user_id])
        return self.recommend_for_user_idx(
            user_idx=user_idx,
            top_k=top_k,
            candidate_k=candidate_k,
            alpha=alpha,
        )

    def recommend_popular(
        self,
        top_k: int = 10,
    ) -> dict:
        recs = self._popular_fallback(blocked_items=set(), top_k=top_k)
        return {
            "user_idx": None,
            "external_user_id": None,
            "strategy": "popular_baseline",
            "seen_items_count": 0,
            "recommendations": recs,
        }

    def recommend_similar_items(
        self,
        external_item_id: str,
        top_k: int = 10,
        candidate_k: int = 200,
        alpha: float = 0.85,
    ) -> dict:
        if external_item_id not in self.item_map:
            raise ValueError(f"Unknown external_item_id: {external_item_id}")

        item_idx = int(self.item_map[external_item_id])
        item_vector = self.item_factors[item_idx]
        scores = np.asarray(self.item_factors @ item_vector, dtype=np.float32).ravel()
        scores[item_idx] = -np.inf

        available_indices = np.flatnonzero(np.isfinite(scores))
        if available_indices.size == 0:
            return {
                "item_idx": int(item_idx),
                "external_item_id": external_item_id,
                "strategy": "similar_items_no_candidates",
                "recommendations": [],
            }

        candidate_k = min(candidate_k, available_indices.size)
        if candidate_k < available_indices.size:
            best_local = np.argpartition(-scores[available_indices], candidate_k - 1)[:candidate_k]
            candidate_indices = available_indices[best_local]
        else:
            candidate_indices = available_indices

        candidate_scores = scores[candidate_indices]
        als_norm = self._normalize_scores(candidate_scores)
        popularity_scores = np.array(
            [float(self.popularity_lookup.get(str(int(item_id)), 0.0)) for item_id in candidate_indices],
            dtype=np.float32,
        )
        final_scores = alpha * als_norm + (1.0 - alpha) * popularity_scores
        order = np.argsort(-final_scores)

        recs: List[dict] = []
        for idx in order[:top_k]:
            recs.append(
                self._format_item(
                    item_idx=int(candidate_indices[idx]),
                    base_score=float(candidate_scores[idx]),
                    popularity_score=float(popularity_scores[idx]),
                    final_score=float(final_scores[idx]),
                )
            )

        return {
            "item_idx": int(item_idx),
            "external_item_id": external_item_id,
            "strategy": "hybrid_item_similarity_popularity",
            "recommendations": recs,
        }