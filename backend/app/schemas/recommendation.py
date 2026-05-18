from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class RecommendationItem(BaseModel):
    item_idx: int
    external_item_id: str
    base_score: Optional[float] = None
    popularity_score: float = 0.0
    final_score: float


class RecommendationResponse(BaseModel):
    user_idx: Optional[int] = None
    external_user_id: Optional[str] = None
    strategy: str
    seen_items_count: int = 0
    recommendations: list[RecommendationItem] = Field(default_factory=list)


class SimilarItemsResponse(BaseModel):
    item_idx: int
    external_item_id: str
    strategy: str
    recommendations: list[RecommendationItem] = Field(default_factory=list)