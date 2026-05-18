from fastapi import APIRouter, HTTPException, Query, Request

from app.schemas.recommendation import RecommendationResponse, SimilarItemsResponse

router = APIRouter(prefix="/recommendations", tags=["recommendations"])


def get_recommender_service(request: Request):
    service = getattr(request.app.state, "recommender", None)
    if service is None:
        raise HTTPException(status_code=503, detail="Recommendation service is not ready")
    return service


@router.get("/popular", response_model=RecommendationResponse)
def get_popular_recommendations(
    request: Request,
    top_k: int = Query(default=10, ge=1, le=50),
):
    service = get_recommender_service(request)
    return service.recommend_popular(top_k=top_k)


@router.get("/users/by-index/{user_idx}", response_model=RecommendationResponse)
def get_recommendations_by_internal_user_idx(
    request: Request,
    user_idx: int,
    top_k: int = Query(default=10, ge=1, le=50),
    candidate_k: int = Query(default=200, ge=10, le=1000),
    alpha: float = Query(default=0.85, ge=0.0, le=1.0),
):
    service = get_recommender_service(request)
    return service.recommend_for_user_idx(
        user_idx=user_idx,
        top_k=top_k,
        candidate_k=candidate_k,
        alpha=alpha,
    )


@router.get("/users/{external_user_id}", response_model=RecommendationResponse)
def get_recommendations_for_user(
    request: Request,
    external_user_id: str,
    top_k: int = Query(default=10, ge=1, le=50),
    candidate_k: int = Query(default=200, ge=10, le=1000),
    alpha: float = Query(default=0.85, ge=0.0, le=1.0),
):
    service = get_recommender_service(request)
    return service.recommend_for_external_user_id(
        external_user_id=external_user_id,
        top_k=top_k,
        candidate_k=candidate_k,
        alpha=alpha,
    )


@router.get("/items/{external_item_id}/similar", response_model=SimilarItemsResponse)
def get_similar_items(
    request: Request,
    external_item_id: str,
    top_k: int = Query(default=10, ge=1, le=50),
    candidate_k: int = Query(default=200, ge=10, le=1000),
    alpha: float = Query(default=0.85, ge=0.0, le=1.0),
):
    service = get_recommender_service(request)
    try:
        return service.recommend_similar_items(
            external_item_id=external_item_id,
            top_k=top_k,
            candidate_k=candidate_k,
            alpha=alpha,
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc