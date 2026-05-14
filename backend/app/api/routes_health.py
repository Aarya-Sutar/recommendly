from datetime import datetime, timezone

from fastapi import APIRouter

from app.core.config import get_settings

router = APIRouter(tags=["health"])


@router.get("/health")
def health_check():
    settings = get_settings()
    return {
        "status": "ok",
        "service": settings.app_name,
        "environment": settings.environment,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }


@router.get("/ready")
def readiness_check():
    return {"status": "ready"}