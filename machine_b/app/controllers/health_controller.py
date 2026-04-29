from datetime import datetime, timezone
from fastapi import APIRouter

router = APIRouter(tags=["health"])

@router.get("/health")
def health_check() -> dict:
    return{
        "status": "ok",
        "service": "machine_b",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }