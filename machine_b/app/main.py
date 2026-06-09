from fastapi import FastAPI

from app.core.database import SessionLocal
from app.controllers.ad_play_log_controller import router as ad_play_log_router
from app.controllers.advertisement_controller import router as advertisement_router
from app.controllers.dashboard_controller import router as dashboard_router
from app.controllers.health_controller import router as health_router
from app.services.ad_play_log_service import AdPlayLogService


app = FastAPI(
    title="Machine B API",
    description="Backend for smart billboard advertisement selection, reporting, and dashboard",
    version="1.0.0",
)

app.include_router(health_router)
app.include_router(advertisement_router)
app.include_router(ad_play_log_router)
app.include_router(dashboard_router)


@app.on_event("startup")
def purge_expired_ad_play_logs() -> None:
    db = SessionLocal()
    try:
        deleted_count = AdPlayLogService(db).purge_old_logs()
        if deleted_count:
            print(f"Purged {deleted_count} ad play log(s) older than 7 days.")
    finally:
        db.close()


@app.get("/")
def root() -> dict:
    return {
        "message": "Machine B API is running",
        "docs": "/docs",
        "health": "/health",
        "dashboard": "/dashboard",
    }
