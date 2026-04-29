from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.DTO.ad_report import AdReportRequest, AdReportResponse
from app.services.ad_play_log_service import AdPlayLogService


router = APIRouter(
    prefix="/api/ad-play-logs",
    tags=["ad-play-logs"],
)


@router.post(
    "/report",
    response_model=AdReportResponse,
    status_code=status.HTTP_201_CREATED,
)
def create_ad_report(
    request: AdReportRequest,
    db: Session = Depends(get_db),
) -> AdReportResponse:
    ad_play_log_service = AdPlayLogService(db)

    try:
        return ad_play_log_service.create_report(request)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except LookupError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc
