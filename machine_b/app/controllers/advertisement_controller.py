from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.DTO.ad_selection import AdSelectionRequest, AdSelectionResponse
from app.services.advertisement_service import AdvertisementService

router = APIRouter(
    prefix="/api/advertisements",
    tags=["advertisements"],
)

@router.post(
    "/select",
    response_model=AdSelectionResponse,
    status_code=status.HTTP_200_OK
)

def select_ad(request: AdSelectionRequest, db: Session = Depends(get_db)):
    advertisement_service = AdvertisementService(db)

    try:
        advertisement = advertisement_service.select_ad(
            viewer_count=request.viewer_count,
            audience_segment_id=request.audience_segment_id,
        )
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
    
    return AdSelectionResponse(
        ad_id=advertisement.id,
        media_filename= advertisement.media_filename,
        duration_seconds=advertisement.duration_seconds,
    )
