from datetime import datetime

from pydantic import BaseModel, Field


class AdSelectionRequest(BaseModel):
    timestamp: datetime = Field(..., description="Time when Machine A sends audience data")
    viewer_count: int = Field(..., ge=0, description="Number of viewers detected in the sampling window")
    avg_age: int = Field(..., ge=0, description="Average age of detected viewers")
    majority_gender: str = Field(..., min_length=1, description="Majority gender of detected viewers")


class AdSelectionResponse(BaseModel):
    ad_id: int = Field(..., description="Selected advertisement ID")
    media_filename: str = Field(..., description="Advertisement video filename")
    duration_seconds: int = Field(..., gt=0, description="Advertisement duration in seconds")
