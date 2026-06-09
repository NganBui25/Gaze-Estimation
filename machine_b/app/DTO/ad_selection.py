from datetime import datetime

from typing import Optional

from pydantic import BaseModel, Field, model_validator


class AdSelectionRequest(BaseModel):
    timestamp: datetime = Field(..., description="Time when Machine A sends audience data")
    viewer_count: int = Field(..., ge=0, description="Number of viewers detected in the sampling window")
    audience_segment_id: Optional[int] = Field(
        None,
        gt=0,
        description="Dominant audience segment ID detected by Machine A",
    )

    @model_validator(mode="after")
    def validate_segment_for_viewers(self):
        if self.viewer_count > 0 and self.audience_segment_id is None:
            raise ValueError("audience_segment_id is required when viewer_count is greater than 0")
        return self


class AdSelectionResponse(BaseModel):
    ad_id: int = Field(..., description="Selected advertisement ID")
    media_filename: str = Field(..., description="Advertisement video filename")
    duration_seconds: int = Field(..., gt=0, description="Advertisement duration in seconds")
