from datetime import date, datetime
from typing import Optional

from pydantic import BaseModel, Field

class ViewerReportItem(BaseModel):
    estimated_age: int = Field(..., ge=0, description="Tuoi uoc luong cua nguoi xem")
    gender: str = Field(..., min_length=1, description="Gioi tinh cua nguoi xem")
    watch_duration: float = Field(..., ge=0, description="Thoi gian nhin quang cao")

class AdReportRequest(BaseModel):
    ad_id: int = Field(..., gt=0, description="ID quang cao da phat")
    start_time: datetime = Field(..., description="Thoi diem bat dau phat")
    end_time: datetime = Field(..., description="Thoi diem ket thuc phat")
    total_viewers: int = Field(..., ge=0, description="Tong so nguoi xem")
    viewers: list[ViewerReportItem] = Field(
        ...,
        description="Danh sach nguoi xem trong luc quang cao chay",
    )

class AdReportResponse(BaseModel):
    ad_play_log_id: int = Field(..., description="ID log cua lan phat quang cao")
    advertisement_id: int = Field(..., description="ID quang cao")
    total_viewers: int = Field(..., ge=0)
    avg_look_duration: float = Field(..., ge=0)
    dominant_audience_segment_id: Optional[int] = Field(None, description="Segment chiem uu the")
    stats_date: date
    message: str = "ad report saved successfully"

