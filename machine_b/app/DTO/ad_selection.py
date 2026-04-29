from datetime import datetime

from pydantic import BaseModel, Field

class AdSelectionRequest(BaseModel):
    timestamp: datetime = Field(..., description="Thời điểm máy A gửi dữ liệu")
    viewer_count: int = Field(..., ge=1, description="Số người được phát hiện trong 3 giây")
    avg_age: int = Field(..., ge=0, description="Tuổi trung bình của nhóm người xem")
    majority_gender: str = Field(..., min_length=1, description="Giới tính chiếm đa số")

class AdSelectionResponse(BaseModel):
    ad_id: int = Field(..., description="ID quảng cáo được chọn")
    media_filename: str = Field(..., description="Tên file video quảng cáo")
    duration_seconds: int = Field(..., gt=0, description="Thời lượng quảng cáo")