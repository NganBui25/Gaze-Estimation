# Machine B - Backend cho hệ thống Smart Billboard

## 1. Vai trò của Machine B trong hệ thống

Machine B là backend trung tâm của hệ thống quảng cáo thông minh. Máy này nhận dữ liệu audience từ Machine A, chọn quảng cáo phù hợp, lưu lịch sử phát quảng cáo, cập nhật hiệu quả quảng cáo và cung cấp dashboard để quản lý/xem báo cáo.

Luồng tổng quát:

```text
Camera/Raspberry Pi
        |
        v
Machine A
- Detect người xem
- Ước lượng tuổi, giới tính, hướng nhìn
- Gom audience trong vài giây
        |
        | POST /api/advertisements/select
        v
Machine B
- Map audience vào audience_segment
- Chọn category phù hợp nhất
- Chọn quảng cáo active trong category đó
        |
        | Trả ad_id, media_filename, duration_seconds
        v
Machine A
- Phát video quảng cáo
- Theo dõi người xem trong lúc phát
        |
        | POST /api/ad-play-logs/report
        v
Machine B
- Lưu log phát quảng cáo
- Cập nhật báo cáo hiệu quả
- Cập nhật điểm category_audience_scores
- Hiển thị dashboard
```

Ứng dụng Machine B là một FastAPI app, entrypoint nằm ở:

```text
app/main.py
```

Chạy server:

```powershell
cd E:\Semester6\PBL5\AgeGender\machine_b
.\venv\Scripts\Activate.ps1
python -m uvicorn app.main:app --host 0.0.0.0 --port 5000
```

Các URL chính:

```text
API docs:  http://localhost:5000/docs
Dashboard: http://localhost:5000/dashboard
Health:    http://localhost:5000/health
```

Database mặc định là SQLite:

```text
machine_b.db
```

Kết nối DB được cấu hình trong:

```text
app/core/database.py
```

---

## 2. API đã tạo

### 2.1. `GET /`

Mục đích: kiểm tra nhanh server Machine B có chạy không và trả về các route gợi ý.

Response mẫu:

```json
{
  "message": "Machine B API is running",
  "docs": "/docs",
  "health": "/health",
  "dashboard": "/dashboard"
}
```

File xử lý:

```text
app/main.py
```

---

### 2.2. `GET /health`

Mục đích: health check cho Machine B. Route này dùng để Machine A hoặc người vận hành kiểm tra backend còn sống không.

Response mẫu:

```json
{
  "status": "ok",
  "service": "machine_b",
  "timestamp": "2026-05-28T01:00:00.000000+00:00"
}
```

File xử lý:

```text
app/controllers/health_controller.py
```

---

### 2.3. `POST /api/advertisements/select`

Mục đích: Machine A gửi thông tin nhóm người xem hiện tại cho Machine B. Machine B chọn quảng cáo phù hợp và trả lại thông tin video để Machine A phát.

File xử lý:

```text
app/controllers/advertisement_controller.py
app/services/advertisement_service.py
app/DTO/ad_selection.py
```

Request body:

```json
{
  "timestamp": "2026-05-28T08:30:00",
  "viewer_count": 3,
  "avg_age": 24,
  "majority_gender": "male"
}
```

Nếu không có người trong cửa sổ lấy mẫu 5 giây, Machine A vẫn gửi request:

```json
{
  "timestamp": "2026-05-28T08:30:00",
  "viewer_count": 0,
  "avg_age": 0,
  "majority_gender": "unknown"
}
```

Ý nghĩa field:

| Field | Kiểu | Ý nghĩa |
| --- | --- | --- |
| `timestamp` | datetime | Thời điểm Machine A gửi dữ liệu |
| `viewer_count` | int | Số người được phát hiện trong cửa sổ lấy mẫu |
| `avg_age` | int | Tuổi trung bình của nhóm audience |
| `majority_gender` | string | Giới tính chiếm đa số, thường là `male`, `female`, hoặc `unknown` |

Response body:

```json
{
  "ad_id": 5,
  "media_filename": "tech_ad_01.mp4",
  "duration_seconds": 15
}
```

Ý nghĩa response:

| Field | Kiểu | Ý nghĩa |
| --- | --- | --- |
| `ad_id` | int | ID quảng cáo được chọn trong DB |
| `media_filename` | string | Tên file video Machine A cần mở để phát |
| `duration_seconds` | int | Thời lượng phát quảng cáo |

Luồng xử lý:

```text
Request từ Machine A
    -> validate bằng AdSelectionRequest
    -> nếu viewer_count > 0: tìm audience_segment theo avg_age + majority_gender
    -> nếu viewer_count > 0: lấy category có current_score cao nhất với segment đó
    -> nếu viewer_count = 0: lấy category active có average current_score thấp nhất
    -> lấy random một advertisement active trong category
    -> trả về ad_id, media_filename, duration_seconds
```

Lỗi có thể gặp:

| HTTP code | Nguyên nhân |
| --- | --- |
| `400` | Dữ liệu không hợp lệ, ví dụ tuổi âm hoặc gender sai |
| `404` | Không tìm thấy audience segment, score hoặc advertisement phù hợp |

---

### 2.4. `POST /api/ad-play-logs/report`

Mục đích: sau khi Machine A phát xong quảng cáo, Machine A gửi report về Machine B để lưu hiệu quả quảng cáo và cập nhật điểm phù hợp.

File xử lý:

```text
app/controllers/ad_play_log_controller.py
app/services/ad_play_log_service.py
app/DTO/ad_report.py
```

Request body:

```json
{
  "ad_id": 5,
  "start_time": "2026-05-28T08:30:05",
  "end_time": "2026-05-28T08:30:20",
  "total_viewers": 2,
  "viewers": [
    {
      "estimated_age": 24,
      "gender": "male",
      "watch_duration": 12.5
    },
    {
      "estimated_age": 31,
      "gender": "female",
      "watch_duration": 10.0
    }
  ]
}
```

Ý nghĩa field:

| Field | Kiểu | Ý nghĩa |
| --- | --- | --- |
| `ad_id` | int | ID quảng cáo đã phát |
| `start_time` | datetime | Thời điểm bắt đầu phát |
| `end_time` | datetime | Thời điểm kết thúc phát |
| `total_viewers` | int | Tổng số người xem trong phiên phát |
| `viewers` | list | Danh sách người xem được Machine A tổng hợp |
| `viewers[].estimated_age` | int | Tuổi ước lượng của từng người xem |
| `viewers[].gender` | string | Giới tính của từng người xem |
| `viewers[].watch_duration` | float | Thời lượng xem quảng cáo của người đó |

Response body:

```json
{
  "ad_play_log_id": 12,
  "advertisement_id": 5,
  "total_viewers": 2,
  "avg_look_duration": 11.25,
  "dominant_audience_segment_id": 2,
  "stats_date": "2026-05-28",
  "message": "ad report saved successfully"
}
```

Luồng xử lý:

```text
Request từ Machine A
    -> kiểm tra advertisement tồn tại
    -> kiểm tra end_time > start_time
    -> kiểm tra total_viewers == số item trong viewers
    -> map từng viewer vào audience_segment
    -> gom viewer_count và total_watch_duration theo segment
    -> tính avg_look_duration toàn phiên
    -> tìm dominant_audience_segment_id
    -> tạo ad_play_logs
    -> cập nhật ad_performance_summary
    -> cập nhật category_audience_scores.current_score
    -> commit transaction
```

Lỗi có thể gặp:

| HTTP code | Nguyên nhân |
| --- | --- |
| `400` | Thời gian không hợp lệ hoặc `total_viewers` không bằng số viewer |
| `404` | Không tìm thấy quảng cáo hoặc audience segment |

---

## 3. Dashboard routes

Dashboard là giao diện web render bằng Jinja2 templates.

File xử lý:

```text
app/controllers/dashboard_controller.py
app/services/dashboard_service.py
app/repositories/dashboard_repo.py
app/templates/
```

### 3.1. `GET /dashboard`

Mục đích: trang tổng quan hệ thống.

Chức năng chính:

- Hiển thị KPI tổng:
  - tổng số lượt phát quảng cáo
  - tổng số viewer
  - thời gian xem trung bình
  - số quảng cáo đang active
- Biểu đồ daily play trend trong 7 ngày gần nhất.
- Biểu đồ viewer trend trong 7 ngày gần nhất.
- Biểu đồ gender distribution.
- Biểu đồ age group distribution theo nhóm tuổi:
  - `0-17`
  - `18-25`
  - `26-35`
  - `36-45`
  - `46-54`
  - `55-65`
  - `66-100`
- Top quảng cáo có hiệu quả tốt.
- Recent play logs.
- Top categories.

---

### 3.2. `GET /dashboard/advertisements`

Mục đích: quản lý danh sách quảng cáo.

Chức năng chính:

- Xem danh sách quảng cáo.
- Tìm kiếm theo text.
- Lọc theo category.
- Lọc theo trạng thái active/inactive.
- Xem metric của từng quảng cáo:
  - tổng số lượt phát
  - tổng viewer
  - average look duration
- Thêm quảng cáo mới.
- Sửa thông tin quảng cáo.
- Bật/tắt quảng cáo.
- Xóa quảng cáo nếu quảng cáo chưa có dữ liệu hiệu quả liên quan.

Các form action liên quan:

```text
POST /dashboard/advertisements/create
POST /dashboard/advertisements/{advertisement_id}/edit
POST /dashboard/advertisements/{advertisement_id}/toggle
POST /dashboard/advertisements/{advertisement_id}/delete
```

---

### 3.3. `GET /dashboard/categories`

Mục đích: xem hiệu quả theo category quảng cáo.

Chức năng chính:

- Xem danh sách category.
- Xem số quảng cáo trong từng category.
- Xem tổng lượt phát, tổng viewer, average look duration.
- Xem top audience segment phù hợp nhất với category đó dựa trên `current_score`.

---

### 3.4. `GET /dashboard/play-logs`

Mục đích: xem lịch sử các lần phát quảng cáo.

Chức năng chính:

- Hiển thị các lượt phát gần nhất.
- Mỗi log gồm:
  - thời điểm phát
  - quảng cáo
  - category
  - total viewers
  - avg look duration
  - dominant audience segment

---

### 3.5. `GET /dashboard/reports`

Mục đích: xem báo cáo hiệu quả quảng cáo theo thời gian.

Query parameters:

| Parameter | Ý nghĩa |
| --- | --- |
| `period` | Chu kỳ lọc, mặc định `daily` |
| `date_from` | Ngày bắt đầu |
| `date_to` | Ngày kết thúc |

Ví dụ:

```text
/dashboard/reports?period=daily&date_from=2026-05-01&date_to=2026-05-28
```

Chức năng chính:

- Tổng hợp total plays.
- Tổng hợp total viewers.
- Average look duration.
- Best advertisement.
- Biểu đồ trend.
- Biểu đồ phân bố age group.
- Biểu đồ average look duration theo category.
- Bảng report chi tiết theo ngày, quảng cáo, category và segment.

---

## 4. Database schema

### 4.1. `categories`

Mục đích: lưu nhóm/category quảng cáo. Category được dùng làm lớp trung gian giữa audience segment và advertisement.

| Cột | Kiểu | Ý nghĩa |
| --- | --- | --- |
| `id` | int | Khóa chính |
| `name` | string | Tên category, unique |
| `created_at` | datetime | Thời điểm tạo |

Ví dụ category:

```text
Entertainment
Fashion
Food
Health and Beauty
Tech
Travel
```

---

### 4.2. `advertisements`

Mục đích: lưu thông tin quảng cáo mà Machine A có thể phát.

| Cột | Kiểu | Ý nghĩa |
| --- | --- | --- |
| `id` | int | Khóa chính |
| `title` | string | Tên quảng cáo |
| `description` | text/null | Mô tả quảng cáo |
| `media_filename` | string | Tên file video quảng cáo |
| `duration_seconds` | int | Thời lượng video, phải lớn hơn 0 |
| `is_active` | bool | Quảng cáo có đang được phép chọn hay không |
| `category_id` | int | Khóa ngoại đến `categories.id` |
| `created_at` | datetime | Thời điểm tạo |
| `updated_at` | datetime | Thời điểm cập nhật |

Quan hệ:

```text
advertisements.category_id -> categories.id
```

Machine B chỉ chọn quảng cáo có:

```text
is_active = true
```

---

### 4.3. `audience_segments`

Mục đích: định nghĩa nhóm người xem theo giới tính và khoảng tuổi. Đây là đơn vị chính để chọn category và cập nhật score.

| Cột | Kiểu | Ý nghĩa |
| --- | --- | --- |
| `id` | int | Khóa chính |
| `gender` | string | `male`, `female`, hoặc nhóm được hỗ trợ |
| `age_group` | string | Label nhóm tuổi |
| `age_min` | int | Tuổi nhỏ nhất |
| `age_max` | int | Tuổi lớn nhất |
| `created_at` | datetime | Thời điểm tạo |

Ràng buộc:

```text
unique(gender, age_min, age_max)
age_min >= 0
age_max >= age_min
```

Bộ nhóm tuổi hiện tại:

```text
0-17
18-25
26-35
36-45
46-54
55-65
66-100
```

Ví dụ:

```text
male 18-25
female 46-54
```

---

### 4.4. `category_audience_scores`

Mục đích: lưu điểm phù hợp giữa từng category và từng audience segment.

Đây là bảng quan trọng nhất cho nghiệp vụ chọn quảng cáo. Khi Machine A gửi audience hiện tại, Machine B sẽ map audience vào `audience_segment`, sau đó nhìn vào bảng này để chọn category có `current_score` cao nhất.

| Cột | Kiểu | Ý nghĩa |
| --- | --- | --- |
| `id` | int | Khóa chính |
| `category_id` | int | Khóa ngoại đến `categories.id` |
| `audience_segment_id` | int | Khóa ngoại đến `audience_segments.id` |
| `initial_score` | float | Điểm khởi tạo ban đầu |
| `current_score` | float | Điểm hiện tại, được cập nhật sau mỗi report |
| `updated_at` | datetime | Thời điểm cập nhật |

Ràng buộc:

```text
unique(category_id, audience_segment_id)
initial_score >= 0
current_score >= 0
```

Quan hệ:

```text
category_audience_scores.category_id -> categories.id
category_audience_scores.audience_segment_id -> audience_segments.id
```

---

### 4.5. `ad_play_logs`

Mục đích: lưu log cho mỗi lần Machine A phát một quảng cáo.

| Cột | Kiểu | Ý nghĩa |
| --- | --- | --- |
| `id` | int/bigint | Khóa chính |
| `advertisement_id` | int | Quảng cáo đã phát |
| `played_at` | datetime | Thời điểm bắt đầu phát |
| `total_viewers` | int | Tổng số viewer trong phiên phát |
| `avg_look_duration` | float | Thời gian xem trung bình |
| `dominant_audience_segment_id` | int/null | Segment chiếm ưu thế trong phiên phát |

Quan hệ:

```text
ad_play_logs.advertisement_id -> advertisements.id
ad_play_logs.dominant_audience_segment_id -> audience_segments.id
```

Ràng buộc:

```text
total_viewers >= 0
avg_look_duration >= 0
```

---

### 4.6. `ad_performance_summary`

Mục đích: lưu dữ liệu tổng hợp hiệu quả quảng cáo theo ngày và audience segment.

Bảng này phục vụ dashboard/report và giúp hệ thống không phải tính lại toàn bộ từ log mỗi lần mở dashboard.

| Cột | Kiểu | Ý nghĩa |
| --- | --- | --- |
| `id` | int | Khóa chính |
| `advertisement_id` | int | Quảng cáo |
| `audience_segment_id` | int | Segment người xem |
| `stats_date` | date | Ngày thống kê |
| `play_count` | int | Số lượt phát |
| `viewer_count` | int | Tổng số viewer |
| `avg_look_duration` | float | Thời gian xem trung bình |
| `created_at` | date | Ngày tạo record |

Ràng buộc:

```text
unique(advertisement_id, audience_segment_id, stats_date)
play_count >= 0
viewer_count >= 0
avg_look_duration >= 0
```

Quan hệ:

```text
ad_performance_summary.advertisement_id -> advertisements.id
ad_performance_summary.audience_segment_id -> audience_segments.id
```

---

## 5. Nghiệp vụ chọn quảng cáo

Quá trình chọn quảng cáo nằm trong:

```text
app/services/advertisement_service.py
```

Hàm chính:

```python
select_ad(viewer_count, avg_age, majority_gender)
```

Machine B có hai nhánh xử lý:

- `viewer_count > 0`: giữ luồng cũ, chọn category phù hợp nhất với audience hiện tại.
- `viewer_count = 0`: không có người trong cửa sổ lấy mẫu, chọn category có average `current_score` thấp nhất để ưu tiên phát các nhóm quảng cáo ít được học/ít được phát.

### Nhánh A: Có người xem

#### Bước 1: Chuẩn hóa gender

Machine B nhận `majority_gender` từ Machine A. Gender được chuẩn hóa về lowercase.

Giá trị hợp lệ hiện tại:

```text
male
female
unknown
```

#### Bước 2: Tìm audience segment

Machine B tìm segment theo:

```text
gender = majority_gender
age_min <= avg_age <= age_max
```

Ví dụ:

```text
avg_age = 24
majority_gender = male
```

Khớp với:

```text
male 18-25
```

#### Bước 3: Chọn category tốt nhất cho segment

Machine B query bảng `category_audience_scores` theo `audience_segment_id`, sắp xếp:

```text
current_score DESC
id ASC
```

Category có `current_score` cao nhất được chọn.

Nếu có nhiều category cùng điểm, record có `id` nhỏ hơn được ưu tiên.

#### Bước 4: Chọn advertisement active trong category

Sau khi có `category_id`, Machine B lấy danh sách quảng cáo:

```text
category_id = selected_category_id
is_active = true
```

Sau đó chọn random một quảng cáo trong danh sách active.

Lý do chọn random: nếu một category có nhiều quảng cáo cùng nhóm nội dung, hệ thống không phát lặp duy nhất một quảng cáo.

### Nhánh B: Không có người xem (`viewer_count = 0`)

Nếu sau 5 giây Machine A không detect được người xem, Machine A vẫn gọi:

```text
POST /api/advertisements/select
```

với:

```json
{
  "timestamp": "2026-05-28T08:30:00",
  "viewer_count": 0,
  "avg_age": 0,
  "majority_gender": "unknown"
}
```

Machine B không dùng `avg_age` và `majority_gender` trong nhánh này. Thay vào đó, Machine B:

1. Tính average `current_score` của từng category trên toàn bộ audience segment.
2. Chỉ xét category đang có ít nhất một advertisement active.
3. Chọn category có average `current_score` thấp nhất.
4. Random một advertisement active trong category đó.
5. Trả `ad_id`, `media_filename`, `duration_seconds` cho Machine A.

Mục đích của nhánh này:

- Tránh một số category/quảng cáo không bao giờ được phát.
- Tạo cơ hội thu thập dữ liệu nếu có người xuất hiện trong lúc quảng cáo đang chạy.
- Giúp các category có score thấp được thử lại, thay vì bị đóng băng.

Nếu không tìm được category có score và active advertisement, Machine B fallback sang random một advertisement active bất kỳ.

### Bước cuối: Trả kết quả cho Machine A

Machine B trả:

```json
{
  "ad_id": 5,
  "media_filename": "tech_ad_01.mp4",
  "duration_seconds": 15
}
```

Machine A dùng `media_filename` để mở file video và phát trên màn hình.

---

## 6. Nghiệp vụ lưu report và cập nhật điểm số

Phần này là nghiệp vụ quan trọng nhất của Machine B.

File chính:

```text
app/services/ad_play_log_service.py
app/services/category_audience_score_service.py
app/services/ad_performance_summary_service.py
```

Sau khi Machine A phát xong quảng cáo, Machine A gửi report về:

```text
POST /api/ad-play-logs/report
```

Report có thông tin:

```text
ad_id
start_time
end_time
total_viewers
viewers[]
```

Mỗi viewer có:

```text
estimated_age
gender
watch_duration
```

### 6.1. Validate report

Machine B kiểm tra:

```text
advertisement phải tồn tại
end_time phải lớn hơn start_time
total_viewers phải bằng len(viewers)
```

Nếu một trong các điều kiện sai, request bị từ chối.

### 6.2. Map từng viewer vào audience segment

Với từng viewer:

```text
estimated_age + gender -> audience_segment
```

Ví dụ:

```text
viewer A: age=24, gender=male   -> male 18-25
viewer B: age=31, gender=female -> female 26-35
```

Sau đó Machine B gom thống kê theo segment:

```text
grouped_stats[segment_id].viewer_count += 1
grouped_stats[segment_id].total_watch_duration += watch_duration
```

Ví dụ:

```text
segment male 18-25:
  viewer_count = 2
  total_watch_duration = 21.5

segment female 26-35:
  viewer_count = 1
  total_watch_duration = 8.0
```

### 6.3. Tính avg look duration của lần phát

Machine B tính:

```text
total_watch_duration = tổng watch_duration của toàn bộ viewers
avg_look_duration = total_watch_duration / total_viewers
```

Ví dụ:

```text
total_watch_duration = 31.5
total_viewers = 3
avg_look_duration = 10.5
```

Giá trị này được lưu vào `ad_play_logs.avg_look_duration`.

### 6.4. Xác định dominant audience segment

Dominant segment là segment chiếm ưu thế trong lần phát quảng cáo.

Code chọn dominant segment bằng tiêu chí:

```text
1. viewer_count cao hơn
2. nếu viewer_count bằng nhau thì total_watch_duration cao hơn
3. nếu vẫn bằng nhau thì segment_id nhỏ hơn
```

Nó được lưu vào:

```text
ad_play_logs.dominant_audience_segment_id
```

### 6.5. Tạo ad_play_logs

Machine B tạo một record trong `ad_play_logs`:

```text
advertisement_id = ad_id
played_at = start_time
total_viewers = total_viewers
avg_look_duration = avg_look_duration
dominant_audience_segment_id = dominant_segment_id
```

Record này đại diện cho một lần phát quảng cáo.

### 6.6. Cập nhật ad_performance_summary

Với mỗi segment trong `grouped_stats`, Machine B cập nhật bảng `ad_performance_summary`.

Khóa unique:

```text
advertisement_id + audience_segment_id + stats_date
```

Nếu chưa có summary:

```text
play_count = 1
viewer_count = viewer_count_increment
avg_look_duration = total_watch_duration_increment / viewer_count_increment
```

Nếu đã có summary:

```text
old_total_watch_duration = old_avg_look_duration * old_viewer_count
new_total_view_count = old_viewer_count + viewer_count_increment
new_total_watch_duration = old_total_watch_duration + total_watch_duration_increment

play_count += 1
viewer_count = new_total_view_count
avg_look_duration = new_total_watch_duration / new_total_view_count
```

Mục đích: giữ thống kê tích lũy theo ngày, theo quảng cáo và theo audience segment.

### 6.7. Cập nhật category_audience_scores.current_score

Đây là bước làm cho hệ thống "học" từ hiệu quả xem thực tế.

Với mỗi segment trong report, Machine B cập nhật điểm cho cặp:

```text
category của quảng cáo vừa phát
audience_segment của nhóm viewer đó
```

Ví dụ:

```text
Quảng cáo vừa phát thuộc category Tech
Viewer thuộc segment male 18-25

=> cập nhật score của:
Tech + male 18-25
```

File xử lý:

```text
app/services/category_audience_score_service.py
```

Các hằng số:

```python
MAX_PRIOR_WEIGHT = 100.0
MAX_VIEWER_WEIGHT = 20.0
```

Ý nghĩa:

| Hằng số | Ý nghĩa |
| --- | --- |
| `MAX_PRIOR_WEIGHT` | Giới hạn độ nặng tối đa của dữ liệu lịch sử |
| `MAX_VIEWER_WEIGHT` | Giới hạn độ nặng tối đa của dữ liệu mới trong một lần report |

#### Bước 1: Tính `actual_score`

`actual_score` là điểm thực tế của lần phát quảng cáo, dựa trên tỷ lệ thời gian xem.

Công thức:

```text
avg_watch_duration = total_watch_duration / viewer_count
actual_score = min(avg_watch_duration / ad_duration_seconds, 1.0)
```

Ý nghĩa:

```text
actual_score = 0.0  -> gần như không xem
actual_score = 0.5  -> trung bình xem khoảng một nửa thời lượng quảng cáo
actual_score = 1.0  -> trung bình xem hết quảng cáo hoặc hơn
```

Ví dụ:

```text
viewer_count = 2
total_watch_duration = 20
ad_duration_seconds = 15

avg_watch_duration = 20 / 2 = 10
actual_score = min(10 / 15, 1.0) = 0.6667
```

Nếu viewer xem trung bình lâu hơn thời lượng quảng cáo:

```text
avg_watch_duration = 18
ad_duration_seconds = 15
actual_score = min(18 / 15, 1.0) = 1.0
```

Score bị chặn tối đa ở `1.0` để tránh một lần report kéo điểm tăng quá mức.

#### Bước 2: Tính `prior_weight` và `viewer_weight`

Phiên bản hiện tại dùng dữ liệu thống kê thật nhưng vẫn có giới hạn để score không bị đóng băng hoặc dao động quá mạnh.

`prior_weight` lấy từ số viewer lịch sử đã xem các quảng cáo thuộc cùng category và cùng audience segment:

```text
historical_viewer_count =
tổng viewer_count trong ad_performance_summary
của các advertisement thuộc category đó
và audience_segment đó
```

Sau đó giới hạn:

```text
prior_weight = min(historical_viewer_count, MAX_PRIOR_WEIGHT)
```

`viewer_weight` lấy từ số viewer mới trong report hiện tại:


```text
viewer_weight = min(viewer_count, MAX_VIEWER_WEIGHT)
```

Với:

```text
MAX_PRIOR_WEIGHT = 100.0
MAX_VIEWER_WEIGHT = 20.0
```

Ví dụ:

```text
historical_viewer_count = 0     -> prior_weight = 0
historical_viewer_count = 40    -> prior_weight = 40
historical_viewer_count = 1000  -> prior_weight = 100

viewer_count = 1   -> viewer_weight = 1
viewer_count = 10  -> viewer_weight = 10
viewer_count = 50  -> viewer_weight = 20
```

#### Bước 3: Trộn score cũ và score mới

Công thức:

```text
new_current_score =
((old_current_score * prior_weight) + (actual_score * viewer_weight))
/ (prior_weight + viewer_weight)
```

Sau đó:

```text
current_score = round(new_current_score, 4)
```

Ý nghĩa:

- `old_current_score` là kinh nghiệm lịch sử.
- `actual_score` là bằng chứng mới từ lần phát vừa xong.
- `prior_weight` dựa trên số viewer lịch sử thật, nhưng bị chặn bởi `MAX_PRIOR_WEIGHT`.
- `viewer_weight` dựa trên số viewer mới thật, nhưng bị chặn bởi `MAX_VIEWER_WEIGHT`.
- Hai giới hạn này giúp score vừa có tính thống kê, vừa không bị lịch sử quá lớn làm đóng băng hoặc report bất thường kéo quá mạnh.

#### Ví dụ đầy đủ

Giả sử:

```text
Category: Tech
Audience segment: male 18-25
old_current_score = 0.8000
historical_viewer_count = 120
viewer_count = 2
total_watch_duration = 20
ad_duration_seconds = 15
MAX_PRIOR_WEIGHT = 100
MAX_VIEWER_WEIGHT = 20
```

Tính `actual_score`:

```text
avg_watch_duration = 20 / 2 = 10
actual_score = min(10 / 15, 1.0) = 0.6667
```

Tính `prior_weight` và `viewer_weight`:

```text
prior_weight = min(120, 100) = 100
viewer_weight = min(2, 20) = 2
```

Tính score mới:

```text
new_current_score =
((0.8000 * 100) + (0.6667 * 2)) / (100 + 2)
= (80.0000 + 1.3334) / 102
= 0.7974
```

Kết quả:

```text
current_score = 0.7974
```

Trường hợp report tốt hơn điểm cũ thì score tăng. Trường hợp report kém hơn điểm cũ thì score giảm.

#### Trường hợp chưa có score

Nếu cặp:

```text
category_id + audience_segment_id
```

chưa tồn tại trong `category_audience_scores`, hệ thống tạo mới:

```text
initial_score = 0.0
current_score = actual_score
```

#### Tác động đến lần chọn quảng cáo sau

Sau khi `current_score` được cập nhật, lần sau nếu Machine A gửi audience cùng segment, Machine B sẽ lại chọn category có `current_score` cao nhất cho segment đó.

Nói cách khác:

```text
Report sau mỗi lần phát
    -> cập nhật current_score
    -> ảnh hưởng category được chọn trong tương lai
```

Đây là vòng học của hệ thống.

---

## 7. Ghi chú vận hành

### 7.1. Machine A cần trỏ đúng IP Machine B

Nếu Machine A và Machine B chạy trên hai máy khác nhau, Machine A cần cấu hình IP của Machine B:

```powershell
$env:MAY_B_IP="192.168.1.20"
python machine_a.py
```

Machine B cần chạy với:

```powershell
python -m uvicorn app.main:app --host 0.0.0.0 --port 5000
```

Nếu Windows Firewall chặn port `5000`, mở port:

```powershell
New-NetFirewallRule -DisplayName "Machine B API Port 5000" -Direction Inbound -Protocol TCP -LocalPort 5000 -Action Allow
```

### 7.2. Xem database

Mở trực tiếp bằng DB Browser for SQLite:

```text
machine_b/machine_b.db
```

Hoặc xem bảng bằng Python:

```powershell
python -c "import sqlite3; con=sqlite3.connect('machine_b/machine_b.db'); print(con.execute('select * from audience_segments').fetchall())"
```

### 7.3. Age group hiện tại

Dashboard và database đang dùng bộ nhóm tuổi:

```text
0-17
18-25
26-35
36-45
46-54
55-65
66-100
```

Không nên dùng các khoảng bị chồng lấn như `0-18` và `18-25`, vì code tìm segment bằng điều kiện:

```text
age_min <= age <= age_max
```

Nếu hai khoảng chồng nhau ở tuổi `18`, hệ thống có thể map không rõ ràng.
