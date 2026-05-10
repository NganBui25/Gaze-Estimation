# Category Audience Score Calculation

## Muc dich

`current_score` duoc dung de danh gia muc do phu hop giua:

- mot `audience_segment`
- va mot `category` quang cao

Gia tri nay duoc cap nhat sau moi lan `Machine A` gui report ve `Machine B`.

Code nguon hien tai:

- `E:\Semester6\PBL5\AgeGender\machine_b\app\services\category_audience_score_service.py`

## Dau vao

Ham `update_current_score(...)` nhan 5 dau vao:

- `category_id`
- `audience_segment_id`
- `viewer_count`
- `total_watch_duration`
- `ad_duration_seconds`

## Buoc 1: Tinh actual_score

Ham:

```python
calculator_actual_score(
    viewer_count,
    total_watch_duration,
    ad_duration_seconds,
)
```

Cong thuc:

```text
avg_watch_duration = total_watch_duration / viewer_count
actual_score = min(avg_watch_duration / ad_duration_seconds, 1.0)
```

Y nghia:

- `avg_watch_duration`: thoi gian xem trung binh cua segment do
- `actual_score`: muc do xem thuc te chuan hoa tren thang `0 -> 1`

Vi du:

```text
viewer_count = 2
total_watch_duration = 39.5
ad_duration_seconds = 30
```

Khi do:

```text
avg_watch_duration = 39.5 / 2 = 19.75
actual_score = 19.75 / 30 = 0.6583
```

## Buoc 2: Tinh trong so cua du lieu moi

Hai tham so cau hinh hien tai:

```python
PRIOR_WEIGHT = 10.0
MAX_VIEWER_WEIGHT = 5.0
```

Trong do:

- `PRIOR_WEIGHT`: trong so cua `current_score` cu
- `MAX_VIEWER_WEIGHT`: gioi han toi da trong so cua du lieu moi

Cong thuc:

```text
viewer_weight = min(viewer_count, MAX_VIEWER_WEIGHT)
```

Y nghia:

- viewer cang nhieu thi report moi cang dang tin
- nhung anh huong cua mot lan report duoc cap lai, khong de score nhay qua manh

Vi du:

```text
viewer_count = 2  -> viewer_weight = 2
viewer_count = 10 -> viewer_weight = 5
```

## Buoc 3: Cap nhat current_score

Cong thuc:

```text
new_current_score =
((old_current_score * PRIOR_WEIGHT) + (actual_score * viewer_weight))
/ (PRIOR_WEIGHT + viewer_weight)
```

Sau do:

```text
current_score = round(new_current_score, 4)
```

Y nghia:

- score cu duoc giu vai tro `prior`
- score moi la bang chung moi tu hanh vi xem thuc te
- he thong cap nhat bang trung binh co trong so

## Vi du day du

Gia su:

```text
old_current_score = 0.80
viewer_count = 2
total_watch_duration = 39.5
ad_duration_seconds = 30
PRIOR_WEIGHT = 10
MAX_VIEWER_WEIGHT = 5
```

### Tinh actual_score

```text
avg_watch_duration = 39.5 / 2 = 19.75
actual_score = 19.75 / 30 = 0.6583
```

### Tinh viewer_weight

```text
viewer_weight = min(2, 5) = 2
```

### Tinh current_score moi

```text
new_current_score =
((0.80 * 10) + (0.6583 * 2)) / (10 + 2)
= (8 + 1.3166) / 12
= 9.3166 / 12
= 0.7764
```

Ket qua:

```text
current_score = 0.7764
```

## Truc giac cua 2 tham so

### PRIOR_WEIGHT

- cang lon -> he thong cang tin vao lich su
- score thay doi cham hon

### MAX_VIEWER_WEIGHT

- cang lon -> mot lan report dong nguoi co the keo score manh hon
- cang nho -> du lieu moi bi han che anh huong nhieu hon

## Truong hop chua co score cu

Neu cap `category_id + audience_segment_id` chua ton tai, he thong tao moi:

```text
initial_score = 0.0
current_score = actual_score
```

Nghia la lan dau tien:

```text
current_score = actual_score
```

## Ket luan

He thong hien tai su dung co che:

1. Tinh `actual_score` tu thoi gian xem thuc te
2. Bien so viewer thanh `viewer_weight`
3. Tron `actual_score` voi `old_current_score` bang weighted average

Cong thuc tong quat:

```text
actual_score = min((total_watch_duration / viewer_count) / ad_duration_seconds, 1.0)
viewer_weight = min(viewer_count, MAX_VIEWER_WEIGHT)
new_current_score =
((old_current_score * PRIOR_WEIGHT) + (actual_score * viewer_weight))
/ (PRIOR_WEIGHT + viewer_weight)
```

Co che nay giup he thong:

- hoc tu du lieu xem thuc te
- nhung van giu duoc do on dinh
- tranh dao dong score qua manh sau mot lan report
