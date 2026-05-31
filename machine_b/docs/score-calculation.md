# Category Audience Score Calculation

## Muc dich

`current_score` duoc dung de danh gia muc do phu hop giua:

- mot `audience_segment`
- va mot `category` quang cao

Gia tri nay duoc cap nhat sau moi lan `Machine A` gui report ve `Machine B`.

Code nguon hien tai:

```text
app/services/category_audience_score_service.py
app/repositories/category_audience_score_repo.py
```

## Dau vao

Ham `update_current_score(...)` nhan 5 dau vao:

- `category_id`
- `audience_segment_id`
- `viewer_count`: so viewer moi trong report hien tai cua segment do
- `total_watch_duration`: tong thoi gian xem cua cac viewer moi trong segment do
- `ad_duration_seconds`: thoi luong quang cao vua phat

## Buoc 1: Tinh actual_score

Cong thuc:

```text
avg_watch_duration = total_watch_duration / viewer_count
actual_score = min(avg_watch_duration / ad_duration_seconds, 1.0)
```

Y nghia:

- `avg_watch_duration`: thoi gian xem trung binh cua segment trong lan report moi
- `actual_score`: hieu qua thuc te cua lan phat, chuan hoa tren thang `0 -> 1`
- `1.0`: xem nhu muc toi da, tranh viec mot report keo diem tang qua manh

Vi du:

```text
viewer_count = 2
total_watch_duration = 39.5
ad_duration_seconds = 30

avg_watch_duration = 39.5 / 2 = 19.75
actual_score = 19.75 / 30 = 0.6583
```

## Buoc 2: Tinh prior_weight tu viewer lich su

Phien ban moi dung so viewer lich su that thay vi mot hang so `PRIOR_WEIGHT` co dinh.

`historical_viewer_count` duoc tinh bang:

```text
Tong viewer_count trong ad_performance_summary
cua cac advertisement thuoc category_id
va audience_segment_id dang cap nhat
```

Sau do gioi han bang:

```python
MAX_PRIOR_WEIGHT = 100.0
```

Cong thuc:

```text
prior_weight = min(historical_viewer_count, MAX_PRIOR_WEIGHT)
```

Y nghia:

- Neu segment/category da co nhieu viewer lich su, score cu dang tin hon.
- Neu lich su qua lon, chi tinh toi da `MAX_PRIOR_WEIGHT` de score khong bi dong bang.
- Neu chua co viewer lich su, `prior_weight = 0`, lan report dau tien se gan score theo `actual_score`.

Vi du:

```text
historical_viewer_count = 0     -> prior_weight = 0
historical_viewer_count = 40    -> prior_weight = 40
historical_viewer_count = 1000  -> prior_weight = 100
```

## Buoc 3: Tinh viewer_weight tu viewer moi

Report moi co nhieu viewer thi dang tin hon report co it viewer, nhung van can gioi han de mot lan report bat thuong khong keo score qua manh.

Tham so:

```python
MAX_VIEWER_WEIGHT = 20.0
```

Cong thuc:

```text
viewer_weight = min(viewer_count, MAX_VIEWER_WEIGHT)
```

Vi du:

```text
viewer_count = 1   -> viewer_weight = 1
viewer_count = 10  -> viewer_weight = 10
viewer_count = 50  -> viewer_weight = 20
```

## Buoc 4: Cap nhat current_score

Neu da co du lieu lich su:

```text
new_current_score =
((old_current_score * prior_weight) + (actual_score * viewer_weight))
/ (prior_weight + viewer_weight)
```

Neu `prior_weight = 0`:

```text
new_current_score = actual_score
```

Sau do:

```text
current_score = round(new_current_score, 4)
```

## Vi du day du

Gia su:

```text
old_current_score = 0.80
historical_viewer_count = 120
viewer_count = 2
total_watch_duration = 39.5
ad_duration_seconds = 30
MAX_PRIOR_WEIGHT = 100
MAX_VIEWER_WEIGHT = 20
```

### Tinh actual_score

```text
avg_watch_duration = 39.5 / 2 = 19.75
actual_score = 19.75 / 30 = 0.6583
```

### Tinh prior_weight va viewer_weight

```text
prior_weight = min(120, 100) = 100
viewer_weight = min(2, 20) = 2
```

### Tinh current_score moi

```text
new_current_score =
((0.80 * 100) + (0.6583 * 2)) / (100 + 2)
= (80 + 1.3166) / 102
= 0.7972
```

Ket qua:

```text
current_score = 0.7972
```

## Vi sao dung cach nay?

### So voi cach cu

Cach cu:

```text
new_score =
((old_score * PRIOR_WEIGHT) + (actual_score * viewer_weight))
/ (PRIOR_WEIGHT + viewer_weight)
```

Trong do `PRIOR_WEIGHT = 10` la mot hang so gia lap do tin cay cua diem cu.

### Cach moi

Cach moi:

```text
prior_weight = min(historical_viewer_count, MAX_PRIOR_WEIGHT)
viewer_weight = min(new_viewer_count, MAX_VIEWER_WEIGHT)
```

No tot hon vi:

- dua tren so viewer lich su that
- report moi co nhieu viewer se co anh huong lon hon report it viewer
- lich su qua lon khong lam score bi dong bang
- mot report bat thuong khong lam score nhay qua manh

## Ket luan

Cong thuc tong quat hien tai:

```text
actual_score = min((total_watch_duration / viewer_count) / ad_duration_seconds, 1.0)
prior_weight = min(historical_viewer_count, MAX_PRIOR_WEIGHT)
viewer_weight = min(viewer_count, MAX_VIEWER_WEIGHT)

if prior_weight == 0:
    new_current_score = actual_score
else:
    new_current_score =
    ((old_current_score * prior_weight) + (actual_score * viewer_weight))
    / (prior_weight + viewer_weight)
```

Co che nay giup he thong:

- hoc tu du lieu xem thuc te
- dung trong so thong ke tu viewer lich su
- van giu duoc do on dinh
- tranh dao dong score qua manh sau mot lan report
