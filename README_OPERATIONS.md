# Smart Billboard - Huong dan cai dat, van hanh va xu ly loi

Tai lieu nay tong hop cach van hanh he thong Smart Billboard gom Raspberry Pi Camera, May A xu ly thi giac may tinh va May B backend/dashboard.

## 1. Kien truc he thong

```text
Raspberry Pi Camera
        |
        | UDP video stream
        v
May A - Vision Processing + Ad Player
        |
        | POST /api/advertisements/select
        v
May B - FastAPI + SQLite + Dashboard
        |
        | ad_id, media_filename, duration_seconds
        v
May A phat quang cao
        |
        | POST /api/ad-play-logs/report
        v
May B luu log va cap nhat thong ke
```

### Raspberry Pi

- Ket noi vao hotspot cua May A.
- Su dung camera OV5647.
- Stream camera qua UDP den May A.

### May A

- Nhan camera stream tu Raspberry Pi.
- Detect khuon mat bang MediaPipe.
- Du doan nhom tuoi, gioi tinh va audience segment.
- Chay gaze estimation de xac dinh nguoi xem co nhin man hinh hay khong.
- Goi API May B de chon quang cao.
- Phat file quang cao local tren man hinh roi.
- Gui report sau khi phat xong quang cao.

### May B

- Chay FastAPI va SQLite.
- Chon quang cao dua tren audience segment.
- Khi khong co nguoi, weighted random category theo average score.
- Luu play log, performance summary va category score.
- Cung cap dashboard quan ly va thong ke.

## 2. Cau hinh mang

### Hotspot May A

```text
SSID: Hius
Password: 10101010
```

IP hotspot May A thuong la:

```text
192.168.137.1
```

Kiem tra:

```powershell
ipconfig
```

Tim adapter `Local Area Connection*` co IPv4 `192.168.137.1`.

### Tim IP Raspberry Pi

Xem danh sach thiet bi trong Windows Mobile Hotspot hoac chay:

```powershell
arp -a
```

IP Raspberry Pi co dang:

```text
192.168.137.xxx
```

Kiem tra SSH:

```powershell
Test-NetConnection 192.168.137.xxx -Port 22
ssh pi@192.168.137.xxx
```

### Kiem tra ket noi May A den May B

Vi du IP May B la `192.168.1.61`, port `8000`:

```powershell
Test-NetConnection 192.168.1.61 -Port 8000
```

Ket qua can co:

```text
TcpTestSucceeded : True
```

## 3. Thu tu chay he thong

Moi lan van hanh, chay theo thu tu:

1. Bat Wi-Fi va hotspot May A.
2. Cap nguon Raspberry Pi.
3. Chay server May B.
4. SSH vao Raspberry Pi va chay camera stream.
5. Chay `machine_a.py` tren May A.
6. Mo dashboard May B de theo doi.

## 4. Chay May B

Tren May B:

```powershell
cd E:\Semester6\PBL5\AgeGender\machine_b

.\venv\Scripts\python.exe -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

URL chinh:

```text
Health:    http://localhost:8000/health
API docs:  http://localhost:8000/docs
Dashboard: http://localhost:8000/dashboard
```

Mo firewall neu May A khong truy cap duoc:

```powershell
New-NetFirewallRule -DisplayName "Machine B API 8000" -Direction Inbound -Protocol TCP -LocalPort 8000 -Action Allow
```

## 5. Chay Raspberry Pi Camera

SSH vao Raspberry Pi:

```powershell
ssh pi@192.168.137.xxx
```

Kiem tra Pi thay May A:

```bash
ping -c 4 192.168.137.1
```

Kiem tra camera:

```bash
rpicam-hello --list-cameras
```

Camera OV5647 ho tro cac mode chinh:

```text
640x480
1296x972
1920x1080
2592x1944
```

Stream khuyen nghi can bang giua do net va hieu nang:

```bash
rpicam-vid --nopreview --width 960 --height 720 --framerate 20 --bitrate 2000000 --intra 20 -t 0 --inline --codec h264 -o - | ffmpeg -f h264 -i - -c:v copy -f mpegts udp://192.168.137.1:5000
```

Neu can nhan dien nguoi dung xa hon:

```bash
rpicam-vid --nopreview --width 1296 --height 972 --framerate 20 --bitrate 3000000 --intra 20 -t 0 --inline --codec h264 -o - | ffmpeg -f h264 -i - -c:v copy -f mpegts udp://192.168.137.1:5000
```

Giu terminal SSH nay chay trong suot qua trinh van hanh.

## 6. Chay May A

Tren May A:

```powershell
cd D:\PBL5\Gaze-Estimation\GazeEstimation2020

$env:MAY_B_IP = "192.168.1.61"
$env:MAY_B_PORT = "8000"
$env:VIDEO_SOURCE = "udp://0.0.0.0:5000"
$env:AUDIENCE_WINDOW_SECONDS = "4"
$env:AD_MEDIA_ROOT = "D:\PBL5\Gaze-Estimation\GazeEstimation2020\ads"

$env:PERFORMANCE_MODE = "balanced"
$env:VISION_PROCESS_WIDTH = "640"
$env:DEMOGRAPHIC_REFRESH_SECONDS = "1.0"
$env:DEMOGRAPHIC_CACHE_TTL_SECONDS = "3.0"
$env:FRAME_STALE_TIMEOUT = "0.75"

$env:AD_WINDOW_FULLSCREEN = "1"
$env:AD_WINDOW_WIDTH = "1920"
$env:AD_WINDOW_HEIGHT = "1080"

python machine_a.py
```

Neu dung stream `1296x972` va muon detect khuon mat xa hon:

```powershell
$env:VISION_PROCESS_WIDTH = "960"
```

Neu may yeu hoac bi lag:

```powershell
$env:PERFORMANCE_MODE = "aggressive"
$env:VISION_PROCESS_WIDTH = "640"
$env:DEMOGRAPHIC_REFRESH_SECONDS = "1.5"
```

## 7. Luong chon va phat quang cao

### Khi co nguoi

1. May A detect khuon mat.
2. May A gom audience segment trong khoang `AUDIENCE_WINDOW_SECONDS`.
3. May A gui:

```text
POST /api/advertisements/select
```

4. May B chon category co score cao nhat voi audience segment.
5. May B random mot quang cao active trong category.
6. May A phat file video local.

### Khi khong co nguoi

1. May A van gui selection request voi `viewer_count = 0`.
2. May B weighted random category theo cong thuc:

```text
weight = 1 / (average_score + epsilon)
```

Category score thap co ti le duoc phat cao hon, nhung category score cao van co kha nang duoc chon.

### Sau khi phat quang cao

May A gui:

```text
POST /api/ad-play-logs/report
```

May B luu report, cap nhat performance summary va category score. `ad_play_logs` cu hon 7 ngay duoc tu dong xoa.

## 8. Toi uu camera va nhan dien

Machine A da duoc toi uu theo cac huong:

- Gaze estimation van chay o ca luc lay mau va luc phat quang cao.
- Age/gender chay trong worker nen, tranh khoa giao dien camera.
- Ket qua age/gender duoc cache de khong nhap nhay.
- Frame bi skip van hien thi annotation gan nhat.
- UDP grabber chiu duoc loi frame ngan va tu mo lai ket noi neu loi lien tuc.
- Stream do phan giai cao duoc giu de crop khuon mat.
- MediaPipe va gaze chi xu ly tren ban resize gioi han boi `VISION_PROCESS_WIDTH`.
- Face crop co padding `30%` de phu hop hon voi model age/gender.

Neu nguoi dung dung xa khong detect duoc:

1. Tang stream len `1296x972`.
2. Tang `VISION_PROCESS_WIDTH` tu `640` len `960`.
3. Cai thien anh sang va dat camera ngang tam mat.
4. Khong nen tang ngay len Full HD vi se tang tai CPU va do tre.

## 9. Loi camera thuong gap

### Khong nhan duoc frame tren May A

```text
Khong nhan duoc frame tu udp://0.0.0.0:5000
```

Nguyen nhan:

- Raspberry Pi chua chay stream.
- Sai IP dich.
- Firewall May A chan UDP port `5000`.

### Raspberry Pi khong thay camera

```text
ERROR: no cameras available
```

Kiem tra:

```bash
rpicam-hello --list-cameras
```

Neu van khong thay, tat Pi, rut nguon va kiem tra day ribbon CSI.

### Camera bi timeout

```text
Camera frontend has timed out!
```

Thuong do day ribbon bi long, tiep xuc kem, day hong, camera hong hoac nguon Pi yeu.

### Camera dang bi tien trinh khac su dung

```text
Pipeline handler in use by another process
```

Tim tien trinh:

```bash
ps aux | grep -E "rpicam|libcamera|ffmpeg"
```

Dung tien trinh:

```bash
pkill -f rpicam-vid
pkill -f libcamera
pkill -f ffmpeg
```

## 10. Quan ly database May B

Tren May B:

```powershell
cd E:\Semester6\PBL5\AgeGender\machine_b
```

### Xem categories

```powershell
.\venv\Scripts\python.exe -c "import sqlite3; con=sqlite3.connect('machine_b.db'); [print(r) for r in con.execute('SELECT id, name, created_at FROM categories ORDER BY id')]"
```

### Xem audience segments

```powershell
.\venv\Scripts\python.exe -c "import sqlite3; con=sqlite3.connect('machine_b.db'); [print(r) for r in con.execute('SELECT id, gender, age_group, age_min, age_max FROM audience_segments ORDER BY gender, age_min')]"
```

### Xem advertisements

```powershell
.\venv\Scripts\python.exe -c "import sqlite3; con=sqlite3.connect('machine_b.db'); [print(r) for r in con.execute('SELECT id, title, media_filename, duration_seconds, is_active, category_id FROM advertisements ORDER BY id')]"
```

### Sua title va duration theo media filename

```powershell
.\venv\Scripts\python.exe -c "import sqlite3; con=sqlite3.connect('machine_b.db'); con.execute('UPDATE advertisements SET title=?, duration_seconds=? WHERE media_filename=?', ('Jollibee', 30, 'tech_01.mp4')); con.commit(); print(con.execute('SELECT id, title, media_filename, duration_seconds FROM advertisements WHERE media_filename=?', ('tech_01.mp4',)).fetchall())"
```

### Vo hieu hoa advertisement theo ID

```powershell
.\venv\Scripts\python.exe -c "import sqlite3; con=sqlite3.connect('machine_b.db'); con.execute('UPDATE advertisements SET is_active=0 WHERE id=?', (15,)); con.commit()"
```

Vo hieu hoa an toan hon xoa neu advertisement da co play log.

## 11. Quan ly video quang cao

- File video duoc luu local tren May A trong thu muc `ads`.
- `media_filename` trong database phai trung chinh xac voi ten file.
- `duration_seconds` nen trung voi thoi luong that cua video.
- Neu DB duration dai hon video, May A co the phat lai video de du thoi gian.
- OpenCV hien tai chi phat hinh anh, khong phat audio cua video.

## 12. Git va thu muc ads

Thu muc video quang cao khong duoc push len Git:

```gitignore
GazeEstimation2020/ads/
```

Neu video da tung duoc Git track:

```powershell
git rm -r --cached -- GazeEstimation2020/ads
```

Lenh nay chi go video khoi Git index, khong xoa file video tren may.

## 13. Dau hieu he thong hoat dong dung

Terminal May A:

```text
Loaded gaze, pupil, and age/gender models successfully.
Performance mode: balanced
Selected ad: ...
Ad report queued: ...
```

Terminal May B:

```text
POST /api/advertisements/select HTTP/1.1 200 OK
POST /api/ad-play-logs/report HTTP/1.1 201 Created
```

Dashboard:

```text
http://localhost:8000/dashboard
```
