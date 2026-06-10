# Luong video Raspberry Pi Camera den May A

## 1. Gioi thieu

Trong he thong Smart Billboard, Raspberry Pi chi dam nhan viec:

1. Lay hinh anh tu Raspberry Pi Camera.
2. Ma hoa hinh anh thanh video H.264.
3. Dong goi H.264 vao MPEG-TS bang FFmpeg.
4. Phat luong MPEG-TS qua mang den May A.

May A nhan va giai ma video, sau do dua tung frame vao pipeline nhan dien
khuon mat, tuoi, gioi tinh va huong nhin.

Luong du lieu tong quat:

```text
Raspberry Pi Camera
  -> rpicam-vid ma hoa H.264
  -> FFmpeg dong goi MPEG-TS
  -> UDP port 5000
  -> OpenCV/FFmpeg tren May A
  -> LatestFrameGrabber
  -> MediaPipe + PupilNet + Age/Gender
  -> Tracking View
```

## 2. Lenh phat video tren Raspberry Pi

Lenh dang duoc su dung:

```bash
rpicam-vid --nopreview \
  --width 960 \
  --height 720 \
  --framerate 20 \
  --bitrate 3000000 \
  --intra 20 \
  -t 0 \
  --inline \
  --codec h264 \
  -o - |
ffmpeg -f h264 -i - -c:v copy -f mpegts udp://192.168.137.1:5000
```

Y nghia cac tham so quan trong:

| Tham so | Y nghia |
|---|---|
| `--width 960 --height 720` | Do phan giai video camera |
| `--framerate 20` | Camera phat 20 frame moi giay |
| `--bitrate 3000000` | Bitrate H.264 khoang 3 Mbps |
| `--intra 20` | Tao keyframe sau moi 20 frame, tuong duong khoang 1 giay |
| `--inline` | Gui SPS/PPS kem keyframe de decoder co the bat dau va phuc hoi luong |
| `--codec h264` | Ma hoa video bang H.264 |
| `-t 0` | Phat lien tuc den khi nguoi dung dung chuong trinh |
| `-c:v copy` | FFmpeg chi dong goi, khong ma hoa lai video |
| `-f mpegts` | Dong goi H.264 vao MPEG Transport Stream |
| `udp://192.168.137.1:5000` | Gui den dia chi May A, port UDP 5000 |

Dia chi `192.168.137.1` phai la dia chi IP thuc te cua May A tren mang ma
Raspberry Pi dang ket noi.

## 3. Cach chay May A

Mo PowerShell tai thu muc `GazeEstimation2020`:

```powershell
$env:VIDEO_SOURCE = "udp://0.0.0.0:5000"
$env:MAY_B_IP = "192.168.1.10"
python machine_a.py
```

`udp://0.0.0.0:5000` co nghia la May A lang nghe du lieu UDP tren port
`5000` tu moi card mang.

Can cho phep Python/OpenCV nhan UDP port `5000` qua Windows Firewall.

## 4. Cac file chiu trach nhiem truc tiep

### `machine_a/config.py`

Chua toan bo cau hinh dau vao video:

- `VIDEO_SOURCE`: dia chi camera hoac luong mang.
- `FRAME_WIDTH`, `FRAME_HEIGHT`: kich thuoc frame mong muon.
- `FRAME_BUFFERSIZE`: so frame OpenCV giu trong buffer.
- `FRAME_STALE_TIMEOUT`: thoi gian toi da mot frame duoc xem la con moi.
- `VIDEO_RECONNECT_FAILED_READS`: so lan doc loi truoc khi mo lai decoder.
- `VIDEO_IDLE_SLEEP_SECONDS`: khoang nghi khi chua co frame moi.
- `VIDEO_UDP_FIFO_SIZE`: FIFO FFmpeg dung khi nhan UDP.
- `PERFORMANCE_MODE`: quyet dinh tan suat xu ly AI.

Day la file can xem dau tien khi muon thay dia chi camera, port, do phan giai
hoac che do hieu nang.

### `machine_a/video.py`

Chiu trach nhiem nhan va giai ma luong video:

- Lop `LatestFrameGrabber` mo `cv2.VideoCapture`.
- Uu tien backend `cv2.CAP_FFMPEG`.
- Them cac tuy chon FIFO khi nguon la UDP.
- Doc camera lien tuc trong mot thread rieng.
- Chi luu frame moi nhat de pipeline AI khong bi ton dong frame cu.
- Gan `frame_sequence` cho moi frame moi.
- Danh dau frame cu bang `FRAME_STALE_TIMEOUT`.
- Tu dong dong va mo lai capture khi doc that bai lien tiep.

Loi H.264 nhu sau duoc sinh ra tai backend FFmpeg trong luong nay:

```text
left block unavailable for requested intra mode
error while decoding MB
bytestream
```

Nhung loi tren cho biet decoder nhan du lieu H.264 thieu hoac hong.

### `machine_a.py`

La chuong trinh dieu phoi chinh cua May A:

- Tao `LatestFrameGrabber(VIDEO_SOURCE)`.
- Cho toi da 10 giay de nhan frame dau tien.
- Doc `(ret, frame, source_frame_sequence)` tu grabber.
- Chi chay AI khi `source_frame_sequence` thay doi.
- Hien thi frame camera trong cua so `Tracking View`.
- Mo va phat cac file video quang cao cuc bo.

File nay su dung hai loai video khac nhau:

1. Video camera Raspberry Pi, nhan qua `LatestFrameGrabber`.
2. Video quang cao cuc bo, mo bang `cv2.VideoCapture(selected_ad_path)`.

Vi ca hai deu co the la H.264, log FFmpeg trong terminal co the den tu camera
hoac file quang cao.

## 5. Cac file lien quan gian tiep

| File | Vai tro |
|---|---|
| `machine_a/pipeline.py` | Nhan frame camera da giai ma va chay nhan dien |
| `machine_a/vision_utils.py` | Cat mat, chay PupilNet va xu ly frame |
| `machine_a/ui.py` | Tao va sap xep cua so hien thi |
| `machine_a/models.py` | Tai cac model AI dung de xu ly frame |
| `requirements.txt` | Khai bao OpenCV, MediaPipe, PyTorch va cac thu vien |
| `README_GAZE_PUPILNET.md` | Tai lieu pipeline nhan dien huong nhin |

## 6. Cac file khong phai luong camera chinh

- `EyeTracking.py`: demo webcam doc lap, khong phai chuong trinh Smart Billboard.
- `calibrate_mediapipe.py`: cong cu hieu chinh bang webcam.
- `machine_a_copy.py`: ban sao/legacy, khong duoc `machine_a.py` import.
- `example.avi`: video mau de kiem thu, khong phai luong Raspberry Pi.

## 7. Tai sao UDP co the gay vo hinh

UDP uu tien do tre thap, nhung khong dam bao goi tin den day du va dung thu tu.

Neu mot goi MPEG-TS/H.264 bi mat:

1. Decoder khong co du du lieu de dung macroblock.
2. FFmpeg bao loi `error while decoding MB`.
3. Frame co the bi vo thanh cac khoi vuong.
4. Cac P-frame tiep theo co the tiep tuc dua tren frame hong.
5. Hinh anh thuong chi phuc hoi tot khi nhan duoc keyframe/IDR day du.

`--intra 20` va `--inline` giup luong co kha nang phuc hoi, nhung khong ngan
duoc mat goi UDP.

Nhung nguyen nhan mat goi pho bien:

- Raspberry Pi va May A ket noi bang Wi-Fi/hotspot khong on dinh.
- Bitrate cao hon kha nang duong truyen tai thoi diem do.
- May A dang qua tai va thread nhan video khong doc du lieu kip.
- Windows Firewall, driver Wi-Fi hoac card mang lam roi goi.
- Raspberry Pi bi nong va giam hieu nang.

## 8. Phan biet loi camera va loi video quang cao

De xac dinh decoder nao dang loi:

1. Chay May A nhung tam thoi khong cho phat quang cao.
2. Neu van co loi H.264, loi den tu luong camera Raspberry Pi.
3. Neu chi loi ngay sau dong `Selected ad`, kiem tra file MP4 quang cao.
4. Neu `Tracking View` bi vo khoi, luong camera da bi hong.
5. Neu chi `Ad Display` bi vo khoi, file quang cao hoac decoder quang cao bi loi.

## 9. Kiem tra nhanh ket noi

Tren May A:

```powershell
ipconfig
```

Xac nhan IP ma Raspberry Pi gui den ton tai tren May A.

Tren Raspberry Pi:

```bash
ping 192.168.137.1
```

Neu Raspberry Pi khong ping duoc May A, luong UDP cung se khong den dung dich.

Co the thu nhan video ma khong chay AI bang FFplay neu May A co FFmpeg:

```powershell
ffplay -fflags nobuffer -flags low_delay -i "udp://0.0.0.0:5000"
```

Neu FFplay cung bi vo hinh, nguyen nhan nam o lenh phat, mang hoac du lieu camera,
khong nam trong pipeline nhan dien gaze.

## 10. Luu y khi thay doi giao thuc

Neu chuyen tu UDP sang TCP, SRT, RTSP hoac giao thuc khac, can thay doi dong bo:

1. Lenh phat tren Raspberry Pi.
2. Bien `VIDEO_SOURCE` tren May A.
3. Kiem tra OpenCV/FFmpeg tren May A co ho tro giao thuc do.

Khong chi thay `udp://` thanh `tcp://` o mot phia, vi phia phat va phia nhan
phai cung giao thuc va cung vai tro client/server.
