# Smart Billboard - He thong quang cao thong minh

## 1. Muc dich va muc tieu du an

Du an xay dung mot he thong quang cao thong minh co kha nang quan sat nguoi xem truoc man hinh, uoc luong thong tin nguoi xem va tu dong lua chon quang cao phu hop.

Muc dich chinh cua he thong:

- Tang muc do phu hop cua noi dung quang cao voi nguoi xem thuc te.
- Ghi nhan hieu qua phat quang cao dua tren so nguoi xem va thoi gian nhin man hinh.
- Ho tro quan ly quang cao, danh muc va bao cao thong ke thong qua dashboard.
- Tao mo hinh thuc nghiem ket hop thi giac may tinh, camera nhung va backend API.

Muc tieu ky thuat:

- Raspberry Pi dam nhan vai tro camera stream video ve may xu ly.
- May A xu ly camera, nhan dien nguoi xem, uoc luong tuoi/gioi tinh va gaze.
- May B nhan du lieu tu May A, chon quang cao, luu log va hien thi bao cao.
- He thong co the hoat dong gan thoi gian thuc trong moi truong demo.

## 2. Mo ta du an

He thong gom ba thanh phan chinh:

- Raspberry Pi: thiet bi camera, ket noi Wi-Fi/hotspot va stream video ve May A.
- May A: may xu ly thi giac may tinh, hien thi camera tracking va phat quang cao tren man hinh roi.
- May B: backend FastAPI, database SQLite va dashboard quan tri.

May A su dung MediaPipe de nhan dien landmark khuon mat, mo hinh PupilNet va cac model gaze de uoc luong huong nhin, dong thoi ket hop model age/gender de uoc luong tuoi va gioi tinh. Sau moi cua so lay mau, May A gui du lieu nguoi xem sang May B de yeu cau chon quang cao.

May B luu tru danh sach quang cao, danh muc, diem phu hop giua danh muc va nhom nguoi xem. Khi nhan request tu May A, May B chon mot quang cao active, tra ve `ad_id`, `media_filename` va `duration_seconds`. Sau khi May A phat xong quang cao, May A gui report ve May B de cap nhat log va thong ke.

## 3. Moi lien he giua Raspberry Pi, May A va May B

```text
Raspberry Pi Camera
        |
        | UDP video stream
        v
May A - Vision + Ad Player
        |
        | POST /api/advertisements/select
        v
May B - Backend + Database + Dashboard
        |
        | ad_id, media_filename, duration_seconds
        v
May A phat quang cao
        |
        | POST /api/ad-play-logs/report
        v
May B luu log va cap nhat bao cao
```

Vai tro cu the:

- Raspberry Pi ket noi vao hotspot/Wi-Fi va stream camera ve IP cua May A.
- May A nhan stream camera tai `udp://0.0.0.0:5000`, xu ly nhan dien va phat quang cao local.
- May A goi API May B qua dia chi cau hinh bang `MAY_B_IP` va `MAY_B_PORT`.
- May B chay FastAPI, nhan request chon quang cao va report ket qua phat quang cao.
- Dashboard cua May B dung de quan ly advertisement, xem play log, thong ke viewer va hieu qua category.

## 4. Cac luong hoat dong chinh

### Luong khoi dong he thong

1. Bat hotspot/Wi-Fi de Raspberry Pi va May A/May B co the ket noi mang.
2. Raspberry Pi ket noi mang va stream camera ve May A.
3. Khoi dong server May B.
4. Cau hinh May A voi IP/port cua May B va nguon video tu Raspberry Pi.
5. Chay May A de bat dau tracking va phat quang cao.

### Luong chon quang cao khi co nguoi xem

1. May A nhan frame camera tu Raspberry Pi.
2. May A detect khuon mat va uoc luong tuoi, gioi tinh, trang thai nhin man hinh.
3. May A gom du lieu nguoi xem trong khoang 4-5 giay.
4. May A gui request `POST /api/advertisements/select` sang May B.
5. May B map nguoi xem vao audience segment.
6. May B chon category co diem phu hop cao voi segment do.
7. May B chon random mot quang cao active trong category da chon.
8. May A nhan response va phat file quang cao tu thu muc local.

### Luong chon quang cao khi khong co nguoi xem

1. May A van thuc hien cua so lay mau trong khoang 4-5 giay.
2. Neu khong detect duoc nguoi xem, May A gui request voi `viewer_count = 0`.
3. May B thuc hien weighted random theo diem category:
   - category co average score thap se co ti le duoc chon cao hon.
   - category co score cao van co kha nang duoc phat, nhung ti le thap hon.
4. May A phat quang cao duoc May B tra ve.

### Luong report sau khi phat quang cao

1. Trong luc phat quang cao, May A tiep tuc tracking nguoi xem.
2. Khi quang cao ket thuc theo `duration_seconds`, May A tao report.
3. May A gui `POST /api/ad-play-logs/report` sang May B.
4. May B luu `ad_play_logs`, cap nhat `ad_performance_summary` va `category_audience_scores`.
5. Dashboard hien thi thong ke moi.

### Luong dashboard va quan tri

1. Nguoi quan tri mo dashboard May B.
2. Co the xem tong luot phat, tong viewer, thoi gian nhin trung binh.
3. Co the xem bieu do theo ngay, gioi tinh, nhom tuoi, category.
4. Co the quan ly danh sach advertisement, category va trang thai active.

## 5. He thong da lam duoc

He thong hien tai da co cac chuc nang chinh sau:

- Raspberry Pi stream camera ve May A qua UDP.
- May A hien thi tracking camera va phat quang cao tren man hinh roi.
- May A nhan dien khuon mat bang MediaPipe.
- May A uoc luong tuoi, gioi tinh va huong nhin cua nguoi xem.
- May A gom du lieu audience trong cua so thoi gian cau hinh duoc.
- May A gui API chon quang cao sang May B.
- May A gui report sau moi phien phat quang cao.
- May B cung cap API FastAPI cho selection va report.
- May B chon quang cao theo audience segment khi co nguoi xem.
- May B weighted random quang cao theo score category khi khong co nguoi xem.
- May B luu log phat quang cao va thong ke hieu qua.
- May B tu dong xoa `ad_play_logs` cu hon 7 ngay de han che database phinh to.
- Dashboard May B hien thi KPI, bieu do va bang thong ke.
- Dashboard co quan ly advertisement va hien thi cac truc bieu do ro rang hon.

## Cau truc thu muc chinh

```text
AgeGender/
├── GazeEstimation2020/     # May A: eye tracking, gaze estimation, ad player
├── machine_b/              # May B: FastAPI backend, SQLite DB, dashboard
├── agender/                # Ma nguon/model lien quan age-gender
├── MyDataSet/              # Du lieu huan luyen/thuc nghiem
└── README_PROJECT.md       # Tai lieu tong quan du an
```

## Ghi chu van hanh

- May A can cau hinh dung `VIDEO_SOURCE`, `MAY_B_IP`, `MAY_B_PORT` va `AD_MEDIA_ROOT`.
- May B phai chay voi `--host 0.0.0.0` de May A goi API qua mang LAN.
- Firewall cua May B can mo TCP port API, vi du `8000`.
- Firewall cua May A can mo UDP port video, vi du `5000`.
- `duration_seconds` trong database nen khop voi thoi luong that cua file quang cao de tranh cam giac video bi lap.
