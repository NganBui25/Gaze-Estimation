# Kiểm tra NHANH: vector đặc trưng 14 chiều mà EyeTracking.py dựng có KHỚP với
# định dạng dữ liệu huấn luyện (unityeyes_processed_14d.csv / best_model) không —
# TRƯỚC khi chạy webcam.
#
# Cách làm: dựng lại đặc trưng từ chính các file UnityEyes json (notebooks/imgs/*.json)
# theo NHIỀU thứ tự điểm khác nhau, rồi:
#   1) tìm hàng CSV gần nhất (khoảng cách ~0 => đúng công thức tạo CSV),
#   2) đưa qua best_model so với target của hàng CSV khớp đó.
import os, sys, glob, json
import numpy as np
import pandas as pd
import joblib
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

def find(paths):
    for p in paths:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(paths)

CSV = find(["../EyeTracking/data/unityeyes_processed_14d.csv",
            "EyeTracking/data/unityeyes_processed_14d.csv",
            "data/unityeyes_processed_14d.csv"])
MODEL = find(["models/best_model.joblib", "../EyeTracking/models/best_model.joblib"])
IMGS = find(["notebooks/imgs", "imgs"])
print(f"CSV   : {CSV}\nMODEL : {MODEL}\nIMGS  : {IMGS}\n")

# ============ PART A — Hợp đồng đặc trưng của CSV ============
df = pd.read_csv(CSV)
fc = [c for c in df.columns if c.startswith("point_")]
X = df[fc].values.astype(np.float32)            # (N,14)
Y = df[["target_yaw", "target_pitch"]].values
P = X.reshape(-1, 7, 2)
print("=" * 64)
print("PART A — Hợp đồng đặc trưng CSV (model mong đợi đúng định dạng này)")
print("=" * 64)
roles = []
for i in range(7):
    mag = np.linalg.norm(P[:, i, :], axis=1)
    print(f"  point_{i+1}: mean=({P[:,i,0].mean():+.3f},{P[:,i,1].mean():+.3f})  "
          f"|mag|={mag.mean():.3f}  std|mag|={mag.std():.3f}")
print("  => point_1=khóe TRONG (0,0); point_2=khóe NGOÀI (|mag|=1); "
      "point_7=đồng tử (std lớn nhất).")
print(f"  Hướng khóe trong->ngoài: x={'+' if P[:,1,0].mean()>0 else '-'} "
      f"(canonical của CSV)\n")

# ============ PART B — best_model round-trip trên CSV ============
model = joblib.load(MODEL)
n = min(20000, len(X))
pred = model.predict(X[:n])
mae = (np.abs(pred[:, 0] - Y[:n, 0]).mean() + np.abs(pred[:, 1] - Y[:n, 1]).mean()) / 2
print("=" * 64)
print("PART B — best_model dự đoán lại trên chính CSV (kiểm tra model lành mạnh)")
print("=" * 64)
print(f"  MAE = {mae:.4f} (kỳ vọng ~2.2 => model + định dạng CSV khớp nhau)\n")

# ============ PART C — Dựng lại từ json & so khớp CSV ============
def proc(lst, H):
    a = np.array([eval(s) for s in lst], dtype=np.float64)  # (n,3): (x,y,z)
    a[:, 1] = H - a[:, 1]                                    # lật y giống tiền xử lý
    return a[:, :2]

def feat_from_pts(order_pts, inner):
    norm = np.linalg.norm(order_pts[0] - order_pts[1])  # dist 2 điểm đầu (khóe-khóe)
    if norm < 1e-6:
        return None
    return ((np.array(order_pts) - inner) / norm).reshape(-1).astype(np.float32)

json_files = sorted(glob.glob(os.path.join(IMGS, "*.json")),
                    key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))[:150]

print("=" * 64)
print("PART C — Đảo-ngược-kỹ-thuật thứ tự điểm từ UnityEyes json")
print("=" * 64)
# Chuẩn hóa GIỐNG CSV: gốc = khóe trong (caruncle[3]), scale = dist(caruncle[3], interior_margin[8]).
# Tính vị trí chuẩn hóa TRUNG BÌNH của 16 điểm interior_margin (+ caruncle + tâm iris),
# thử cả 2 dấu trục y, rồi so với mean của point_1..7 trong CSV để biết điểm CSV = điểm json nào.
csv_mean = P.mean(axis=0)  # (7,2)

for ysign in (+1, -1):
    acc = {f"im{j}": [] for j in range(16)}
    acc["iris"] = []
    for jf in json_files:
        d = json.load(open(jf))
        im = proc(d["interior_margin_2d"], 0); im[:, 1] *= ysign
        car = proc(d["caruncle_2d"], 0); car[:, 1] *= ysign
        iris_c = proc(d["iris_2d"], 0); iris_c[:, 1] *= ysign; iris_c = iris_c.mean(axis=0)
        inner, outer = car[3], im[8]
        norm = np.linalg.norm(inner - outer)
        if norm < 1e-6:
            continue
        for j in range(16):
            acc[f"im{j}"].append((im[j] - inner) / norm)
        acc["iris"].append((iris_c - inner) / norm)
    means = {k: np.mean(v, axis=0) for k, v in acc.items() if len(v)}
    print(f"\n  --- Dấu trục y = {'+' if ysign>0 else '-'} (gốc=caruncle[3], scale=|car3-im8|) ---")
    print("  Với mỗi point CSV, tìm điểm json chuẩn hóa GẦN NHẤT:")
    for i in range(7):
        tgt = csv_mean[i]
        best_k, best_d = None, 1e9
        for k, m in means.items():
            dd = np.linalg.norm(m - tgt)
            if dd < best_d:
                best_d, best_k = dd, k
        print(f"    point_{i+1} CSV=({tgt[0]:+.3f},{tgt[1]:+.3f}) ~ {best_k:>5} "
              f"=({means[best_k][0]:+.3f},{means[best_k][1]:+.3f})  d={best_d:.3f}")

print("\n" + "=" * 64)
print("KẾT LUẬN")
print("=" * 64)
print("  - PART A: model MONG ĐỢI [khóe TRONG=(0,0), khóe NGOÀI(|mag|=1), 4 điểm mí, đồng tử].")
print("  - EyeTracking.py đang lấy shape[36:42] theo thứ tự dlib")
print("    = [khóe NGOÀI, mí, mí, khóe TRONG, mí, mí] => point_1 thành khóe NGOÀI (SAI).")
print("  - PART C cho biết point_2..6 của CSV ứng với interior_margin index nào")
print("    và dấu trục y đúng => xếp lại thứ tự điểm webcam cho khớp.")
