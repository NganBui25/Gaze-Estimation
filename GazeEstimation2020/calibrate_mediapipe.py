# Hiệu chỉnh / kiểm tra ánh xạ chỉ số MediaPipe -> 7 điểm của unityeyes_processed_14d.csv.
#
# Vì MediaPipe và UnityEyes là 2 hệ landmark khác nhau, script này đo TRÊN CHÍNH KHUÔN MẶT
# của bạn: với mỗi điểm CSV (point_2..6) nó tìm chỉ số MediaPipe ở viền mắt có vị trí CHUẨN HÓA
# gần nhất, để bạn biết nên đặt MP_RIGHT/MP_LEFT["lids"] trong EyeTracking.py thành chỉ số nào.
#
# Dùng:  python calibrate_mediapipe.py           (chụp 1 ảnh từ webcam)
#        python calibrate_mediapipe.py anh.jpg   (đọc từ file ảnh có khuôn mặt)
import sys
import numpy as np
import cv2
import mediapipe as mp
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# Vị trí CHUẨN HÓA mục tiêu (canonical) lấy từ CSV — xem check_feature.py / PART A.
CSV_TARGETS = {
    "point_2 (khóe NGOÀI)":   (0.991,  0.059),
    "point_3 (mí dưới giữa)": (0.546,  0.274),
    "point_4 (mí dưới trong)": (0.166,  0.207),
    "point_5 (mí trên giữa)": (0.558, -0.174),
    "point_6 (mí trên ngoài)": (0.903, -0.058),
}
# Chỉ số đang dùng trong EyeTracking.py (để in chất lượng hiện tại)
CUR = {"RIGHT": [144, 154, 160, 161], "LEFT": [373, 381, 387, 388]}

# Toàn bộ viền mắt MediaPipe (corner + mí trên + mí dưới) để dò chỉ số tốt nhất.
RING = {
    "RIGHT": {"inner": 133, "outer": 33,
              "cands": [33, 7, 163, 144, 145, 153, 154, 155, 133,
                        173, 157, 158, 159, 160, 161, 246]},
    "LEFT":  {"inner": 362, "outer": 263,
              "cands": [362, 382, 381, 380, 374, 373, 390, 249, 263,
                        466, 388, 387, 386, 385, 384, 398]},
}


def get_landmarks(img):
    h, w = img.shape[:2]
    fm = mp.solutions.face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True,
                                         min_detection_confidence=0.5)
    res = fm.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    if not res.multi_face_landmarks:
        return None
    lm = res.multi_face_landmarks[0].landmark
    return np.array([[p.x * w, p.y * h] for p in lm[:468]], dtype=np.float32)


def normalize(mp_shape, spec):
    """Trả về hàm: idx -> vị trí chuẩn hóa (gốc=khóe trong, scale=|trong-ngoài|, lật về canonical)."""
    inner = mp_shape[spec["inner"]].astype(np.float32)
    outer = mp_shape[spec["outer"]].astype(np.float32)
    norm = np.linalg.norm(inner - outer) + 1e-9
    mirrored = (outer[0] - inner[0]) < 0   # khóe ngoài ở -x => lật

    def f(idx):
        p = (mp_shape[idx].astype(np.float32) - inner) / norm
        if mirrored:
            p = p.copy(); p[0] *= -1
        return p
    return f, mirrored


def main():
    if len(sys.argv) > 1:
        img = cv2.imread(sys.argv[1])
        if img is None:
            print("Không đọc được ảnh:", sys.argv[1]); return
    else:
        cap = cv2.VideoCapture(0)
        ok, img = cap.read(); cap.release()
        if not ok:
            print("Không mở được webcam. Hãy truyền đường dẫn ảnh: python calibrate_mediapipe.py anh.jpg"); return

    mp_shape = get_landmarks(img)
    if mp_shape is None:
        print("Không phát hiện khuôn mặt trong ảnh."); return

    for eye in ("RIGHT", "LEFT"):
        spec = RING[eye]
        f, mirrored = normalize(mp_shape, spec)
        print("=" * 60)
        print(f"MẮT {eye}  (mirrored={mirrored}; khóe trong={spec['inner']}, ngoài={spec['outer']})")
        print("=" * 60)
        # Vị trí các chỉ số đang dùng
        print("  Chỉ số đang dùng (EyeTracking.py):")
        for i, idx in enumerate(CUR[eye]):
            p = f(idx)
            print(f"    point_{i+3} <- MP {idx}: ({p[0]:+.3f},{p[1]:+.3f})")
        # Dò chỉ số gần nhất cho từng target
        print("  Đề xuất (MP gần nhất với từng point CSV):")
        for name, tgt in CSV_TARGETS.items():
            best_idx, best_d, best_p = None, 1e9, None
            for idx in spec["cands"]:
                p = f(idx)
                d = float(np.linalg.norm(p - np.array(tgt)))
                if d < best_d:
                    best_d, best_idx, best_p = d, idx, p
            print(f"    {name:<22} target=({tgt[0]:+.3f},{tgt[1]:+.3f}) "
                  f"~ MP {best_idx:>3} ({best_p[0]:+.3f},{best_p[1]:+.3f})  d={best_d:.3f}")
        print()
    print("=> Nếu chỉ số đề xuất khác chỉ số đang dùng và d nhỏ hơn, hãy cập nhật")
    print("   MP_RIGHT/MP_LEFT['lids'] trong EyeTracking.py theo thứ tự [point_3, point_4, point_5, point_6].")


if __name__ == "__main__":
    main()
