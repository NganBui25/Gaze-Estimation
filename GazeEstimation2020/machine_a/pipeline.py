import cv2
import math
import mediapipe as mp
import numpy as np
import time

from .ad import AdSelectionRequest, extract_ad_selection, resolve_media_path
from .config import (
    AGE_GENDER_EVERY_N_FRAMES,
    AD_WINDOW_HEIGHT,
    AD_WINDOW_WIDTH,
    AUDIENCE_WINDOW_SECONDS,
    CAMERA_HFOV_DEG,
    GAZE_BEARING_CORRECTION,
    GAZE_HEAD_WEIGHT,
    GAZE_PITCH_MAX,
    GAZE_PITCH_MIN,
    GAZE_SMOOTH_ALPHA,
    GAZE_STATE_TTL_SECONDS,
    GAZE_YAW_MAX,
    GAZE_YAW_MIN,
    REPORT_AD_URL,
    VISION_PROCESS_WIDTH,
)
from .reporting import finalize_attention_session, iso_now, majority_vote, post_json_async, post_json_async_many
from .tracking import ViewerTrackManager
from .vision_utils import (
    MP_LEFT,
    MP_RIGHT,
    build_face_bbox,
    eye_points_7,
    face_bearing_deg,
    get_mediapipe_landmarks,
    head_angles_from_matrix,
    map_to_dlib_style,
    predict_gaze_deg,
    predict_pupil,
    segment_eyes,
)


def smooth_gaze_for_face(gaze_state, face_center, yaw, pitch, now_ts, match_dist):
    """EMA làm mượt (yaw, pitch) THEO TỪNG khuôn mặt — ghép mặt giữa các frame bằng
    tâm bbox gần nhất (trong bán kính match_dist). Trạng thái cũ quá TTL bị xóa.
    Trước đây cả đám đông dùng chung một cặp smooth_x/smooth_y nên góc nhìn của
    nhiều người bị trộn lẫn vào nhau."""
    faces = gaze_state.setdefault("faces", [])
    faces[:] = [item for item in faces if now_ts - item["ts"] <= GAZE_STATE_TTL_SECONDS]

    best = None
    best_dist = None
    for item in faces:
        if item["ts"] >= now_ts:  # đã được một mặt khác trong frame này nhận
            continue
        dist = math.hypot(item["center"][0] - face_center[0], item["center"][1] - face_center[1])
        if best_dist is None or dist < best_dist:
            best, best_dist = item, dist

    if best is not None and best_dist <= match_dist:
        best["yaw"] = GAZE_SMOOTH_ALPHA * yaw + (1 - GAZE_SMOOTH_ALPHA) * best["yaw"]
        best["pitch"] = GAZE_SMOOTH_ALPHA * pitch + (1 - GAZE_SMOOTH_ALPHA) * best["pitch"]
        best["center"] = face_center
        best["ts"] = now_ts
        return best["yaw"], best["pitch"]

    faces.append({"center": face_center, "yaw": yaw, "pitch": pitch, "ts": now_ts})
    return yaw, pitch


def build_selection_payload(tracks, window_start_ts, window_end_ts):
    viewer_summaries = [track.summary(session_end_ts=window_end_ts) for track in tracks if track.frames_seen > 0]
    classified_viewers = [
        viewer
        for viewer in viewer_summaries
        if viewer["audience_segment_id"] is not None
    ]
    viewer_count = len(classified_viewers)
    audience_segment_id = majority_vote(
        [viewer["audience_segment_id"] for viewer in classified_viewers],
        default=None,
    )

    return {
        "viewer_count": viewer_count,
        "timestamp": iso_now(window_end_ts),
        "audience_segment_id": audience_segment_id,
        "window_start": iso_now(window_start_ts),
        "window_end": iso_now(window_end_ts),
    }


def build_report_payload(ad_id, tracks, start_ts, end_ts):
    viewers = [track.summary(session_end_ts=end_ts) for track in tracks if track.frames_seen > 0]
    return {
        "ad_id": ad_id,
        "start_time": iso_now(start_ts),
        "end_time": iso_now(end_ts),
        "total_viewers": len(viewers),
        "viewers": viewers,
    }


def annotate_detections(display_frame, detections):
    h, _, _ = display_frame.shape
    for detection in detections:
        x1, y1, x2, y2 = detection["bbox"]
        cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 255), 2)

        gender_label = detection.get("gender_label")
        age_range = detection.get("age_range")
        age_range_confidence = detection.get("age_range_confidence")
        if gender_label is not None and age_range is not None:
            cv2.putText(
                display_frame,
                f"{gender_label} | Age range: {age_range} ({age_range_confidence:.2f})",
                (x1, max(20, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 255),
                2,
            )

        is_looking_at_screen = detection.get("looking", False)
        status_text = "Looking" if is_looking_at_screen else "Not looking"
        gaze_yaw = detection.get("gaze_yaw")
        gaze_pitch = detection.get("gaze_pitch")
        if gaze_yaw is not None and gaze_pitch is not None:
            status_text += f" (y={gaze_yaw:.0f} p={gaze_pitch:.0f})"
        status_color = (0, 255, 0) if is_looking_at_screen else (0, 0, 255)
        cv2.putText(
            display_frame,
            status_text,
            (x1, min(h - 10, y2 + 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            status_color,
            2,
        )
    return display_frame


def collect_viewer_detections(frame, frame_index, gaze_state, *, demographic_predictor, pupil_model, device, gaze_model, face_landmarker, annotate=True):
    display_frame = frame.copy()
    detections = []
    source_h, source_w, _ = frame.shape
    now_ts = time.time()

    if source_w > VISION_PROCESS_WIDTH:
        process_scale = VISION_PROCESS_WIDTH / source_w
        process_h = max(1, int(source_h * process_scale))
        process_frame = cv2.resize(
            frame,
            (VISION_PROCESS_WIDTH, process_h),
            interpolation=cv2.INTER_AREA,
        )
    else:
        process_frame = frame

    rgb_frame = cv2.cvtColor(process_frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    results = face_landmarker.detect(mp_image)

    if not results.face_landmarks:
        return display_frame, detections

    # Cắt mắt từ frame GỐC (không downscale): với mặt nhỏ/đứng xa, crop mắt từ frame
    # đã thu nhỏ chỉ còn vài pixel nên PupilNet đoán tâm đồng tử rất kém.
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Head pose từng khuôn mặt (cùng thứ tự với face_landmarks).
    head_matrixes = getattr(results, "facial_transformation_matrixes", None) or []

    for face_index, face_landmarks in enumerate(results.face_landmarks):
        # Landmark là tọa độ chuẩn hóa [0,1] -> nhân thẳng với kích thước frame gốc
        # (detect vẫn chạy trên frame thu nhỏ để giữ tốc độ).
        full_mp_shape = get_mediapipe_landmarks(face_landmarks, source_w, source_h)
        shape = map_to_dlib_style(full_mp_shape)
        x1, y1, x2, y2 = build_face_bbox(full_mp_shape, source_w, source_h)

        face_crop = frame[y1:y2, x1:x2]
        age_range = None
        age_range_confidence = None
        audience_segment_id = None
        gender_label = None

        prediction = demographic_predictor.get_prediction((x1, y1, x2, y2), now_ts)
        if prediction is not None:
            (
                gender_label,
                _,
                age_range,
                age_range_confidence,
                audience_segment_id,
            ) = prediction

        if face_crop.size > 0 and frame_index % AGE_GENDER_EVERY_N_FRAMES == 0:
            demographic_predictor.submit((x1, y1, x2, y2), face_crop, now_ts)

        is_looking_at_screen = False
        smooth_yaw = None
        smooth_pitch = None
        try:
            eye_samples = segment_eyes(gray_frame, shape)
            pupil_predicts = predict_pupil(pupil_model, device, eye_samples)

            left_eyes = [item for item in pupil_predicts if item.eye_sample.is_left]
            right_eyes = [item for item in pupil_predicts if not item.eye_sample.is_left]

            if left_eyes and right_eyes:
                center_left = tuple(left_eyes[0].landmarks[0].astype(int))
                center_right = tuple(right_eyes[0].landmarks[0].astype(int))

                # Dự đoán ĐỒNG THỜI (yaw, pitch) theo độ bằng best_model:
                # 7 điểm/mắt lấy trực tiếp từ mediapipe landmarks (full_mp_shape)
                # theo ánh xạ MP_RIGHT/MP_LEFT, cộng tâm đồng tử từ PupilNet.
                pts7_r = eye_points_7(full_mp_shape, MP_RIGHT, center_right)
                pts7_l = eye_points_7(full_mp_shape, MP_LEFT, center_left)
                yaw_r, pitch_r = predict_gaze_deg(gaze_model, pts7_r)
                yaw_l, pitch_l = predict_gaze_deg(gaze_model, pts7_l)
                eye_yaw = (yaw_r + yaw_l) / 2.0
                eye_pitch = (pitch_r + pitch_l) / 2.0

                # Bù góc quay đầu: best_model cho góc mắt SO VỚI ĐẦU (train trên UnityEyes),
                # người quay đầu về màn hình nhưng mắt thẳng vẫn phải tính là đang nhìn.
                head_yaw, head_pitch = 0.0, 0.0
                if face_index < len(head_matrixes):
                    head_yaw, head_pitch = head_angles_from_matrix(head_matrixes[face_index])
                total_yaw = eye_yaw + GAZE_HEAD_WEIGHT * head_yaw
                total_pitch = eye_pitch + GAZE_HEAD_WEIGHT * head_pitch

                # Bù góc phương vị: người đứng lệch biên khung hình nhìn về màn hình
                # (đặt cạnh camera) có hướng nhìn = -phương_vị, cộng lại để quy về ~0.
                face_center = (0.5 * (x1 + x2), 0.5 * (y1 + y2))
                if GAZE_BEARING_CORRECTION:
                    bearing_yaw, bearing_pitch = face_bearing_deg(
                        face_center, source_w, source_h, CAMERA_HFOV_DEG
                    )
                    total_yaw += bearing_yaw
                    total_pitch += bearing_pitch

                # Làm mượt EMA theo TỪNG khuôn mặt (ghép theo tâm bbox gần nhất)
                match_dist = max(20.0, 0.5 * (x2 - x1))
                smooth_yaw, smooth_pitch = smooth_gaze_for_face(
                    gaze_state, face_center, total_yaw, total_pitch, now_ts, match_dist
                )

                is_looking_at_screen = (
                    GAZE_YAW_MIN < smooth_yaw < GAZE_YAW_MAX
                ) and (GAZE_PITCH_MIN < smooth_pitch < GAZE_PITCH_MAX)

                if annotate:
                    cv2.circle(display_frame, center_left, 2, (0, 0, 255), -1)
                    cv2.circle(display_frame, center_right, 2, (0, 0, 255), -1)

                    # Vẽ vector hướng nhìn (xanh lá): góc mắt + góc đầu (không gồm
                    # bù phương vị — đó là hiệu chỉnh hình học, không phải hướng nhìn).
                    GAZE_LEN = 2.5
                    draw_yaw_r = yaw_r + GAZE_HEAD_WEIGHT * head_yaw
                    draw_pitch_r = pitch_r + GAZE_HEAD_WEIGHT * head_pitch
                    draw_yaw_l = yaw_l + GAZE_HEAD_WEIGHT * head_yaw
                    draw_pitch_l = pitch_l + GAZE_HEAD_WEIGHT * head_pitch
                    norm_r = np.linalg.norm(full_mp_shape[MP_RIGHT["inner"]] - full_mp_shape[MP_RIGHT["outer"]])
                    norm_l = np.linalg.norm(full_mp_shape[MP_LEFT["inner"]] - full_mp_shape[MP_LEFT["outer"]])
                    end_r = (
                        int(center_right[0] + math.sin(math.radians(draw_yaw_r)) * norm_r * GAZE_LEN),
                        int(center_right[1] + math.sin(math.radians(draw_pitch_r)) * norm_r * GAZE_LEN),
                    )
                    end_l = (
                        int(center_left[0] + math.sin(math.radians(draw_yaw_l)) * norm_l * GAZE_LEN),
                        int(center_left[1] + math.sin(math.radians(draw_pitch_l)) * norm_l * GAZE_LEN),
                    )
                    cv2.line(display_frame, center_right, end_r, (0, 255, 0), 2)
                    cv2.line(display_frame, center_left, end_l, (0, 255, 0), 2)
        except Exception as exc:
            print(f"Gaze prediction error: {exc}")

        detections.append(
            {
                "bbox": (x1, y1, x2, y2),
                "audience_segment_id": audience_segment_id,
                "looking": is_looking_at_screen,
                "gaze_yaw": smooth_yaw,
                "gaze_pitch": smooth_pitch,
                "gender_label": gender_label,
                "age_range": age_range,
                "age_range_confidence": age_range_confidence,
            }
        )

    if annotate:
        annotate_detections(display_frame, detections)

    return display_frame, detections


def finalize_ad_session(ad_info, ad_manager, ad_start_ts, ad_end_ts):
    report_payload = build_report_payload(
        ad_info["ad_id"],
        ad_manager.tracks,
        ad_start_ts,
        ad_end_ts,
    )
    post_json_async(REPORT_AD_URL, report_payload)
    print("Ad report queued:", report_payload)
