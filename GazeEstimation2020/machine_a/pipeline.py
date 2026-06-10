import cv2
import mediapipe as mp
import numpy as np
import time

from .ad import AdSelectionRequest, extract_ad_selection, resolve_media_path
from .config import (
    AGE_GENDER_EVERY_N_FRAMES,
    AD_WINDOW_HEIGHT,
    AD_WINDOW_WIDTH,
    AUDIENCE_WINDOW_SECONDS,
    GAZE_MAX_ABS_PITCH_DEG,
    GAZE_MAX_ABS_YAW_DEG,
    GAZE_MAX_PITCH_DISAGREEMENT_DEG,
    GAZE_MIN_EYE_WIDTH_PX,
    GAZE_PITCH_MAX,
    GAZE_PITCH_MIN,
    GAZE_YAW_MAX,
    GAZE_YAW_MIN,
    REPORT_AD_URL,
    VISION_PROCESS_WIDTH,
)
from .reporting import finalize_attention_session, iso_now, majority_vote, post_json_async, post_json_async_many
from .tracking import ViewerTrackManager
from .vision_utils import (
    MP_LEFT_EYE,
    MP_RIGHT_EYE,
    build_face_bbox,
    eye_points_7,
    get_mediapipe_landmarks,
    map_to_dlib_style,
    predict_gaze_degrees,
    predict_pupil,
    segment_eyes,
)


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
        if is_looking_at_screen is None:
            status_text = "Gaze unavailable"
            status_color = (0, 165, 255)
        else:
            status_text = "Looking" if is_looking_at_screen else "Not looking"
            status_color = (0, 255, 0) if is_looking_at_screen else (0, 0, 255)
        yaw = detection.get("yaw")
        pitch = detection.get("pitch")
        if yaw is not None and pitch is not None:
            status_text += f" | yaw={yaw:.1f} pitch={pitch:.1f}"
        status_y = y2 + 20 if y2 + 42 < h else max(20, y1 + 20)
        cv2.putText(
            display_frame,
            status_text,
            (x1, status_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            status_color,
            1,
        )
        boundary_text = (
            f"Left/Right={GAZE_YAW_MIN:.0f}/{GAZE_YAW_MAX:.0f} "
            f"Up/Down={GAZE_PITCH_MIN:.0f}/{GAZE_PITCH_MAX:.0f} deg"
        )
        cv2.putText(
            display_frame,
            boundary_text,
            (x1, min(h - 10, status_y + 18)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (255, 255, 0),
            1,
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

    process_h, process_w, _ = process_frame.shape
    scale_x = source_w / process_w
    scale_y = source_h / process_h

    rgb_frame = cv2.cvtColor(process_frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    results = face_landmarker.detect(mp_image)
    gray_frame = cv2.cvtColor(process_frame, cv2.COLOR_BGR2GRAY)

    if not results.face_landmarks:
        return display_frame, detections

    for face_landmarks in results.face_landmarks:
        full_mp_shape = get_mediapipe_landmarks(face_landmarks, process_w, process_h)
        shape = map_to_dlib_style(full_mp_shape)
        process_x1, process_y1, process_x2, process_y2 = build_face_bbox(
            full_mp_shape,
            process_w,
            process_h,
        )
        x1 = max(0, int(process_x1 * scale_x))
        y1 = max(0, int(process_y1 * scale_y))
        x2 = min(source_w, int(process_x2 * scale_x))
        y2 = min(source_h, int(process_y2 * scale_y))

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

        is_looking_at_screen = None
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

                norm_right = float(
                    np.linalg.norm(
                        full_mp_shape[MP_RIGHT_EYE["inner"]]
                        - full_mp_shape[MP_RIGHT_EYE["outer"]]
                    )
                )
                norm_left = float(
                    np.linalg.norm(
                        full_mp_shape[MP_LEFT_EYE["inner"]]
                        - full_mp_shape[MP_LEFT_EYE["outer"]]
                    )
                )

                if norm_right >= GAZE_MIN_EYE_WIDTH_PX and norm_left >= GAZE_MIN_EYE_WIDTH_PX:
                    points_right = eye_points_7(full_mp_shape, MP_RIGHT_EYE, center_right)
                    points_left = eye_points_7(full_mp_shape, MP_LEFT_EYE, center_left)
                    yaw_right, pitch_right = predict_gaze_degrees(gaze_model, points_right)
                    yaw_left, pitch_left = predict_gaze_degrees(gaze_model, points_left)

                    # Horizontal eye angles can legitimately have opposite signs
                    # because both eyes converge toward the same nearby target.
                    pitch_disagreement = abs(pitch_right - pitch_left)
                    avg_yaw = (yaw_right + yaw_left) / 2.0
                    avg_pitch = (pitch_right + pitch_left) / 2.0

                    gaze_is_plausible = (
                        pitch_disagreement <= GAZE_MAX_PITCH_DISAGREEMENT_DEG
                        and abs(avg_yaw) <= GAZE_MAX_ABS_YAW_DEG
                        and abs(avg_pitch) <= GAZE_MAX_ABS_PITCH_DEG
                    )
                    if gaze_is_plausible:
                        smooth_yaw, smooth_pitch = gaze_state.update(
                            (x1, y1, x2, y2),
                            avg_yaw,
                            avg_pitch,
                            now_ts,
                        )
                        is_looking_at_screen = (
                            GAZE_YAW_MIN < smooth_yaw < GAZE_YAW_MAX
                            and GAZE_PITCH_MIN < smooth_pitch < GAZE_PITCH_MAX
                        )

                if annotate:
                    display_center_left = (
                        int(center_left[0] * scale_x),
                        int(center_left[1] * scale_y),
                    )
                    display_center_right = (
                        int(center_right[0] * scale_x),
                        int(center_right[1] * scale_y),
                    )
                    cv2.circle(display_frame, display_center_left, 2, (0, 0, 255), -1)
                    cv2.circle(display_frame, display_center_right, 2, (0, 0, 255), -1)
        except Exception as exc:
            print(f"Gaze prediction error: {exc}")

        detections.append(
            {
                "bbox": (x1, y1, x2, y2),
                "audience_segment_id": audience_segment_id,
                "looking": is_looking_at_screen,
                "yaw": smooth_yaw,
                "pitch": smooth_pitch,
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
