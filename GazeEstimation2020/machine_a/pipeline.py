import cv2
import mediapipe as mp
import numpy as np

from .ad import AdSelectionRequest, extract_ad_selection, resolve_media_path
from .config import AGE_GENDER_EVERY_N_FRAMES, AD_WINDOW_HEIGHT, AD_WINDOW_WIDTH, AUDIENCE_WINDOW_SECONDS, REPORT_AD_URL
from .reporting import finalize_attention_session, iso_now, majority_vote, post_json_async, post_json_async_many
from .tracking import ViewerTrackManager
from .vision_utils import build_face_bbox, get_mediapipe_landmarks, map_to_dlib_style, predict_age_gender, predict_pupil, segment_eyes


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


def collect_viewer_detections(frame, frame_index, gaze_state, *, age_gender_model, pupil_model, device, model_x, model_y, face_landmarker, annotate=True):
    display_frame = frame.copy()
    detections = []
    h, w, _ = frame.shape

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    results = face_landmarker.detect(mp_image)
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if not results.face_landmarks:
        return display_frame, detections

    for face_landmarks in results.face_landmarks:
        full_mp_shape = get_mediapipe_landmarks(face_landmarks, w, h)
        shape = map_to_dlib_style(full_mp_shape)
        x1, y1, x2, y2 = build_face_bbox(full_mp_shape, w, h)

        face_crop = frame[y1:y2, x1:x2]
        age_range = None
        age_range_confidence = None
        audience_segment_id = None
        gender_label = None

        if face_crop.size > 0 and frame_index % AGE_GENDER_EVERY_N_FRAMES == 0:
            try:
                (
                    gender_label,
                    _,
                    age_range,
                    age_range_confidence,
                    audience_segment_id,
                ) = predict_age_gender(age_gender_model, face_crop)
            except Exception as exc:
                print("Age/Gender prediction error:", exc)

        is_looking_at_screen = False
        try:
            eye_samples = segment_eyes(gray_frame, shape)
            pupil_predicts = predict_pupil(pupil_model, device, eye_samples)

            left_eyes = [item for item in pupil_predicts if item.eye_sample.is_left]
            right_eyes = [item for item in pupil_predicts if not item.eye_sample.is_left]

            if left_eyes and right_eyes:
                center_left = tuple(left_eyes[0].landmarks[0].astype(int))
                center_right = tuple(right_eyes[0].landmarks[0].astype(int))

                norm_right = np.linalg.norm(shape[36] - shape[39])
                norm_left = np.linalg.norm(shape[42] - shape[45])

                if norm_right > 0 and norm_left > 0:
                    look_x_r = model_x.predict(
                        ((np.vstack([shape[36:42], center_right]) - shape[36]) / norm_right).reshape(1, -1)
                    )[0]
                    look_y_r = model_y.predict(
                        np.append(
                            ((np.vstack([shape[36:42], center_right]) - shape[36]) / norm_right).reshape(1, -1).flatten(),
                            look_x_r,
                        ).reshape(1, -1)
                    )[0]

                    look_x_l = model_x.predict(
                        ((np.vstack([shape[42:48], center_left]) - shape[42]) / norm_left).reshape(1, -1)
                    )[0]
                    look_y_l = model_y.predict(
                        np.append(
                            ((np.vstack([shape[42:48], center_left]) - shape[42]) / norm_left).reshape(1, -1).flatten(),
                            look_x_l,
                        ).reshape(1, -1)
                    )[0]

                    avg_raw_x = (float(look_x_r) + float(look_x_l)) / 2.0
                    avg_raw_y = (float(look_y_r) + float(look_y_l)) / 2.0

                    gaze_state["smooth_x"] = 0.2 * avg_raw_x + 0.8 * gaze_state["smooth_x"]
                    gaze_state["smooth_y"] = 0.2 * avg_raw_y + 0.8 * gaze_state["smooth_y"]

                    screen_x_min, screen_x_max = -0.5, 0.5
                    screen_y_min, screen_y_max = -0.15, 0.8
                    is_looking_at_screen = (
                        screen_x_min < gaze_state["smooth_x"] < screen_x_max
                    ) and (screen_y_min < gaze_state["smooth_y"] < screen_y_max)

                if annotate:
                    cv2.circle(display_frame, center_left, 2, (0, 0, 255), -1)
                    cv2.circle(display_frame, center_right, 2, (0, 0, 255), -1)
        except Exception as exc:
            print(f"Gaze prediction error: {exc}")

        detections.append(
            {
                "bbox": (x1, y1, x2, y2),
                "audience_segment_id": audience_segment_id,
                "looking": is_looking_at_screen,
            }
        )

        if annotate:
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
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
            status_text = "Looking" if is_looking_at_screen else "Not looking"
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
