import time

import cv2  # type: ignore

from machine_a.config import (
    AD_MISSING_FRAME_TIMEOUT,
    AD_WINDOW_FULLSCREEN,
    AD_WINDOW_HEIGHT,
    AD_WINDOW_NAME,
    AD_WINDOW_WIDTH,
    AUDIENCE_WINDOW_SECONDS,
    DEMOGRAPHIC_CACHE_TTL_SECONDS,
    DEMOGRAPHIC_REFRESH_SECONDS,
    FRAME_HEIGHT,
    FRAME_WIDTH,
    PERFORMANCE_MODE,
    NO_VIEWER_SIGNAL_DELAY_SECONDS,
    PROCESS_EVERY_N_FRAMES,
    SELECT_AD_URL,
    SERVER_IP,
    SERVER_PORT,
    TRACKING_WINDOW_HEIGHT,
    TRACKING_WINDOW_NAME,
    TRACKING_WINDOW_WIDTH,
    VIDEO_SOURCE,
)
from machine_a.models import load_models
from machine_a.demographics import DemographicPredictor
from machine_a.ad import AdSelectionRequest, extract_ad_selection, resolve_media_path
from machine_a.pipeline import (
    annotate_detections,
    build_report_payload,
    build_selection_payload,
    collect_viewer_detections,
    finalize_ad_session,
)
from machine_a.sensor import SensorMonitor
from machine_a.tracking import GazeStateManager, ViewerTrackManager
from machine_a.video import LatestFrameGrabber, create_black_frame, resize_canvas
from machine_a.ui import setup_windows


def main():
    sensor_monitor = SensorMonitor().start()
    grabber = LatestFrameGrabber(VIDEO_SOURCE).start()
    setup_windows()
    device, gaze_model, pupil_model, age_gender_model, face_landmarker = load_models()
    demographic_predictor = DemographicPredictor(
        age_gender_model,
        refresh_seconds=DEMOGRAPHIC_REFRESH_SECONDS,
        cache_ttl_seconds=DEMOGRAPHIC_CACHE_TTL_SECONDS,
    ).start()
    print(f"Performance mode: {PERFORMANCE_MODE} | process every {PROCESS_EVERY_N_FRAMES} frame(s)")

    deadline = time.time() + 10.0
    while time.time() < deadline:
        ret, _ = grabber.read()
        if ret:
            break
        time.sleep(0.2)
    else:
        sensor_monitor.stop()
        grabber.release()
        demographic_predictor.stop()
        raise RuntimeError(
            f"Không nhận được frame từ {VIDEO_SOURCE} trong 10 giây. "
            "Hãy kiểm tra IP Raspberry Pi, port UDP, hoặc biến môi trường VIDEO_SOURCE."
        )

    print("Nhấn Q để thoát...")

    frame_index = 0
    gaze_state = GazeStateManager()
    selection_manager = ViewerTrackManager()
    ad_manager = ViewerTrackManager()

    state = "idle"
    selection_start_ts = None
    selection_request = None
    selection_generation_id = 0
    selected_ad = None
    selected_ad_path = None
    ad_capture = None
    ad_start_ts = None
    last_ad_frame_ts = None
    last_ad_display_frame = None
    last_tracking_display_frame = create_black_frame(TRACKING_WINDOW_WIDTH, TRACKING_WINDOW_HEIGHT)
    last_tracking_detections = []
    no_viewer_start_ts = None
    no_viewer_cooldown_start_ts = None
    no_viewer_signal_sent_for_cycle = False
    no_viewer_signal_request = None

    try:
        while True:
            sensor_state = sensor_monitor.get_state()
            ret, frame = grabber.read()
            now_ts = time.time()
            should_process_frame = frame_index % PROCESS_EVERY_N_FRAMES == 0
            tracking_display_frame = last_tracking_display_frame.copy()
            ad_display_frame = create_black_frame(AD_WINDOW_WIDTH, AD_WINDOW_HEIGHT)

            if state == "ad":
                if ret:
                    if should_process_frame:
                        tracking_display_frame, detections = collect_viewer_detections(
                            frame,
                            frame_index,
                            gaze_state,
                            demographic_predictor=demographic_predictor,
                            pupil_model=pupil_model,
                            device=device,
                            gaze_model=gaze_model,
                            face_landmarker=face_landmarker,
                            annotate=True,
                        )
                        ad_manager.update(detections, now_ts)
                        last_tracking_detections = detections
                        last_tracking_display_frame = tracking_display_frame.copy()
                    else:
                        tracking_display_frame = annotate_detections(
                            frame.copy(),
                            last_tracking_detections,
                        )
                else:
                    tracking_display_frame = create_black_frame(TRACKING_WINDOW_WIDTH, TRACKING_WINDOW_HEIGHT)

                if ad_capture is not None and (last_ad_frame_ts is None or now_ts - last_ad_frame_ts >= 0.03):
                    ok, ad_frame = ad_capture.read()
                    if ok:
                        last_ad_display_frame = resize_canvas(ad_frame, AD_WINDOW_WIDTH, AD_WINDOW_HEIGHT)
                        ad_display_frame = last_ad_display_frame.copy()
                        last_ad_frame_ts = now_ts
                    else:
                        if selected_ad_path is not None:
                            ad_capture.release()
                            ad_capture = cv2.VideoCapture(selected_ad_path)
                            ok, ad_frame = ad_capture.read()
                            if ok:
                                last_ad_display_frame = resize_canvas(ad_frame, AD_WINDOW_WIDTH, AD_WINDOW_HEIGHT)
                                ad_display_frame = last_ad_display_frame.copy()
                                last_ad_frame_ts = now_ts

                if last_ad_display_frame is not None:
                    ad_display_frame = last_ad_display_frame.copy()

                if selected_ad is not None and ad_start_ts is not None:
                    elapsed = now_ts - ad_start_ts
                    if elapsed >= selected_ad["duration_seconds"]:
                        finalize_ad_session(selected_ad, ad_manager, ad_start_ts, now_ts)
                        state = "idle"
                        selection_start_ts = None
                        selected_ad = None
                        selected_ad_path = None
                        ad_start_ts = None
                        last_ad_frame_ts = None
                        last_ad_display_frame = None
                        ad_manager.reset()
                        if ad_capture is not None:
                            ad_capture.release()
                            ad_capture = None
                        no_viewer_cooldown_start_ts = now_ts
                        no_viewer_signal_sent_for_cycle = False
                        no_viewer_start_ts = None

                cv2.imshow(TRACKING_WINDOW_NAME, tracking_display_frame)
                cv2.imshow(AD_WINDOW_NAME, ad_display_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                frame_index += 1
                continue

            if sensor_state == "Dark":
                tracking_display_frame = create_black_frame(TRACKING_WINDOW_WIDTH, TRACKING_WINDOW_HEIGHT)
                ad_display_frame = create_black_frame(AD_WINDOW_WIDTH, AD_WINDOW_HEIGHT)
                cv2.putText(
                    tracking_display_frame,
                    "Display off - sensor dark",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.85,
                    (255, 255, 255),
                    2,
                )
                last_tracking_display_frame = tracking_display_frame.copy()
                cv2.imshow(TRACKING_WINDOW_NAME, tracking_display_frame)
                cv2.imshow(AD_WINDOW_NAME, ad_display_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                frame_index += 1
                continue

            if not ret:
                tracking_display_frame = create_black_frame(TRACKING_WINDOW_WIDTH, TRACKING_WINDOW_HEIGHT)
                ad_display_frame = create_black_frame(AD_WINDOW_WIDTH, AD_WINDOW_HEIGHT)
                cv2.imshow(TRACKING_WINDOW_NAME, tracking_display_frame)
                cv2.imshow(AD_WINDOW_NAME, ad_display_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                frame_index += 1
                time.sleep(0.02)
                continue

            if not should_process_frame:
                tracking_display_frame = annotate_detections(
                    frame.copy(),
                    last_tracking_detections,
                )
                cv2.putText(
                    tracking_display_frame,
                    f"Skipping frame ({PERFORMANCE_MODE})",
                    (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 255, 255),
                    2,
                )
                last_tracking_display_frame = tracking_display_frame.copy()
                cv2.imshow(TRACKING_WINDOW_NAME, tracking_display_frame)
                cv2.imshow(AD_WINDOW_NAME, ad_display_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                frame_index += 1
                continue

            tracking_display_frame, detections = collect_viewer_detections(
                frame,
                frame_index,
                gaze_state,
                demographic_predictor=demographic_predictor,
                pupil_model=pupil_model,
                device=device,
                gaze_model=gaze_model,
                face_landmarker=face_landmarker,
                annotate=True,
            )
            has_viewers = len(detections) > 0
            last_tracking_detections = detections
            last_tracking_display_frame = tracking_display_frame.copy()

            if selection_start_ts is None:
                selection_start_ts = now_ts
                selection_manager.reset()
                no_viewer_cooldown_start_ts = None
                no_viewer_signal_sent_for_cycle = False

            if has_viewers:
                selection_manager.update(detections, now_ts)

            if selection_request is None:
                elapsed = now_ts - selection_start_ts
                if elapsed >= AUDIENCE_WINDOW_SECONDS:
                    selection_payload = build_selection_payload(
                        selection_manager.tracks,
                        selection_start_ts,
                        now_ts,
                    )
                    selection_request = AdSelectionRequest(
                        SELECT_AD_URL,
                        selection_payload,
                        selection_generation_id,
                    ).start()
                    status_text = (
                        "Selecting next ad..."
                        if selection_payload["viewer_count"] > 0
                        else "No viewers - selecting fallback ad..."
                    )
                    cv2.putText(
                        tracking_display_frame,
                        status_text,
                        (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 255),
                        2,
                    )

            if no_viewer_signal_request is not None and no_viewer_signal_request.event.is_set():
                if no_viewer_signal_request.error is not None:
                    print(f"No viewers signal failed: {no_viewer_signal_request.error}")
                else:
                    print(f"No viewers signal sent: {no_viewer_signal_request.payload}")
                    try:
                        selected_ad = extract_ad_selection(no_viewer_signal_request.response)
                        selected_ad_path = resolve_media_path(selected_ad["media_filename"])
                        ad_capture = cv2.VideoCapture(selected_ad_path)
                        if not ad_capture.isOpened():
                            raise RuntimeError(f"Cannot open ad media: {selected_ad_path}")
                        ad_manager.reset()
                        state = "ad"
                        ad_start_ts = now_ts
                        last_ad_frame_ts = None
                        last_ad_display_frame = None
                        selection_manager.reset()
                        print(f"No viewers response: {selected_ad}")
                    except Exception as exc:
                        print(f"No viewers response invalid: {exc}")
                no_viewer_signal_request = None
                no_viewer_cooldown_start_ts = None

            if selection_request is not None and selection_request.event.is_set():
                if selection_request.generation_id != selection_generation_id:
                    selection_request = None
                    selection_start_ts = None
                    selection_manager.reset()
                elif selection_request.error is not None:
                    print(f"Ad selection failed: {selection_request.error}")
                    selection_request = None
                    selection_start_ts = now_ts
                    selection_manager.reset()
                else:
                    try:
                        selected_ad = extract_ad_selection(selection_request.response)
                        selected_ad_path = resolve_media_path(selected_ad["media_filename"])
                        ad_capture = cv2.VideoCapture(selected_ad_path)
                        if not ad_capture.isOpened():
                            raise RuntimeError(f"Cannot open ad media: {selected_ad_path}")
                        ad_manager.reset()
                        state = "ad"
                        ad_start_ts = now_ts
                        last_ad_frame_ts = None
                        last_ad_display_frame = None
                        selection_manager.reset()
                        print(f"Selected ad: {selected_ad}")
                    except Exception as exc:
                        print(f"Invalid ad response: {exc}")
                        selected_ad = None
                        selected_ad_path = None
                        last_ad_display_frame = None
                        selection_request = None
                        selection_start_ts = now_ts
                        selection_manager.reset()
                selection_request = None

            remaining = 0.0 if selection_start_ts is None else max(0.0, AUDIENCE_WINDOW_SECONDS - (now_ts - selection_start_ts))
            cv2.putText(
                tracking_display_frame,
                f"Sensor: {sensor_state} | Sampling: {remaining:.1f}s",
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
            )

            cv2.imshow(TRACKING_WINDOW_NAME, tracking_display_frame)
            cv2.imshow(AD_WINDOW_NAME, ad_display_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            frame_index += 1

    finally:
        if selected_ad is not None and ad_start_ts is not None:
            finalize_ad_session(selected_ad, ad_manager, ad_start_ts, time.time())
        if ad_capture is not None:
            ad_capture.release()
        grabber.release()
        demographic_predictor.stop()
        sensor_monitor.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
