import unittest

import numpy as np

from machine_a.tracking import GazeStateManager, ViewerTrack
from machine_a.vision_utils import build_gaze_feature_14d, predict_gaze_degrees


class StubGazeModel:
    def predict(self, features):
        return np.asarray([[12.0, -3.0]], dtype=np.float32)


class GazeFeatureTests(unittest.TestCase):
    def test_feature_uses_inner_corner_as_origin(self):
        points = np.asarray(
            [[10, 10], [20, 10], [17, 12], [12, 12], [16, 8], [19, 9], [15, 10]],
            dtype=np.float32,
        )

        feature, mirrored = build_gaze_feature_14d(points)

        self.assertFalse(mirrored)
        np.testing.assert_allclose(feature.reshape(7, 2)[0], [0.0, 0.0])
        np.testing.assert_allclose(feature.reshape(7, 2)[1], [1.0, 0.0])

    def test_mirrored_eye_flips_feature_and_yaw(self):
        points = np.asarray(
            [[20, 10], [10, 10], [13, 12], [18, 12], [14, 8], [11, 9], [15, 10]],
            dtype=np.float32,
        )

        feature, mirrored = build_gaze_feature_14d(points)
        yaw, pitch = predict_gaze_degrees(StubGazeModel(), points)

        self.assertTrue(mirrored)
        np.testing.assert_allclose(feature.reshape(7, 2)[1], [1.0, 0.0])
        self.assertEqual(yaw, -12.0)
        self.assertEqual(pitch, -3.0)


class GazeStateManagerTests(unittest.TestCase):
    def test_smoothing_is_separate_for_different_faces(self):
        manager = GazeStateManager(iou_threshold=0.3, ttl_seconds=2.0, alpha=0.5)

        first = manager.update((0, 0, 100, 100), 10.0, 2.0, 1.0)
        second = manager.update((200, 0, 300, 100), -20.0, -4.0, 1.0)
        first_updated = manager.update((2, 0, 102, 100), 20.0, 4.0, 1.1)

        self.assertEqual(first, (10.0, 2.0))
        self.assertEqual(second, (-20.0, -4.0))
        self.assertEqual(first_updated, (15.0, 3.0))


class ViewerTrackTests(unittest.TestCase):
    def test_unknown_gaze_does_not_extend_watch_duration(self):
        track = ViewerTrack(1, (0, 0, 10, 10), 1.0, 1.0)
        track.update((0, 0, 10, 10), None, True, 1.0)
        track.update((0, 0, 10, 10), None, True, 2.0)
        track.update((0, 0, 10, 10), None, None, 3.0)

        summary = track.summary(session_end_ts=4.0)

        self.assertEqual(summary["watch_duration"], 1.0)


if __name__ == "__main__":
    unittest.main()
