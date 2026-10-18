import unittest

import numpy as np

from shape_tracing_game import detect_wand


class DetectWandTests(unittest.TestCase):
    def test_inference_uses_cpu(self):
        class Model:
            def __init__(self):
                self.predict_options: dict[str, object] = {}
                self.frame_shape = None

            def predict(self, frame, **kwargs):
                self.predict_options = kwargs
                self.frame_shape = frame.shape
                return []

        model = Model()

        self.assertIsNone(detect_wand(model, np.zeros((8, 8, 3), dtype=np.uint8)))
        self.assertEqual(model.frame_shape, (8, 8, 3))
        self.assertEqual(model.predict_options["device"], "cpu")


if __name__ == "__main__":
    unittest.main()
