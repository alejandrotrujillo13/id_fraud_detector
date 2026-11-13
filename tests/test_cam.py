import importlib.util
import unittest

import numpy as np

HAS_TORCH = importlib.util.find_spec("torch") is not None and importlib.util.find_spec("torchvision") is not None


@unittest.skipUnless(HAS_TORCH, "the train group (PyTorch) is not installed")
class GradCamTest(unittest.TestCase):
    def test_maps_have_the_grid_of_the_last_block_and_stay_in_range(self):
        import torch

        from id_fraud_detector.model.cam import grad_cam
        from id_fraud_detector.model.net import build

        torch.manual_seed(0)
        model = build("mobilenet_v3_large", pretrained=False)
        scores, maps = grad_cam(model, torch.randn(2, 3, 160, 256))
        self.assertEqual(scores.shape, (2,))
        self.assertEqual(maps.shape, (2, 5, 8))  # the network reduces each side 32 times
        self.assertTrue(((scores > 0) & (scores < 1)).all())
        self.assertGreaterEqual(float(maps.min()), 0.0)
        self.assertLessEqual(float(maps.max()), 1.0 + 1e-6)

    def test_overlay_keeps_the_photo_where_there_is_no_heat(self):
        from id_fraud_detector.model.cam import overlay

        photo = np.full((40, 64, 3), 200, dtype=np.uint8)
        heat = np.zeros((5, 8), dtype=np.float32)
        heat[0, 0] = 1.0
        painted = overlay(photo, heat)
        self.assertEqual(painted.shape, photo.shape)
        self.assertTrue((painted[-1, -1] == 200).all())      # far corner: untouched grey
        self.assertGreater(int(painted[0, 0, 0]), int(painted[0, 0, 2]))  # heated corner: warm hue

    def test_border_share(self):
        from id_fraud_detector.model.cam import border_share

        centre = np.zeros((8, 8))
        centre[3:5, 3:5] = 1
        edge = np.zeros((8, 8))
        edge[0, :] = 1
        self.assertEqual(border_share(centre), 0.0)
        self.assertEqual(border_share(edge), 1.0)
        self.assertTrue(np.isnan(border_share(np.zeros((8, 8)))))


if __name__ == "__main__":
    unittest.main()
