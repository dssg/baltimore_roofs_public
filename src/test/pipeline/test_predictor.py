import unittest

from src.pipeline.predictor import get_damage_score_based_on_dark
from src.test.features.test_helpers import get_sample_image


class TestPredictor(unittest.TestCase):
    def test_get_damage_score_based_on_dark(self):
        blocklot = "0216 009.npy"
        grayscale_image = get_sample_image(blocklot, True)
        score = get_damage_score_based_on_dark(grayscale_image, 50)

        self.assertEqual(score, 0.303)


if __name__ == "__main__":
    unittest.main()
