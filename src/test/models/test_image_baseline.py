import unittest

import numpy as np
import torch
from torchvision.transforms import Resize

from src.models.image_baseline import DarkImageBaseline
from src.test.features.test_helpers import (
    get_sample_images_blocklots,
    get_sample_images,
)


class TestDarkImageBaseline(unittest.TestCase):
    def test_fit(self):
        d = DarkImageBaseline(0)
        self.assertEqual(d.fit([], []), d)

    def test_predict_proba(self):
        blocklots = get_sample_images_blocklots()
        images = get_sample_images(blocklots, grayscale=True)
        resizer = Resize((256, 256))
        X = torch.stack([resizer(i) for i in images])
        d = DarkImageBaseline(30)
        scores = d.predict_proba(X)
        print(scores)
