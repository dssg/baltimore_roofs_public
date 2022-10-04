from pathlib import Path
import unittest
import os

from src.features.image_variations import generate_rotated_images
from src.test.features.test_helpers import get_sample_images_directory


class TestGenerateRotatedImages(unittest.TestCase):
    def test_it_generates_new_images(self):
        input_dir = get_sample_images_directory()
        angles = [i * 60 for i in range(6)]  # 0,60,120,180,240,300

        rotated_image_tensors = generate_rotated_images(input_dir, angles)

        num_input_images = len(os.listdir(input_dir))
        self.assertEqual(len(rotated_image_tensors), num_input_images * len(angles))


if __name__ == "__main__":
    unittest.main()
