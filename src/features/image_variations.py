import logging
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms.functional as TF

import src.data.image_parser as parser

# Reorder given matplotlib and pytorch have different order of channel, height, width.
# pytorch:    [C, H, W]
# matplotlib: [H, W, C]
TENSOR_TO_NUMPY = [1, 2, 0]
NUMPY_TO_TENSOR = [2, 0, 1]


def generate_rotated_images(input_dir, angles):
    """
    Take all the .npy images in the input_dir
    Return tensors of images rotated at the specified angles
    """
    fnames = os.listdir(input_dir)
    rotated_tensors = []
    for f in fnames:
        img_f = np.load(input_dir / f)
        tensor = numpy_to_tensor(img_f)
        rotated_tensors += rotate_tensor(tensor, angles)
    return rotated_tensors


def tensor_to_numpy(t):
    return t.numpy().transpose(TENSOR_TO_NUMPY)


def numpy_to_tensor(n):
    return torch.from_numpy(n.transpose(NUMPY_TO_TENSOR))


def get_numpys_from_directory(input_dir):
    """
    Return a list of numpys from a specified directory
    """
    fnames = os.listdir(input_dir)
    np_arrays = []
    for f in fnames:
        np_arrays.append(np.load(input_dir / f))
    return np_arrays


def rotate_tensor(tensor, angles):
    """
    Take a tensor
    Return tensors of images rotated at the specified angles
    """
    return [
        TF.rotate(
            tensor.to(torch.uint8),
            angle,
            expand=True,
            interpolation=TF.InterpolationMode.BILINEAR,
        )
        for angle in angles
    ]


def view_tensors(tensors, *args, **kwargs):
    for t in tensors:
        view_tensor(t, *args, **kwargs)


def view_tensor(tensor, *args, **kwargs):
    plt.imshow(tensor_to_numpy(tensor), *args, **kwargs)
    plt.show()

def rotate_blocklot_tensors(blocklots, angles_for_rotation):
    train_blocklot_to_image = {}
    for blocklot in blocklots:
        image = parser.fetch_image(blocklot)
        if image is not None and len(image.shape) != 0:
            train_blocklot_to_image[blocklot] = numpy_to_tensor(image)
        else:
            logging.warning('No image found for %s', blocklot)

    return {
        blocklot: rotate_tensor(tensor, angles=angles_for_rotation)
        for blocklot, tensor in train_blocklot_to_image.items()
    }