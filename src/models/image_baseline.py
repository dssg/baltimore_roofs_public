import logging

import numpy as np
import torch
from torchvision.transforms.functional import rgb_to_grayscale

from src.features.image_variations import numpy_to_tensor

DARK_THRESHOLDS = range(10, 200, 10)  # 10, 20 .. 200
PROP_DARK_THRESHOLDS = np.linspace(0.1, 0.9, 9)  # .1, .2, ... .9


class DarkImageBaseline:
    def __init__(self, dark_threshold):
        self.threshold = dark_threshold

    def fit(self, X, y):
        return self

    def preprocess(self, X):
        numpy_to_tensor(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        # How much of each image is darker than the darkness threshold?
        if X.ndim == 0:
            return None
        X = rgb_to_grayscale(numpy_to_tensor(X).unsqueeze(0))
        is_darker = torch.where(X.isnan(), X, X < self.threshold).to(torch.float32)
        return is_darker.nanmean(axis=(2, 3), dtype=torch.float32).squeeze().item()


def predict_damage_based_on_darkness(
    grayscale_image, dark_channel_threshold, dark_prop_threshold
):
    """Predict whether this is an image of a damaged roof using
       the following baseline model:
       If the grayscale image has more than dark_prop_threshold pixels that are "dark"
       (channel below dark_threshold),
       then we predict this is an image of a damaged roof

     Args:
        grayscale_image: Tensor image that has been converted from RGB to grayscale
        dark_channel_theshold: a channel below is a "dark" pixel
        dark_prop_threshold: the proportion of "dark" pixels needed for predicition

    Returns:
        Boolean value
        Prediction of whether the image is of a damaged (True) or undamaged roof (False)
    """
    logging.debug(
        f"Predicting for an image of shape {grayscale_image.shape} for "
        + f"dark_channel_threshold {dark_channel_threshold} and "
        + f"dark_prop_threshold {dark_prop_threshold}"
    )
    return (
        get_prop_of_dark(grayscale_image, dark_channel_threshold) > dark_prop_threshold
    )


def get_damage_score_based_on_dark(grayscale_image, dark_channel_threshold):
    """Predict the damage score of the grayscale image
       calculated by finding the percent of pixels which are
       "dark" (channel below dark_channel_threshold)

     Args:
        grayscale_image: Tensor image that has been converted from RGB to grayscale
        dark_channel_theshold: a channel below is a "dark" pixel

    Returns:
        Score of dark pixels (value between 0-1)
        rounded to three decimal places
    """
    prop_of_dark = get_prop_of_dark(grayscale_image, dark_channel_threshold)
    return round(prop_of_dark.item(), 3)


def get_prop_of_dark(grayscale_image, dark_channel_threshold):
    is_dark_pixel = grayscale_image < dark_channel_threshold
    is_not_nan_pixel = grayscale_image.isnan().logical_not()

    return is_dark_pixel.sum() / is_not_nan_pixel.sum()


def get_thresholds_to_predictions(
    images, dark_channel_thresholds, dark_prop_thresholds
):
    """Note: This is a temporary method that won't be used.
    Return map of thresholds (dark_channel_threshold, dark_prop_threshold)
       to the number of predictions of damaged roofs

     Args:
        images: Tensor grayscale images to make predictions for
        dark_channel_thesholds: List of thresholds that determine if a pixel is dark
        dark_prop_threshold: List of thresholds that determine if an image is dark

    Returns:
        Dictionary of thresholds to predictions
    """
    logging.info(
        f"making predictions for {len(dark_channel_thresholds)} dark_channel_thresholds "
        + f"and {len(dark_prop_thresholds)} dark_prop_thresholds."
    )
    thresholds_to_predictions = {}
    for proportion_threshold in dark_prop_thresholds:
        for channel_threshold in dark_channel_thresholds:
            list_predictions = []
            for image in images:
                list_predictions.append(
                    predict_damage_based_on_darkness(
                        image, channel_threshold, proportion_threshold
                    )
                )

            thresholds_to_predictions[
                (channel_threshold, proportion_threshold)
            ] = list_predictions
    return thresholds_to_predictions
