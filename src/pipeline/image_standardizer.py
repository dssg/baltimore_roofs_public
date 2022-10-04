import torch
import torchvision.transforms.functional as transform
import torchvision


class ImageStandardizer:
    def __init__(self, output_dims=None, pad=True):
        self.output_dims = output_dims
        self.pad = pad

    def __call__(self, x):

        if self.pad:
            max_dim = max(x.shape[1], x.shape[2])
            padding = (int((max_dim - x.shape[2]) / 2), int((max_dim - x.shape[1]) / 2))
            x = transform.pad(x, padding, fill=0)

        if self.output_dims is not None:
            x = transform.resize(
                x,
                self.output_dims,
                interpolation=torchvision.transforms.InterpolationMode.BICUBIC,
            )
        x = x.nan_to_num()
        return x.to(torch.float32)


def standardize_tensors(blocklot_to_tensors: dict[str, list[torch.Tensor]]):
    standardizer = ImageStandardizer(pad=False)

    standardized_images = {}

    for blocklot, variants in blocklot_to_tensors.items():
        image_list = []
        for image in variants:
            image_list.append(standardizer(image))
        standardized_images[blocklot] = image_list
    return standardized_images
