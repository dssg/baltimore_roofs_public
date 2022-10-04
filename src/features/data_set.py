import torch
from tqdm.auto import tqdm

from src.data.image_parser import fetch_image
from src.features.image_variations import numpy_to_tensor

import torchvision.transforms.functional as TF


class BlocklotDataset(torch.utils.data.Dataset):
    "Characterizes a blocklot dataset for PyTorch"

    def __init__(self, df, transform=None, angle_variation=[0]):
        "Initialization"
        self.blocklot_ids = df["blocklot"].to_list()
        self.len_blocklots = len(self.blocklot_ids)
        self.angle_variations = angle_variation
        self.angles_len = len(self.angle_variations)
        self.angles = self.angle_variations * self.len_blocklots
        self.blocklot_ids = [
            val for val in self.blocklot_ids for _ in range(self.angles_len)
        ]

        if "label" in df.columns:
            self.labels = df["label"].to_list()
            self.labels = [val for val in self.labels for _ in range(self.angles_len)]
        else:
            df["label"] = None
        self.transform = transform

    def __len__(self):
        "Denotes the total number of samples"
        return len(self.blocklot_ids)

    def __getitem__(self, index):
        self.get_image(index)

    def get_image(self, index):
        "Generates one sample of data"
        # Select sample
        blocklot_id = self.blocklot_ids[index]

        # Load data and get label
        image_data = fetch_image(blocklot_id)
        label = self.labels[index]

        angle = self.angles[index]

        image_data = TF.rotate(
            numpy_to_tensor(image_data).to(torch.uint8),
            angle,
            expand=True,
            interpolation=TF.InterpolationMode.BILINEAR,
        )

        if self.transform is None:
            return image_data, label

        return self.transform(image_data), label


class InMemoryBlocklotDataset(BlocklotDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dataset = [
            self.get_image(i)
            for i in tqdm(
                range(len(self)), desc="Loading images", leave=False, smoothing=0
            )
        ]

    def __getitem__(self, index):
        return self.dataset[index]
