from pathlib import Path
import pandas as pd
import numpy as np
import torchvision.transforms.functional as transform
import torchvision

from src.features.data_splitter import get_split_data
from src.features.image_variations import rotate_tensor, numpy_to_tensor
from src.db import run_query


# Get Dataset: 2018 data
results = run_query(
    """
        SELECT tp.blocklot, 
            -- Limited to attributes that are particularly interesting or not empty 
            roof.mainrooffe as roof_damage, mainroofpc as roof_damage_score,
            bumproofpc as bump_damage_score,
            groupstatu AS demo_status, decision AS demo_decision, 
            iia.name AS iia_name, 
            onv.datenotice AS vacancy_notice, onv.noticetype 
        FROM raw.tax_parcel_address AS tp
        LEFT JOIN raw.building_construction_permits AS bcp
            ON tp.blocklot = bcp.blocklot
        LEFT JOIN raw.communitydevelopmentzones AS cdz
            ON ST_Intersects(tp.wkb_geometry, cdz.shape)
        LEFT JOIN raw.completedcitydemolition AS ccd
            ON ST_Intersects(tp.wkb_geometry, ccd.shape)
        LEFT JOIN raw.developmentdivisionprojects AS ddp
            ON ST_Intersects(tp.wkb_geometry, ddp.shape)
        LEFT JOIN raw.filedandopenreceiverships AS r
            ON tp.blocklot = r.blocklot   
        LEFT JOIN raw.impactinvestmentareas AS iia
            ON ST_Intersects(tp.wkb_geometry, iia.shape)
        LEFT JOIN raw.majorredevelopmentareas AS mra
        ON ST_Intersects(tp.wkb_geometry, mra.shape)
        LEFT JOIN raw.open_notice_vacant AS onv
            ON ST_Intersects(tp.wkb_geometry, onv.wkb_geometry)
        LEFT JOIN raw.real_estate_data AS red 
            ON tp.blocklot = red.blocklot 
        LEFT JOIN raw.roofdata_2018 AS roof
            ON tp.blocklot = roof.blocklot
        WHERE mainrooffe is not NULL
    """
)

df = pd.DataFrame(results, columns=results[0].keys())
print("df shape: " + str(df.shape))

# Filter to Cohort: SKIP

# Labeler
label_one = df.query(
    "roof_damage_score in ['10-24', '25-49', '50-99', '100'] | bump_damage_score in ['10-24', '25-49', '50-99', '100']"
)
df["label"] = df["blocklot"].isin(label_one["blocklot"]).astype("int")
print(f"number of high/med roof_damage {df.label.sum()}")

# TODO: check number
blocklot_list = df["blocklot"].unique()
print(f"len of blocklots {len(blocklot_list)}")


def fetch_image(blocklot):
    block = blocklot[:5].strip()
    lot = blocklot[5:].strip()
    base_dir = Path("/mnt/data/projects/baltimore-roofs/data/blocklot_images/2018/")
    img_f = base_dir / block / f"{lot}.npy"
    try:
        return np.load(img_f)
    except Exception:
        print(f"Missing {img_f}")


# Data Splitter
train, test = get_split_data(df)
print("train shape: " + str(train.shape))

# Image Variation

train_images = {}
for b in train.blocklot.unique():
    image = fetch_image(b)
    if image is not None and len(image.shape) != 0:
        train_images[b] = numpy_to_tensor(image)
# train_images = {b: numpy_to_tensor(fetch_image(b)) for b in train.blocklot.unique()}

rotated_tensors = {
    b: rotate_tensor(tensor, angles=[0, 90, 180, 270])
    for b, tensor in train_images.items()
}
print(f"len rotated tensor: {len(rotated_tensors)}")

# Image Standardization
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
        # x = x.nan_to_num()
        return x


standardizer = ImageStandardizer(pad=False)

standardized_images = {}

for b, variants in train_images.items():
    image_list = []
    for image in variants:
        image_list.append(standardizer(image))
    standardized_images[b] = image_list


list_images = []
list_labels = []
# Model Trainer
for b, variants in standardized_images.items():
    # get label from train?
    label = train.query(f"blocklot == '{b}'")["label"].values[0]
    for image in variants:
        list_images.append(image)
        list_labels.append(label)


def dark_image(tensor_image):
    # pytorch:    [C, H, W]
    tensor_image.sum(0)


pixel_count = pd.DataFrame

# Predictor

# Evaluator
