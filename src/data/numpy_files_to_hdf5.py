import logging
from pathlib import Path

import h5py
import numpy as np
from tqdm.auto import tqdm

from src.config import experiment_config as config

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

output_path = Path(config.blocklot_image_path).parent / "2017_buffered_20.hdf5"
logger.info("Writing to %s", output_path)
print(output_path)
print()

f = h5py.File(str(output_path), "w")
filenames = list(Path(config.blocklot_image_path).glob("*/*.npy"))

for filename in tqdm(filenames, smoothing=0):
    block = filename.parent.name
    lot = filename.stem
    data = np.load(filename)
    f.create_dataset(f"{block}/{lot}", data=data)
f.close()
