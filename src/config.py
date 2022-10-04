from pathlib import Path

import yaml
from munch import DefaultMunch

currdir = Path(__file__).resolve().parent
fname = currdir.parent / "experiment_config.yaml"
with open(fname) as f:
    experiment_config = DefaultMunch.fromDict(yaml.safe_load(f))
