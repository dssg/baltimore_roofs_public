Finding Roof Damage in Baltimore City
=====================================

Table of Contents
-----------------

1. [Background](#background)
2. [Requirements](#requirements)
3. [Steps](#steps)
    * [3.1 Load the data](#1-load-the-data)
    * [3.2 Cut Out Blocklot Images](#2-cut-out-blocklot-images)
    * [3.3 Set the Experiment Config](#3-set-the-experiment-config)
    * [3.4 Tensorboard setup](#4-tensorboard-setup)
    * [3.5 Run the Pipeline](#5-run-the-pipeline)
    * [3.6 Making Predictions](#6-making-predictions)
    * [3.7 Outputting Score Lists](#7-outputting-score-lists)
4. [Running Tests](#running-tests)
5. [Project Organization](#project-organization)
6. [Contributors](#contributors)
7. [Acknowledgements](#acknowledgements)

Background
----------

The city of Baltimore has been facing population decline over five consecutive decades. As a result, there are 15,000 vacant buildings that the city is currently aware of, many of which are not maintained. These abandoned buildings negatively impact neighbors and neighborhoods, especially because structural damage tends to spread to adjacent homes.

The Baltimore City [Department of Housing and Community Development (DHCD)](https://dhcd.baltimorecity.gov/) partnered with the [Data Science for Social Good](https://www.dssgfellowship.org/) summer fellowship program to tackle this issue. Using a number of models trained on aerial images of Baltimore City and other inspections and housing data provided by the city, it is possible to predict a score of roof damage for each residential row home in Baltimore City. This allows DHCD to prioritize properties in the late stage of deterioration and intervene in a timely and effective manner.

This repository hosts the code for loading, cleaning, and parsing data, and training and evaluating these models.

Requirements
----------

All development and running of the code was done on Linux machines, primarily [Ubuntu 22.04](https://releases.ubuntu.com/22.04/). While this isn't strictly a requirement, it is the only tested configuration of the hardware so far.

For running predictions:

* [Python](https://www.python.org/) >= 3.10
* [PostgreSQL](https://www.postgresql.org/) >= 13
* [PostGIS](https://postgis.net/) extensions for PostgreSQL
* [GDAL](https://gdal.org/)
* Once these are installed and the code is downloaded, the rest of the requirements can be installed with `make requirements`.

Additional requirements for training:

* You're probably going to want a machine with a GPU that supports either [CUDA](https://developer.nvidia.com/cuda-downloads) (Nvidia) or [ROCm](https://www.amd.com/en/graphics/servers-solutions-rocm) (AMD). A computer with a regular CPU will also do in a pinch, but things will run *much* slower.
* The [PyTorch](https://pytorch.org/) version (>= 1.12) that runs on the computing infrastucture you have available.

### Development specific to DSSG

1. To load the project-specific environment variables (such as python path),
cd to your directory `/mnt/data/projects/baltimore-roofs/[user]` (replace `[user]` with your username) and add the following two lines to your `.envrc` file:

   ```bash
   source_up
   export PYTHONPATH="/mnt/data/projects/baltimore-roofs/[username]/baltimore_roofs"
   ```

   Without the above, there will be issues finding the correct path.

2. Set `.env` following the example `.env.sample`

#### Setting up ID

* Select the right python interpreter path:
Command + shift + p (for mac) & select Python ('.venv')

Steps
-----


### 1. Load the data

The files obtained were in gdb format used by [Geographic Information System](https://en.wikipedia.org/wiki/Geographic_information_system) software like [ArcGIS](https://www.arcgis.com/) or [QGIS](https://qgis.org/).
It basically consisted of 16 tables with georeference data. In order to properly process all this information we will store it on a PostgreSQL database.

The conversion for the SQL database can be easily done using [GDAL](https://gdal.org/). The function for the conversion is called *ogr2ogr* and can be easily accessed via the *make* command below. Please make sure to replace the variables marked by {brackets} with the appropriate value. You may also store this information in your environment or on a .env file to skip manually typing all the parameters. This repository has a .env.sample illustrating the parameters required.

```bash
make convert_to_psql FILENAME={filename} PGSCHEMA={schema} PGHOST={host} PGUSER={USER} PGDATABASE={db} PGPASSWORD={password}
```

Run the command for all files available. Once the conversion is done, the schema defined should contain all 16 tables.

To load CSV or Excel files into the database, run `make import_tables`.

Run the command for all files avaialable. Once the conversion is done, the schema defined should contain all 16 tables.

### 2. Cut Out Blocklot Images

The image data used in this project are orthographic photographs that tile all of Baltimore City. We can't use these tiles as-is because each contains many properties. To run a classification model, we need to cut out the image of each blocklot from their respective tiles. When run from the project root, the command to cut out out these blocklot images is:

```bash
python3 src/data/image_parser.py [OPTIONS] IMAGE_ROOT OUTPUT_ROOT
```

* `IMAGE_ROOT`: The location on the filesystem where the current image tiles are stored.
* `OUTPUT_ROOT`: The location on the filesystem where the new blocklot images should be stored. This stores the blocklot images to an [HDF5](https://www.h5py.org/) file by default.
* `OPTIONS`: Run `python3 src/data/image_parser.py --help` for all the options.

The command we used to parse the 2017 images is available as `make blocklot_images`.

### 3. Set the Experiment Config

The experiment_config.yaml file contains the configurations to run the pipeline with. The current setup creates a symbolic link to the existing target with the specified name link `my_experiment_config.yaml`. This allows you to specify your own configurations.

**Before you start, set the `schema_prefix` of your experiment (e.g. `chaewonl_pixel_baseline`)**

The file contains:

* `blocklot_image_path`: specifies the path to where the hdf5 image files are.

* `schema_prefix`: A database schema represents the storage of your data in a database. The schema prefix will be used to output tables to the following schemas: `{schema_prefix}`_model_predictions, `{schema_prefix}`_model_prep, and `{schema_prefix}`_model_results.

* `cohort`: defines the group of interest that we want to run models on (e.g. blocklots that have buildings on them and are zones as residential). The cohort `query` specifies the query to get this group from the database. Note the the sample `my_experiment_config.yaml` limits to 2000 to run quickly for a trial run - you'll likely want to remove this.

* `splitter`: specifies how to split the data to `n_folds` for model training.

* `labeler`: defines how to label each blocklots. Currently, the `query` labels the blocklots that looked over but not labeled by interns as 0.

* `matrix_creator`: specifies how to construct the different features you want to try.

* `model_trainer`: specifies how to train models. It lists the different model classes to use, what features they use, and the parameters for the model class.

* `evaluator`: shows the precision recall graphs that allow us to measure the performance for different proporitons of the population.

* `bias_auditor`: takes a single model and performs an audit bias using [aequitas](http://www.datasciencepublicpolicy.org/our-work/tools-guides/aequitas/)

### 4. Tensorboard setup

While training the deep learning model, it might be useful to visualize the performance obtained, especially when using a combination of different hyperparameters during this process. [Tensorboard](https://www.tensorflow.org/tensorboard/) is a powerful tool for such a task as it provides a clean and user-friendly interface with results as they are obtained.
The tool will write files under the "model_trainer/model_dir" folder specified on the .yaml config file with the values of each run. A Tensorboard server will basically translate the information contained in those files into a visual interface accessed via a browser.
Running this server is not necessary to run the model and only provides a closer observation of the training process. Hence, this step can be skipped.

To run the Tensorboard server it is first necessary to have it installed. It can be installed with the command
`pip install tensorboard` .

After that, you can run the server by typing (preferentially on a [screen](https://www.gnu.org/software/screen/) window)

    tensorboard --logdir={model_dir}
Make sure to replace {model_dir} with the same directory chosen on the yaml config file. By default, the server will start on port 6006, making the dashboard accessible on your browser via localhost:6006. It may be necessary to create a tunnel between your local machine and the server to port forward the 6006 endpoint.

### 5. Run the pipeline

All the process of training and evaluating the model is contained in one command `make train`. This command calls for the file 'pipeline_runner' in 'src/pipeline'.

The script will automatically obtain information from the config file, including the models that will be run. The *pipeline_runner* will perform a training and evaluation run for each model present on the config file under the 'model_trainer' parameter. Similarly, if more than one hyperparameter is present in the config file, the combination of all these hyperparameters will be calculated and an additional run will be executed for each.

Each run is composed of a splitting, training, validation, predictor, and evaluator. The first step consists of splitting the training data into as many sets as defined by the user in the config file. For each combination of sets, the model trainer will be called using the hyperparameters for that run.

With the model obtained, the next step will be the predictor, which consists of applying the trained model to the validation set to obtain predictions.

Lastly, the evaluator will be called, assessing the number of false positives and false negatives on the combined validation sets.

Throughout the process, all the split ids, the model id, the model files, predictions, and all information regarding it are stored on the database under the schemas named on the config file.

### 6. Making Predictions

To make predictions, change the `experiment_config.yaml` to the desired settings (specifically the `predictor` section) and run `make predictions`.

### 7. Outputting Score Lists

The above steps will put predictions in the database, but for sharing scores out to others, it's useful to have them in a friendlier format. After configuring the `list_creator` elements of the `experiment_config.yaml`, run `make scores` to output scores in CSV format.

Running Tests
-------------

To run the unit tests, run `make test`. Run `direnv allow`
Ideally, this can be run before and after new code is added to make sure functionality isn't broken.

Project Organization
--------------------

```text
├── LICENSE
├── Makefile               <- Makefile with commands like `make blocklot_images` or
|                             `make predictions`.
├── README.md              <- The top-level README for developers using this project.
│
├── models                 <- Trained and serialized models.
│
├── notebooks              <- Jupyter notebooks. Naming convention is the creator's initials,
|                             a number of ordering, and a short `-` delimited description, e.g.
│                             `jdcc_0.2_formulation_eda.ipynb`.
│
├── requirements.txt       <- The requirements file for reproducing the analysis environment, e.g.
│                             generated with `pip freeze > requirements.txt`
│
├── setup.py               <- makes project pip installable (pip install -e .) so src can be imported
├── src                    <- Source code for use in this project.
│   ├── __init__.py        <- Makes src a Python module
│   │
│   ├── data               <- Scripts to download or generate data
│   │
│   ├── features           <- Scripts to turn raw data into features for modeling
│   │
│   ├── models             <- Implmentations of various models
│   │
│   └── pipeline           <- All the scripts related to running the modeling pipeline.
│
├── bin                    <- Binary executables
|
├── experiment_config.yaml <- yaml file with settings for running pipeline
└── tox.ini                <- tox file with settings for running tox; see tox.readthedocs.io
```

Contributors
------------

Those listed below have contributed directly to the implementation of this project (lastname alphabetical order):

* Justin Clark
* Jonas Coelho
* Chae Won Lee

Those listed below have contributed to the project:

* Kit Rodolfa as technical mentor
* Adolfo De Unánue as data science support
* Abiola Oyebanjo as project manager

Acknowledgements
----------------

This work was conducted in the [Data Science for Social Good Fellowship](https://www.dssgfellowship.org/) at [Carnegie Mellon University](https://www.cmu.edu/) with the support from [Baltimore City's Department of Housing and Community Development (DHCD)](https://dhcd.baltimorecity.gov/).

Specifically, we want to acknowledge the contribution of our partners at DHCD:

* Henry Waldron
* Jason Hessler
* Justin Elszasz
