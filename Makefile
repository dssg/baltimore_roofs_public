.PHONY: clean data lint requirements test convert_to_psql clean_tables

#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
BUCKET = [OPTIONAL] your-bucket-for-syncing-data (do not include 's3://')
PROFILE = default
PROJECT_NAME = baltimore-roofs
PYTHON_INTERPRETER = python3
DATA_PATH := $(shell $(PROJECT_DIR)/bin/yq '.data_path' < $(PROJECT_DIR)/experiment_config.yaml)

ifeq (,$(shell which conda))
HAS_CONDA=False
else
HAS_CONDA=True
endif

#################################################################################
# COMMANDS                                                                      #
#################################################################################

## Install Python Dependencies
requirements: test_environment
	$(PYTHON_INTERPRETER) -m pip install -U pip setuptools wheel
	$(PYTHON_INTERPRETER) -m pip install -r requirements.txt

## Delete all compiled Python files
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

## Lint using flake8
lint:
	flake8 src

## Set up python interpreter environment
create_environment:
ifeq (True,$(HAS_CONDA))
		@echo ">>> Detected conda, creating conda environment."
ifeq (3,$(findstring 3,$(PYTHON_INTERPRETER)))
	conda create --name $(PROJECT_NAME) python=3
else
	conda create --name $(PROJECT_NAME) python=2.7
endif
		@echo ">>> New conda env created. Activate with:\nsource activate $(PROJECT_NAME)"
else
	$(PYTHON_INTERPRETER) -m pip install -q virtualenv virtualenvwrapper
	@echo ">>> Installing virtualenvwrapper if not already installed.\nMake sure the following lines are in shell startup file\n\
	export WORKON_HOME=$$HOME/.virtualenvs\nexport PROJECT_HOME=$$HOME/Devel\nsource /usr/local/bin/virtualenvwrapper.sh\n"
	@bash -c "source `which virtualenvwrapper.sh`;mkvirtualenv $(PROJECT_NAME) --python=$(PYTHON_INTERPRETER)"
	@echo ">>> New virtualenv created. Activate with:\nworkon $(PROJECT_NAME)"
endif

## Test python environment is setup correctly
test_environment:
	$(PYTHON_INTERPRETER) test_environment.py

#################################################################################
# PROJECT RULES                                                                 #
#################################################################################

## Convert the gdb file into psql database
convert_to_psql:
	ogr2ogr -progress -f "PostgreSQL" -lco SCHEMA=$(PGSCHEMA) PG:"host=$(PGHOST) user=$(PGUSER) dbname=$(PGDATABASE) password=$(PGPASSWORD)" $(FILENAME)

## Parse out blocklot images from aerial orthographic tiles
blocklot_images:
	$(PYTHON_INTERPRETER) src/data/image_parser.py \
						  /mnt/data/projects/baltimore-roofs/data/20220614_harddrives \
						  /mnt/data/projects/baltimore-roofs/data/blocklot_images/2017 \
						  2017

## Parse out blocklot images from 2020 aerial orthographic tiles
blocklot_images_2020:
	$(PYTHON_INTERPRETER) src/data/image_parser.py \
						  /mnt/data/projects/baltimore-roofs/data/20220614_harddrives \
						  /mnt/data/projects/baltimore-roofs/data/blocklot_images/2020 \
						  2020

## Parse out blocklot images from 2022 aerial orthographic tiles
blocklot_images_2022:
	$(PYTHON_INTERPRETER) src/data/image_parser.py \
						  /mnt/data/projects/baltimore-roofs/data/2022_ortho_exports \
						  /mnt/data/projects/baltimore-roofs/data/blocklot_images/2022.hdf5

## Run unit tests
test:
	$(PYTHON_INTERPRETER) -m unittest

## Run the pipeline
train:
	$(PYTHON_INTERPRETER) src/pipeline/pipeline_runner.py
 
## Run predictions for a certain cohort with a certain model
predictions:
	$(PYTHON_INTERPRETER) src/predict.py

## Write out scores based on the config
scores:
	$(PYTHON_INTERPRETER) src/pipeline/list_creator.py

## Run transfer learning predictions on 2022 images
2022_image_predictions:
	$(PYTHON_INTERPRETER) src/pipeline/predictor.py \
		/mnt/data/projects/baltimore-roofs/data/blocklot_images/2022.hdf5 \
		c4a22829-26f5-4cdc-abd1-905016a56721 \
		jdcc_tl_dropout_6 jdcc_2022_tl_preds

## Import structured data into tables
import_tables:
	-$(PYTHON_INTERPRETER) src/data/import_tables.py $(DATA_PATH)/20220706/CompletedDemo_070622.xlsx raw demolitions_as_of_20220706
	-$(PYTHON_INTERPRETER) src/data/import_tables.py $(DATA_PATH)/20220706/Blocks_SampleAddress_2018.xlsx raw reference_addresses_for_2018_coding
	-$(PYTHON_INTERPRETER) src/data/import_tables.py $(DATA_PATH)/20220707/All_VBNs.csv raw all_vacant_building_notices
	-$(PYTHON_INTERPRETER) src/data/import_tables.py $(DATA_PATH)/20220707/Findings8.5.19.xls raw roofdata_2019

## Process raw tables into clean tables
clean_tables:
	$(PYTHON_INTERPRETER) src/data/process_tables.py

#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

# Inspired by <http://marmelab.com/blog/2016/02/29/auto-documented-makefile.html>
# sed script explained:
# /^##/:
# 	* save line in hold space
# 	* purge line
# 	* Loop:
# 		* append newline + line to hold space
# 		* go to next line
# 		* if line starts with doc comment, strip comment character off and loop
# 	* remove target prerequisites
# 	* append hold space (+ newline) to line
# 	* replace newline plus comments by `---`
# 	* print line
# Separate expressions are necessary because labels cannot be delimited by
# semicolon; see <http://stackoverflow.com/a/11799865/1968>
.PHONY: help
help:
	@echo "$$(tput bold)Available rules:$$(tput sgr0)"
	@echo
	@sed -n -e "/^## / { \
		h; \
		s/.*//; \
		:doc" \
		-e "H; \
		n; \
		s/^## //; \
		t doc" \
		-e "s/:.*//; \
		G; \
		s/\\n## /---/; \
		s/\\n/ /g; \
		p; \
	}" ${MAKEFILE_LIST} \
	| LC_ALL='C' sort --ignore-case \
	| awk -F '---' \
		-v ncol=$$(tput cols) \
		-v indent=19 \
		-v col_on="$$(tput setaf 6)" \
		-v col_off="$$(tput sgr0)" \
	'{ \
		printf "%s%*s%s ", col_on, -indent, $$1, col_off; \
		n = split($$2, words, " "); \
		line_length = ncol - indent; \
		for (i = 1; i <= n; i++) { \
			line_length -= length(words[i]) + 1; \
			if (line_length <= 0) { \
				line_length = ncol - indent - length(words[i]) - 1; \
				printf "\n%*s ", -indent, " "; \
			} \
			printf "%s ", words[i]; \
		} \
		printf "\n"; \
	}' \
	| more $(shell test $(shell uname) = Darwin && echo '--no-init --raw-control-chars')
