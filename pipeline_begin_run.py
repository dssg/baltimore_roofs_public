from tqdm.auto import tqdm

from src.config import experiment_config as config
from src.pipeline.data_splitter import DatasetSplitter
from src.pipeline.evaluator import Evaluator
from src.pipeline.labeler import Labeler
from src.pipeline.matrix_creator import MatrixCreator
from src.pipeline.model_trainer import ModelTrainer
from src.pipeline.predictor import Predictor

from src.db import maybe_drop_existing_schemas


maybe_drop_existing_schemas(config.schema_prefix)

splitter = DatasetSplitter(config.splitter, config.cohort.query, config.random_seed)
splitter.write_splits_to_db()

blocklots = DatasetSplitter.blocklots_for_all_splits()
Labeler(config.labeler).write_labels_to_db(blocklots)