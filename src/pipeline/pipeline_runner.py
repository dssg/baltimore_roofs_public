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

creator = MatrixCreator(
    config.matrix_creator, f"{config.schema_prefix}_model_prep.split_kinds"
)
creator.write_feature_matrices(creator.max_date)

trainer = ModelTrainer(config.model_trainer)
predictor = Predictor()
evaluator = Evaluator(config.evaluator)

models = trainer.models_to_train()
training_splits = splitter.get_training_splits()

pbar = tqdm(
    total=sum([len(s) for s in models.values()]) * len(training_splits),
    desc="Models",
    smoothing=0,
)

for model_class, hyperparam_combos in models.items():
    for hyperparams in hyperparam_combos:
        for split_id in training_splits:
            model_id = trainer.train_model_on_split(model_class, hyperparams, split_id)
            val_split_id = splitter.validation_split_for_train_split(split_id)
            predictor.write_preds_to_db(model_id, val_split_id)
            evaluator.write_evaluation_to_db(model_id, val_split_id)
            pbar.update()
pbar.close()

for model_group_id in tqdm(
    ModelTrainer.model_group_ids(), desc="Group eval", smoothing=0
):
    evaluator.write_group_evaluation_to_db(model_group_id)
