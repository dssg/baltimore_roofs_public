import click
from tqdm.auto import tqdm

from src.config import experiment_config as config
from src.db import maybe_drop_existing_schemas
from src.pipeline.data_splitter import DatasetSplitter
from src.pipeline.matrix_creator import MatrixCreator
from src.pipeline.model_trainer import ModelTrainer
from src.pipeline.predictor import Predictor


@click.command()
@click.argument("image_path", type=click.Path(exists=True))
@click.argument("model_id", type=str)
@click.argument("schema_prefix", type=str)
@click.argument("output_schema_prefix", type=str)
def predict_on_hdf5(image_path, model_id, schema_prefix, output_schema_prefix):
    predictor = Predictor()
    preds = predictor.make_preds_for_hdf5(model_id, image_path, schema_prefix)
    predictor.write_completed_preds_to_db(model_id, preds, output_schema_prefix)


def make_and_save_predictions(model, model_details, model_id, X=None):
    feature_set = model_details["feature_set"]
    if feature_set == "matrix":
        if X is None:
            matrix_creator = MatrixCreator(config.matrix_creator, None)
            X = matrix_creator.calc_feature_matrix_for_blocklots(blocklots)
            matrix_creator.save_matrix_to_disk(X, 1)

        preds = predictor.predict_matrix(X, model)
        predictor.write_completed_preds_to_db(model_id, preds, config.schema_prefix)

    if feature_set == "blocklots":
        assert "image_path" in config.predictor
        predict_on_hdf5(
            config.predictor.image_path,
            config.predictor.model.id,
            config.predictor.model.schema_prefix,
            config.schema_prefix,
        )


if __name__ == "__main__":
    assert "model" in config.predictor
    assert ("id" in config.predictor.model) or ("group_id" in config.predictor.model)
    assert "schema_prefix" in config.predictor.model

    maybe_drop_existing_schemas(config.schema_prefix)

    blocklots = DatasetSplitter.cohort_blocklots()
    predictor = Predictor()

    if "id" in config.predictor.model:
        model, model_details = predictor.load_model(
            config.predictor.model.id, config.predictor.model.schema_prefix
        )
        make_and_save_predictions(model, model_details, config.predictor.model.id)
    elif "group_id" in config.predictor.model:
        matrix_creator = MatrixCreator(config.matrix_creator, None)
        X = matrix_creator.calc_feature_matrix_for_blocklots(blocklots)
        matrix_creator.save_matrix_to_disk(X, 1)

        for model_id in tqdm(
            ModelTrainer.models_in_group(
                config.predictor.model.group_id, config.predictor.model.schema_prefix
            ).keys(),
            desc="Models",
            smoothing=0,
        ):
            model, model_details = Predictor.load_model(
                model_id, config.predictor.model.schema_prefix
            )
            make_and_save_predictions(model, model_details, model_id, X)
