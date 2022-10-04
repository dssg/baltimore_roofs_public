import click
import h5py
import numpy as np
from psycopg2 import sql
from tqdm.auto import tqdm
from typing import List

from src.config import experiment_config as config
from src.db import batch_insert, run, run_query
from src.pipeline.data_splitter import DatasetSplitter
from src.pipeline.model_trainer import ModelTrainer


class Predictor:
    def __init__(self, config=None):
        self.config = config

    @classmethod
    def _create_table(cls, model_id, schema_prefix=config.schema_prefix):
        preds_table = cls.model_table(model_id)
        run(
            f"""
            CREATE SCHEMA IF NOT EXISTS {schema_prefix}_model_predictions
            AUTHORIZATION "baltimore-roofs-role"
        """
        )
        run(
            sql.SQL(
                """
            CREATE TABLE IF NOT EXISTS {schema}.{table} (
                blocklot varchar(10),
                score real
            )
        """
            ).format(
                table=sql.Identifier(preds_table),
                schema=sql.Identifier(f"{schema_prefix}_model_predictions"),
            )
        )
        run(
            sql.SQL(
                """
            CREATE UNIQUE INDEX IF NOT EXISTS {index}
            ON {schema}.{table} (blocklot)
        """
            ).format(
                table=sql.Identifier(preds_table),
                index=sql.Identifier(f"{preds_table}_blocklot_idx"),
                schema=sql.Identifier(f"{schema_prefix}_model_predictions"),
            )
        )

    @staticmethod
    def model_table(model_id):
        return f"preds_of_{model_id}"

    def predict(self, model_id, split_id, schema_prefix=config.schema_prefix):
        model, model_details = self.load_model(model_id, schema_prefix)
        feature_set = model_details["feature_set"]
        X = ModelTrainer.fetch_X(
            split_id, feature_set=feature_set, schema_prefix=schema_prefix
        )
        if feature_set == "image":
            return self.predict_image(X, model)
        if feature_set == "matrix":
            return self.predict_matrix(X, model)
        if feature_set == "blocklots":
            return self.predict_blocklots(X, model)
        raise Exception("Unknown feature set")

    @staticmethod
    def blocklots_in_hdf5(filename):
        blocklots = []
        f = h5py.File(filename, "r")
        for block in f:
            for lot in f[block]:
                if f[f"{block}/{lot}"].size > 1:
                    blocklots.append(f"{block:5}{lot}")
        return blocklots

    @staticmethod
    def load_model(model_id, schema_prefix):
        model_details = ModelTrainer.load_model_details(model_id, schema_prefix)
        model = model_details["model"]
        if "class" in model_details:
            model_class = model_details["class"]
            if hasattr(model_class, "load"):
                model = model_class.load(model)
        return model, model_details

    def make_preds_for_hdf5(
        self, model_id, filename, schema_prefix=config.schema_prefix
    ):
        model, model_details = self.load_model(model_id, schema_prefix)
        feature_set = model_details["feature_set"]
        X = self.blocklots_in_hdf5(filename)
        if feature_set == "image":
            return self.predict_image(X, model)
        if feature_set == "matrix":
            return self.predict_matrix(X, model)
        if feature_set == "blocklots":
            return self.predict_blocklots(X, model)
        raise Exception("Unknown feature set")

    @staticmethod
    def hdf5_to_numpy(dataset, shape=None):
        arr = np.empty_like(dataset)
        dataset.read_direct(arr)
        return arr

    def predict_image(self, X, model):
        outputs = {}
        for blocklot, image in tqdm(X.items()):
            if isinstance(image, h5py.Dataset):
                image = self.hdf5_to_numpy(image)
            outputs[blocklot] = model.predict_proba(image)
        return outputs

    def predict_matrix(self, X, model):
        # We're relying on dict maintaining the order of the blocklots here.
        return dict(zip(X.index.values, model.predict_proba(X.values)[:, 1]))

    def predict_blocklots(self, X: List[str], model) -> np.ndarray:
        return model.predict_proba(X)

    @classmethod
    def load_preds(cls, model_id, split_id, schema_prefix=config.schema_prefix):
        blocklots = DatasetSplitter.blocklots_for_split(split_id, schema_prefix)
        model_table = cls.model_table(model_id)
        results = run_query(
            sql.SQL(
                """
            SELECT blocklot, score FROM {schema}.{table}
            WHERE blocklot IN %s
        """
            ).format(
                table=sql.Identifier(model_table),
                schema=sql.Identifier(f"{schema_prefix}_model_predictions"),
            ),
            (tuple(blocklots),),
        )
        return (
            {row["blocklot"]: row["score"] for row in results}
            if len(results) > 0
            else None
        )

    @classmethod
    def load_all_preds(cls, model_id, schema_prefix=config.schema_prefix):
        model_table = cls.model_table(model_id)
        results = run_query(
            sql.SQL("""SELECT blocklot, score FROM {schema}.{table}""").format(
                table=sql.Identifier(model_table),
                schema=sql.Identifier(f"{schema_prefix}_model_predictions"),
            )
        )
        return (
            {row["blocklot"]: row["score"] for row in results}
            if len(results) > 0
            else None
        )

    def write_preds_to_db(self, model_id, split_id, schema_prefix=config.schema_prefix):
        self._create_table(model_id, schema_prefix)
        preds = self.predict(model_id, split_id, schema_prefix)
        self.write_completed_preds_to_db(model_id, preds, schema_prefix)

    def write_completed_preds_to_db(
        self, model_id, preds, schema_prefix=config.schema_prefix
    ):
        self._create_table(model_id, schema_prefix)
        rows = []
        for blocklot, pred in preds.items():
            if pred is not None:
                pred = round(pred, 6)
            rows.append([blocklot, pred])
        batch_insert(
            sql.SQL(
                """
            INSERT INTO {schema}.{table} (blocklot, score) VALUES %s
            ON CONFLICT DO NOTHING
        """
            ).format(
                table=sql.Identifier(self.model_table(model_id)),
                schema=sql.Identifier(f"{schema_prefix}_model_predictions"),
            ),
            rows,
        )

    @classmethod
    def preds_exist(cls, model_id, split_id):
        result = run_query(
            f"""
            SELECT EXISTS (
                SELECT FROM
                    pg_tables
                WHERE
                    schemaname = '{config.schema_prefix}_model_predictions' AND
                    tablename  = %s
                );
            """,
            (cls.model_table(model_id),),
        )
        table_exists = result[0][0]
        if not table_exists:
            return False
        result = run_query(
            sql.SQL(
                """SELECT COUNT(*) FROM
                    {schema}.{table}"""
            ).format(
                table=sql.Identifier(cls.model_table(model_id)),
                schema=sql.Identifier(f"{config.schema_prefix}_model_predictions"),
            )
        )
        return result[0][0] > 0


@click.command()
@click.argument("image_path", type=click.Path(exists=True))
@click.argument("model_id", type=str)
@click.argument("schema_prefix", type=str)
@click.argument("output_schema_prefix", type=str)
def predict_on_hdf5(image_path, model_id, schema_prefix, output_schema_prefix):
    predictor = Predictor()
    preds = predictor.make_preds_for_hdf5(model_id, image_path, schema_prefix)
    predictor.write_completed_preds_to_db(model_id, preds, output_schema_prefix)


if __name__ == "__main__":
    predict_on_hdf5()
