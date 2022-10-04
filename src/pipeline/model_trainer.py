import hashlib
from itertools import product
from pathlib import Path
from typing import Sequence
import uuid

from joblib import dump, load
import numpy as np
import pandas as pd
from psycopg2.extras import Json
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from tqdm.auto import tqdm

from src.config import experiment_config as config
from src.db import run_query, run
from src.data.image_parser import fetch_image
from src.models.standardized_logistic_regression import StandardizedLogisticRegression
from src.models.image_baseline import DarkImageBaseline
from src.pipeline.data_splitter import DatasetSplitter
from src.models.transfer_learning import TransferLearning


SLUG_TO_MODEL = {
    "decision_tree": DecisionTreeClassifier,
    "logistic_regression": StandardizedLogisticRegression,
    "random_forest": RandomForestClassifier,
    "dark_image_baseline": DarkImageBaseline,
    "boosted_tree": HistGradientBoostingClassifier,
    "transfer_learning": TransferLearning,
}


class ModelTrainer:
    def __init__(self, config):
        self.model_classes = config.model_classes
        self.model_dir = config.model_dir
        self.model_class_to_feature_set = {
            model_class: details.features
            for model_class, details in config.model_classes.items()
        }
        self.use_cache = config.use_cache

    @staticmethod
    def calc_param_space(param_values):
        param_names = list(param_values.keys())
        return [
            dict(zip(param_names, [None if v == "None" else v for v in values]))
            for values in product(*param_values.values())
        ]

    @staticmethod
    def _create_table():
        run(
            f"""
            CREATE SCHEMA IF NOT EXISTS {config.schema_prefix}_model_results
            AUTHORIZATION "baltimore-roofs-role"
        """
        )
        run('CREATE EXTENSION IF NOT EXISTS "pgcrypto"')
        run(
            f"""
            CREATE TABLE IF NOT EXISTS {config.schema_prefix}_model_results.models (
                id varchar(36) PRIMARY KEY,
                model_class varchar(50),
                trained_on_split_id integer
                    REFERENCES {config.schema_prefix}_model_prep.split_kinds (id),
                trained_on_feature_set varchar(50),
                hyperparams json,
                model_group varchar(40) GENERATED ALWAYS AS (
                    encode(
                        digest(
                            model_class
                            || '-' || trained_on_feature_set ||
                            '-' || hyperparams::text, 'sha1'), 'hex')) STORED,
                filepath varchar(300)
            )
        """
        )

    @classmethod
    def fetch_X(
        cls, split_id, feature_set="matrix", schema_prefix=config.schema_prefix
    ):
        if feature_set == "matrix":
            return cls.fetch_matrix_X(split_id, schema_prefix)
        if feature_set == "image":
            return cls.fetch_image_X(split_id)
        if feature_set == "blocklots":
            return cls.fetch_blocklot_X(split_id)
        raise Exception("Unknown feature group")

    @classmethod
    def fetch_matrix_X(cls, split_id, schema_prefix=config.schema_prefix):
        row = run_query(
            f"""SELECT * FROM {schema_prefix}_model_prep.feature_matrices
            WHERE from_split = %s""",
            (split_id,),
        )
        return pd.read_pickle(row[0]["filepath"])

    @classmethod
    def fetch_feature_names(cls, split_id):
        features = run_query(
            f"""SELECT * FROM {config.schema_prefix}_model_prep.features
            WHERE split_kind_id = %s""",
            (split_id,),
        )
        columns = list(features[0].keys())
        columns.remove("blocklot")
        columns.remove("split_kind_id")
        return columns

    @staticmethod
    def fetch_y(split_id):
        results = run_query(
            f"""
            SELECT * FROM {config.schema_prefix}_model_prep.labels AS l
            JOIN {config.schema_prefix}_model_prep.splits AS s
                ON l.blocklot = s.blocklot
            WHERE s.split_kind_id = %s""",
            (split_id,),
        )
        return {row["blocklot"]: row["label"] for row in results}

    @classmethod
    def fetch_ordered_y(cls, split_id, blocklots):
        blocklot_to_label = cls.fetch_y(split_id)
        return [blocklot_to_label[b] for b in blocklots]

    @staticmethod
    def merge_X_y(X: pd.DataFrame, y: dict[str, int]) -> pd.DataFrame:
        return X.merge(
            pd.Series(y, name="label"), how="left", left_index=True, right_index=True
        )

    @staticmethod
    def flatten_X_y(X: pd.DataFrame, y: dict[str, int]) -> tuple[list, list]:
        flat_X, flat_y = [], []
        for blocklot, features in X.iterrows():
            flat_X.append(features.values)
            flat_y.append(y.get(blocklot, None))
        return flat_X, flat_y

    @staticmethod
    def flatten_image_X_y(
        X: dict[str, np.ndarray], y: dict[str, int]
    ) -> tuple[list, list]:
        flat_X, flat_y = [], []
        for blocklot, features in X.items():
            flat_X.append(features)
            flat_y.append(y.get(blocklot, None))
        return flat_X, flat_y

    @staticmethod
    def fetch_image_X(split_id, schema_prefix=config.schema_prefix):
        blocklots = DatasetSplitter.blocklots_for_split(split_id, schema_prefix)
        X = {}
        for blocklot in tqdm(blocklots, desc="Loading images", smoothing=0):
            X[blocklot] = fetch_image(blocklot)
        return X

    @staticmethod
    def fetch_blocklot_X(split_id):
        return DatasetSplitter.blocklots_for_split(split_id)

    def fetch_X_y(self, split_id, feature_set="matrix"):
        m = hashlib.sha256()
        m.update(str(split_id).encode("utf-8"))
        m.update(feature_set.encode("utf-8"))
        hashed = m.hexdigest()

        #        loaded = self.load_X_y_from_cache(hashed)
        #        if loaded is not None:
        #            return loaded

        if feature_set == "matrix":
            output = self.flatten_X_y(self.fetch_X(split_id), self.fetch_y(split_id))
        if feature_set == "image":
            output = self.flatten_image_X_y(
                self.fetch_image_X(split_id), self.fetch_y(split_id)
            )
        if feature_set == "blocklots":
            blocklots = self.fetch_blocklot_X(split_id)
            output = (blocklots, self.fetch_ordered_y(split_id, blocklots))

        self.save_X_y_to_cache(output, hashed)
        return output

    def load_X_y_from_cache(self, hash):
        if not self.use_cache:
            return
        path = self.cache_filename_for_hashed(hash)
        if path.is_file():
            return load(path)

    def cache_filename_for_hashed(self, hash):
        directory = Path(config.model_trainer.X_y_cache_dir) / config.schema_prefix
        directory.mkdir(parents=True, exist_ok=True)
        filename = f"{hash}.df.pkl"
        return directory / filename

    def save_X_y_to_cache(self, matrix, hash):
        if not self.use_cache:
            return
        path = self.cache_filename_for_hashed(hash)
        dump(matrix, path)

    @staticmethod
    def models_in_group(group_id, schema_prefix=config.schema_prefix):
        models = run_query(
            f"""
            SELECT * FROM {schema_prefix}_model_results.models
            WHERE model_group = %s""",
            (group_id,),
        )
        return {m["id"]: m for m in models}

    @staticmethod
    def model_group_ids():
        models = run_query(
            f"""
            SELECT DISTINCT(model_group) AS id
            FROM {config.schema_prefix}_model_results.models
            """
        )
        return [row["id"] for row in models]

    @staticmethod
    def filter_to_labeled(X, y):
        new_X, new_y = [], []
        for i, label in enumerate(y):
            if label is not None:
                new_X.append(X[i]), new_y.append(label)
        return new_X, new_y

    def models_to_train(self):
        return {
            model_class: self.calc_param_space(config.params)
            for model_class, config in self.model_classes.items()
        }

    def train_model_on_split(self, model_class, hyperparams, split_id):
        feature_set = self.model_class_to_feature_set[model_class]
        X, y = self.filter_to_labeled(*self.fetch_X_y(split_id, feature_set))
        model = SLUG_TO_MODEL[model_class](**hyperparams)
        model.fit(X, y)
        return self.save_model(model_class, model, hyperparams, split_id, feature_set)

    def train(self, split_ids: Sequence[int]):
        model_params = self.models_to_train()
        pbar = tqdm(
            total=len(split_ids)
            * len([v for params in model_params.values() for v in params]),
            desc="Training",
        )
        for split_id in split_ids:
            for model_class, param_space in model_params.items():
                feature_set = self.model_class_to_feature_set[model_class]
                X, y = self.filter_to_labeled(*self.fetch_X_y(split_id, feature_set))
                for params in param_space:
                    model = SLUG_TO_MODEL[model_class](**params)
                    model.fit(X, y)
                    self.save_model(model_class, model, params, split_id, feature_set)
                    pbar.update()
        pbar.close()

    def write_model_to_disk(self, model_id, model, params, split_id, feature_set):
        model_path = Path(self.model_dir) / config.schema_prefix / f"{model_id}.pkl"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        with open(model_path, "wb") as f:
            dump(
                {
                    "class": model.__class__,
                    "model": model.to_save() if hasattr(model, "to_save") else model,
                    "params": params,
                    "split_id": split_id,
                    "feature_set": feature_set,
                },
                f,
            )
        return model_path

    def write_model_to_db(
        self, model_id, model_class, params, split_id, feature_set, model_path
    ):
        self._create_table()
        run(
            f"""
            INSERT INTO {config.schema_prefix}_model_results.models
                (id, model_class, trained_on_split_id,
                 trained_on_feature_set, hyperparams, filepath)
                VALUES (%s, %s, %s, %s, %s, %s)""",
            (model_id, model_class, split_id, feature_set, Json(params), model_path),
        )

    @staticmethod
    def model_ids():
        return [
            row["id"]
            for row in run_query(
                f"SELECT id FROM {config.schema_prefix}_model_results.models"
            )
        ]

    @staticmethod
    def load_models():
        return {
            row["id"]: row
            for row in run_query(
                f"SELECT * FROM {config.schema_prefix}_model_results.models"
            )
        }

    def save_model(self, model_class, model, params, split_id, feature_set):
        model_id = str(uuid.uuid4())
        path = self.write_model_to_disk(model_id, model, params, split_id, feature_set)
        self.write_model_to_db(
            model_id, model_class, params, split_id, feature_set, str(path)
        )
        return model_id

    @staticmethod
    def load_model_details(model_id, schema_prefix=config.schema_prefix):
        filepath = Path(
            run_query(
                f"""SELECT filepath FROM {schema_prefix}_model_results.models
            WHERE id = %s""",
                (model_id,),
            )[0]["filepath"]
        )

        # Models might have moved paths if from a different machine
        if not filepath.exists():
            filepath = Path(config.model_trainer.model_dir) / Path(*filepath.parts[-2:])

        with open(filepath, "rb") as f:
            model_details = load(f)
        return model_details

    @staticmethod
    def model_group_for_model_id(model_id):
        pass

    @staticmethod
    def schema_prefix_for_model_group_id(model_group_id):
        results = run_query(
            """
            SELECT table_schema
            FROM information_schema."columns" c
            WHERE table_name = 'group_evaluations'
                AND column_name = 'model_group_id'
        """
        )
        for row in results:
            schema_results = run_query(
                f"""
                SELECT COUNT(*) AS n_rows
                FROM {row['table_schema']}.group_evaluations
                WHERE model_group_id = %s
            """,
                (model_group_id,),
            )
            if schema_results[0]["n_rows"] > 0:
                return row["table_schema"].replace("_model_results", "")

    @staticmethod
    def schema_prefix_for_model_id(model_id):
        results = run_query(
            """
            SELECT
                REPLACE(table_schema,'_model_predictions', '') AS schema_prefix
            FROM information_schema."tables" t
            WHERE table_name = %s
        """,
            (f"preds_of_{model_id}",),
        )
        return results[0]["schema_prefix"]


if __name__ == "__main__":
    trainer = ModelTrainer(config.model_trainer)
    trainer.train([1])
