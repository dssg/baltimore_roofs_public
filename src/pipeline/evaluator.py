from collections import namedtuple
import logging

import numpy as np
import pandas as pd

from src.config import experiment_config as config
from src.db import run, run_query
from src.pipeline.data_splitter import DatasetSplitter
from src.pipeline.labeler import Labeler
from src.pipeline.model_trainer import ModelTrainer
from src.pipeline.predictor import Predictor

logger = logging.getLogger(__name__)

EvalCounts = namedtuple(
    "EvalCounts",
    ("n", "n_labeled", "n_labeled_pos", "n_labeled_neg", "precision", "recall"),
)

Column = namedtuple("Column", ("name", "type", "ref", "metric"))


class Evaluator:
    def __init__(self, config):
        self.thresholds = np.linspace(
            config.thresholds.start, config.thresholds.stop, config.thresholds.n_steps
        )
        self.top_k = config.top_k

    def _pull_eval_counts(self, df, n_total_positive):
        label_counts = df.label.value_counts()
        precision = None
        recall = None
        if label_counts.sum() > 0:
            precision = label_counts.get(1.0, default=0) / label_counts.sum()
        if n_total_positive > 0:
            recall = label_counts.get(1.0, default=0) / n_total_positive
        return EvalCounts(
            n=int(df.shape[0]),
            n_labeled=int(label_counts.sum()),
            n_labeled_pos=int(label_counts.get(1.0, default=0)),
            n_labeled_neg=int(label_counts.get(0.0, default=0)),
            precision=precision,
            recall=recall,
        )

    def build_evaluation(self, df, thresholds=None, ks=None):
        if thresholds is None:
            thresholds = self.thresholds
        if ks is None:
            ks = [self.top_k]

        df = df[~df.score.isna()].copy()
        n_total_positive = df.label.sum()
        df["threshold"] = df.score.rank(ascending=False, pct=True)
        threshold_counts = {}
        for threshold in thresholds:
            to_eval = df[df.threshold < threshold]
            threshold_counts[threshold] = self._pull_eval_counts(
                to_eval, n_total_positive
            )

        top_k_counts = {}
        for k in ks:
            top_k = df.sort_values("score", ascending=False).head(k)
            top_k_counts[k] = self._pull_eval_counts(top_k, n_total_positive)

        return threshold_counts, top_k_counts

    @staticmethod
    def build_score_df(preds):
        blocklots = list(preds.keys())
        labels = Labeler.load_labels(blocklots)
        return pd.DataFrame(
            {"label": {b: l for b, l in labels.items()}, "score": preds}
        )

    @classmethod
    def model_group_scores(cls, group_id, schema_prefix=config.schema_prefix):
        models = ModelTrainer.models_in_group(group_id, schema_prefix)
        preds = {}
        for model_id, model in models.items():
            val_split = DatasetSplitter.validation_split_for_train_split(
                model["trained_on_split_id"], schema_prefix
            )
            preds.update(Predictor.load_preds(model_id, val_split, schema_prefix))
        return cls.build_score_df(preds)

    def evaluate_model_group(self, group_id, schema_prefix=config.schema_prefix):
        df = self.model_group_scores(group_id, schema_prefix)
        threshold_counts, top_k_counts = self.build_evaluation(df)
        top_k_counts = top_k_counts[self.top_k]
        return threshold_counts, top_k_counts

    def evaluate(self, model_id, split_id):
        preds = Predictor.load_preds(model_id, split_id)
        df = self.build_score_df(preds)
        threshold_counts, top_k_counts = self.build_evaluation(df)
        top_k_counts = top_k_counts[self.top_k]
        return threshold_counts, top_k_counts

    def write_evaluation_to_db(self, model_id, split_id):
        self._create_tables()
        threshold_counts, top_k_counts = self.evaluate(model_id, split_id)
        columns = self._get_columns()

        values = [model_id, split_id]
        for column in columns:
            if column.name.startswith("threshold_"):
                threshold = column.ref
                values.append(getattr(threshold_counts[threshold], column.metric))
            elif column.name.startswith("top_"):
                values.append(getattr(top_k_counts, column.metric))
        column_names = ",".join(c.name for c in columns)
        run(
            f"""
            INSERT INTO {config.schema_prefix}_model_results.evaluations
                (model_id, split_id,
                {column_names})
            VALUES (%s, %s, {','.join(['%s'] * len(columns))})
        """,
            values,
        )

    def write_group_evaluation_to_db(self, group_id):
        self._create_tables()
        threshold_counts, top_k_counts = self.evaluate_model_group(group_id)
        columns = self._get_columns()

        values = [group_id]
        for column in columns:
            if column.name.startswith("threshold_"):
                threshold = column.ref
                values.append(getattr(threshold_counts[threshold], column.metric))
            elif column.name.startswith("top_"):
                values.append(getattr(top_k_counts, column.metric))
        column_names = ",".join(c.name for c in columns)
        run(
            f"""
            INSERT INTO {config.schema_prefix}_model_results.group_evaluations
                (model_group_id,
                {column_names})
            VALUES (%s, {','.join(['%s'] * len(columns))})
        """,
            values,
        )

    @staticmethod
    def _name_threshold(threshold):
        return str(threshold).replace(".", "_")

    def _get_columns(self):
        columns = []
        for threshold in self.thresholds:
            threshold_name = self._name_threshold(threshold)
            columns += [
                Column(f"threshold_{threshold_name}_n", "integer", threshold, "n"),
                Column(
                    f"threshold_{threshold_name}_n_labeled",
                    "integer",
                    threshold,
                    "n_labeled",
                ),
                Column(
                    f"threshold_{threshold_name}_n_labeled_pos",
                    "integer",
                    threshold,
                    "n_labeled_pos",
                ),
                Column(
                    f"threshold_{threshold_name}_n_labeled_neg",
                    "integer",
                    threshold,
                    "n_labeled_neg",
                ),
                Column(
                    f"threshold_{threshold_name}_precision",
                    "real",
                    threshold,
                    "precision",
                ),
                Column(
                    f"threshold_{threshold_name}_recall",
                    "real",
                    threshold,
                    "recall",
                ),
            ]
        columns += [
            Column(f"top_{self.top_k}_n", "integer", None, "n"),
            Column(f"top_{self.top_k}_n_labeled", "integer", None, "n_labeled"),
            Column(f"top_{self.top_k}_n_labeled_pos", "integer", None, "n_labeled_pos"),
            Column(f"top_{self.top_k}_n_labeled_neg", "integer", None, "n_labeled_neg"),
            Column(f"top_{self.top_k}_precision", "real", None, "precision"),
            Column(f"top_{self.top_k}_recall", "real", None, "recall"),
        ]
        return columns

    def _create_tables(self):
        run(
            f"""
            CREATE SCHEMA IF NOT EXISTS {config.schema_prefix}_model_results
            AUTHORIZATION "baltimore-roofs-role"
        """
        )
        column_defs = ",".join([f"{c.name} {c.type}" for c in self._get_columns()])
        run(
            f"""
            CREATE TABLE IF NOT EXISTS
                {config.schema_prefix}_model_results.evaluations (
                model_id varchar(36)
                    REFERENCES {config.schema_prefix}_model_results.models (id),
                split_id integer
                    REFERENCES {config.schema_prefix}_model_prep.split_kinds (id),
                {column_defs},
                UNIQUE (model_id, split_id)
            )
        """
        )
        run(
            f"""
            CREATE TABLE IF NOT EXISTS
                {config.schema_prefix}_model_results.group_evaluations (
                model_group_id varchar(40),
                {column_defs},
                UNIQUE (model_group_id)
            )
        """
        )

    @classmethod
    def evals_exist(cls, model_id, split_id):
        cls._create_tables()
        results = run_query(
            f"""
            SELECT COUNT(*) FROM {config.schema_prefix}_model_results.evaluations
            WHERE model_id = %s
            AND split_id = %s
            """,
            (model_id, split_id),
        )
        return results[0][0] > 0

    def load_group_eval(self, group_id, schema_prefix=config.schema_prefix):
        row = run_query(
            f"""
            SELECT * FROM {schema_prefix}_model_results.group_evaluations
            WHERE model_group_id = %s""",
            (group_id,),
        )[0]
        return pd.merge(
            pd.DataFrame(self._get_columns()),
            pd.Series(row, index=row.keys(), name="value"),
            left_on="name",
            right_index=True,
            how="right",
        ).set_index("name")

    def load_eval(self, model_id, split_id):
        row = run_query(
            f"""
            SELECT * FROM {config.schema_prefix}_model_results.evaluations
            WHERE model_id = %s AND split_id = %s""",
            (model_id, split_id),
        )[0]
        return pd.merge(
            pd.DataFrame(self._get_columns()),
            pd.Series(row, index=row.keys(), name="value"),
            left_on="name",
            right_index=True,
            how="right",
        ).set_index("name")


if __name__ == "__main__":
    evaluator = Evaluator(config.evaluator)
    evaluator.write_evaluation_to_db("6b2f9c40-d6f1-4651-9f89-b2f1af5987f3", 1)
