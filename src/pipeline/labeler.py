import logging

from src.config import experiment_config as config
from src.db import batch_insert, run_query, run

logger = logging.getLogger(__name__)


class Labeler:
    def __init__(self, config):
        self.query = config.query

    def label(self, blocklots: list[str]):
        results = run_query(self.query, (tuple(blocklots),))
        return {row["blocklot"]: row["label"] for row in results}

    def _create_table(self):
        run(
            f"""
            CREATE SCHEMA IF NOT EXISTS {config.schema_prefix}_model_prep
            AUTHORIZATION "baltimore-roofs-role"
        """
        )
        run(
            f"""
            CREATE TABLE IF NOT EXISTS {config.schema_prefix}_model_prep.labels (
                blocklot varchar(10),
                label integer
            )
        """
        )
        run(
            f"""
            CREATE UNIQUE INDEX IF NOT EXISTS blocklot_idx
            ON {config.schema_prefix}_model_prep.labels (blocklot)
        """
        )
        pass

    def write_labels_to_db(self, blocklots):
        self._create_table()
        rows = []
        for blocklot, label in self.label(blocklots).items():
            rows.append((blocklot, label))
        batch_insert(
            f"""INSERT INTO {config.schema_prefix}_model_prep.labels
                (blocklot, label) VALUES %s""",
            rows,
        )
        return "model_prep.labels"

    @staticmethod
    def load_labels(blocklots):
        results = run_query(
            f"""SELECT blocklot, label FROM {config.schema_prefix}_model_prep.labels
            WHERE blocklot IN %s""",
            (tuple(blocklots),),
        )
        return {row["blocklot"]: row["label"] for row in results}


if __name__ == "__main__":
    results = run_query(
        f"""
        SELECT blocklot FROM {config.schema_prefix}_model_prep.splits
        """
    )
    blocklots = [row[0] for row in results]
    Labeler(config.labeler).write_labels_to_db(blocklots)
