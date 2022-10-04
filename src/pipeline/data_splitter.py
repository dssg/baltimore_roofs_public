from collections import defaultdict
import logging

from munch import Munch
from sklearn.model_selection import KFold
from tqdm.auto import tqdm

from src.config import experiment_config as config
from src.db import run, run_query, batch_insert

logger = logging.getLogger(__name__)


class DatasetSplitter:
    """Split the dataset into training and validation sets."""

    def __init__(self, config: Munch, cohort_query: str, seed: int):
        """Initialize a new DatasetSplitter.

        Args:
            config (Munch): The splitter configuration.
            cohort_query (str): SQL query that selects the cohort from the database.
            seed (int): Seed for random things.
        """
        self.kfold = KFold(
            n_splits=config.kfold.n_folds, shuffle=True, random_state=seed
        )
        self.splits = config.splits if config.splits is not None else {}
        self.cohort_query = cohort_query

    def _fetch_nontrain_blocklots(self, train_blocklots):
        query = f"{self.cohort_query} AND blocklot NOT IN %s"
        return self._blocklots_from_results(run_query(query, (tuple(train_blocklots),)))

    @classmethod
    def cohort_blocklots(cls):
        return cls._blocklots_from_results(run_query(config.cohort.query))

    @staticmethod
    def _blocklots_from_results(results):
        return list(set([row[0] for row in results]))

    @staticmethod
    def validation_split_for_train_split(
        train_split_id, schema_prefix=config.schema_prefix
    ):
        return run_query(
            f"""
                WITH train_split AS (
                    SELECT * FROM {schema_prefix}_model_prep.split_kinds
                    WHERE id = %s
                )
                SELECT val.* FROM {schema_prefix}_model_prep.split_kinds val
                JOIN train_split ON train_split.kind = val.kind
                AND val.split = 'validation'
            """,
            (train_split_id,),
        )[0]["id"]

    def _pull_kfold_splits(self, pbar=None):
        splits = {}
        cohort_blocklots = self._blocklots_from_results(run_query(self.cohort_query))
        for i, (train_idx, val_idx) in enumerate(
            self.kfold.split([[c] for c in cohort_blocklots])
        ):
            splits[f"fold_{i}_of_{self.kfold.n_splits}"] = (
                [cohort_blocklots[i] for i in train_idx],
                [cohort_blocklots[i] for i in val_idx],
            )
            if pbar:
                pbar.update()
        return splits

    @staticmethod
    def _create_tables():
        run(
            f"""
            CREATE SCHEMA IF NOT EXISTS {config.schema_prefix}_model_prep
            AUTHORIZATION "baltimore-roofs-role"
        """
        )
        run(
            f"""
            CREATE TABLE IF NOT EXISTS {config.schema_prefix}_model_prep.split_kinds (
                id integer PRIMARY KEY GENERATED ALWAYS AS IDENTITY,
                kind varchar(50),
                split varchar(20),
                UNIQUE (kind, split)
            )
        """
        )
        run(
            f"""
            CREATE TABLE IF NOT EXISTS {config.schema_prefix}_model_prep.splits (
                split_kind_id integer
                    REFERENCES {config.schema_prefix}_model_prep.split_kinds (id),
                blocklot varchar(10)
            )
        """
        )

    def pull_splits(self) -> dict[str, tuple[list[str], list[str]]]:
        """Pull splits from the database.

        Returns:
            dict[str, tuple[list[str], list[str]]]: Name of the split to lists of
                 training and validation blocklots.
        """
        splits = {}
        pbar = tqdm(total=self.kfold.n_splits + len(self.splits), desc="Splitting")
        for name, details in self.splits.items():
            splits[name] = self.pull_split(details.train, details.validate)
            pbar.update()
        splits.update(self._pull_kfold_splits(pbar))
        pbar.close()
        return splits

    def pull_split(self, train: str, validate: str) -> tuple[list[str], list[str]]:
        """Pull a split from the database.

        Args:
            train (str): SQL query for the training set.
            validate (str): SQL query or "rest" for the validation set.
                Passing "rest" means the remaining cohort that's not
                in the training set.

        Returns:
            tuple[list[str], list[str]]: Lists of the training and validation blocklots.
        """
        train_query = " AND ".join([self.cohort_query, train])
        train_blocklots = self._blocklots_from_results(run_query(train_query))
        if validate.strip().lower() == "rest":
            validate_blocklots = self._fetch_nontrain_blocklots(train_blocklots)
        else:
            val_query = " AND ".join([self.cohort_query, validate])
            validate_blocklots = self._blocklots_from_results(run_query(val_query))
        return train_blocklots, validate_blocklots

    @classmethod
    def blocklots_for_split(cls, split_id, schema_prefix=config.schema_prefix):
        return cls._blocklots_from_results(
            run_query(
                f"""SELECT blocklot FROM {schema_prefix}_model_prep.splits
                WHERE split_kind_id = %s""",
                (split_id,),
            )
        )

    @staticmethod
    def split_ids():
        return [
            row["id"]
            for row in run_query(
                f"SELECT id FROM {config.schema_prefix}_model_prep.split_kinds"
            )
        ]

    @staticmethod
    def get_splits():
        splits = defaultdict(dict)
        results = run_query(
            f"SELECT * FROM {config.schema_prefix}_model_prep.split_kinds"
        )
        for row in results:
            splits[row["kind"]][row["split"]] = row["id"]
        return splits

    @classmethod
    def get_training_splits(cls):
        return [split["train"] for split in cls.get_splits().values()]

    @classmethod
    def blocklots_for_all_splits(cls):
        return cls._blocklots_from_results(
            run_query(f"SELECT blocklot FROM {config.schema_prefix}_model_prep.splits")
        )

    @classmethod
    def _insert_to_split_kind_table(cls, kind, split):
        cls._create_tables()
        return run_query(
            f"""
                INSERT INTO
                    {config.schema_prefix}_model_prep.split_kinds (kind, split)
                    VALUES (%s, %s)
                ON CONFLICT (kind, split) DO UPDATE SET kind = %s
                RETURNING id
            """,
            (kind, split, kind),
        )[0][0]

    def write_splits_to_db(self) -> str:
        """Calculate splits and write to the database.

        Returns:
            str: The schema and table name of the splits table.
        """
        self._create_tables()
        rows = []
        for name, (train, val) in self.pull_splits().items():
            train_split_id = self._insert_to_split_kind_table(name, "train")
            val_split_id = self._insert_to_split_kind_table(name, "validation")
            for blocklot in train:
                rows.append((train_split_id, blocklot))
            for blocklot in val:
                rows.append((val_split_id, blocklot))
        batch_insert(
            f"""INSERT INTO {config.schema_prefix}_model_prep.splits
            (split_kind_id, blocklot) VALUES %s""",
            rows,
        )
        return "model_prep.splits"


if __name__ == "__main__":
    s = DatasetSplitter(config.splitter, config.cohort.query, config.random_seed)
    s.write_splits_to_db()
