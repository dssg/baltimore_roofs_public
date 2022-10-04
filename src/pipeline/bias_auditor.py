from ast import Str
from lib2to3.pgen2.pgen import DFAState
import click
import logging


import pandas as pd
import seaborn as sns
from aequitas.group import Group
from aequitas.bias import Bias
from aequitas.fairness import Fairness
from aequitas.plotting import Plot
import ohio.ext.pandas
from sklearn.neighbors import KernelDensity  # noqa: F401
from sqlalchemy import create_engine

from src import db
from src.config import experiment_config as config
from src.data import process_tables
from src.data import import_tables
from src.db import run_query, run
from src.pipeline.reporter import Evaluator
from src.pipeline.reporter import Reporter

logger = logging.getLogger(__name__)
LABELS_AND_SCORES_TABLE_NAME = "labels_and_scores"
UNPROCESSED_TABLE_NAME = "unprocessed"


class BiasAuditor:
    def __init__(self, config, model_schema_prefix, model_group_id):
        self.model_schema_prefix = model_schema_prefix
        self.model_group_id = model_group_id
        self.schema = f"aequitas_test_cwl_{model_schema_prefix}_model"
        self.engine = create_engine(db.connection_string())
        self.n_quantiles = config.n_quantiles

    def _write_labels_and_scores_to_db(self):
        """ """
        r = Reporter()
        df = r.evaluator.model_group_scores(
            self.model_group_id,
            schema_prefix=self.model_schema_prefix,
        )
        run(
            f"""
            CREATE SCHEMA IF NOT EXISTS {self.schema}
            AUTHORIZATION "baltimore-roofs-role"
        """
        )
        run(
            f"""
            DROP TABLE IF EXISTS {self.schema}.{LABELS_AND_SCORES_TABLE_NAME} cascade
            """
        )
        df.pg_copy_to(LABELS_AND_SCORES_TABLE_NAME, self.engine, schema=self.schema)

    def _load_unprocessed(self):
        run(
            f"""
            set role "chaewonl";
            drop table if exists {self.schema}.{UNPROCESSED_TABLE_NAME} cascade;
            create table {self.schema}.{UNPROCESSED_TABLE_NAME} as (
            select
                    mr.index as entity_id,
                    mr.label as label_value,
                    mr.score,
                    cd.race_black_or_african_american,
                    cd.race_count_total,
                    cd.hh_med_income,
                    cd.tenure_count_owner,
                    cd.tenure_count_renter,
                    cd.tenure_count_total,
                    red.class as redliningclass
            from
                    {self.schema}.{LABELS_AND_SCORES_TABLE_NAME} as mr
            left join raw.tax_parcel_address as tp
                        on
                    mr.index = tp.blocklot
            left join census.blockgroup_shapefiles_2020 as cbs
                        on
                    ST_Within(tp.wkb_geometry,
                    cbs.geometry)
            left join raw.redlining as red
                        on
                    ST_Intersects(cbs.geometry,
                red.shape)
            left join census.demographics_2020 as cd
                        on
                    cd.index = cbs."GEOID"
            );
            """
        )

    def _convert_to_categorical_var(self, df):
        # TODO: This code is specific for race, income, redline, renter - update for other.
        fractions = {
            "race_black_or_african_american": "race_count_total",
            "tenure_count_renter": "tenure_count_total",
            "tenure_count_owner": "tenure_count_total",
        }

        for numerator in fractions.keys():
            denominator = fractions[numerator]
            df[numerator] = df[numerator] / df[denominator]

        df.drop(list(fractions.values()), axis=1, inplace=True)
        categorical_vars = ["hh_med_income"] + list(fractions.keys())

        for var in categorical_vars:
            labels = [f"{var} Q{i}" for i in range(1, self.n_quantiles + 1)]
            df[f"{var}_quantile"] = pd.qcut(df[var], self.n_quantiles, labels).astype(
                str
            )
            df.drop([var], axis=1, inplace=True)

    def get_aequitas_input(self):
        self._write_labels_and_scores_to_db()
        self._load_unprocessed()
        table = f"{self.schema}.{UNPROCESSED_TABLE_NAME}"
        data = run_query(
            f"""
                SELECT *
                FROM
                    {table}
            """
        )
        df = pd.DataFrame(data, columns=data[0].keys())
        self._convert_to_categorical_var(df)
        return df


@click.command()
@click.option("--model_schema_prefix", default="jdcc_third_run", type=str)
@click.option(
    "--model_group_id", default="93671b3fb4b50c91cc18d64fa38071a7e90e2db6", type=str
)
def run_bias_audit(model_schema_prefix, model_group_id):
    ba = BiasAuditor(
        config.bias_auditor,
        model_schema_prefix,
        model_group_id,
    )
    return ba.get_aequitas_input()


if __name__ == "__main__":
    df = run_bias_audit()
    print(df.head(5))
