import logging
import pandas as pd

from src.config import experiment_config as config
from src.db import run_query
from src.pipeline.reporter import Reporter


class ListCreator:
    def __init__(self, this_config=config.list_creator):
        if this_config:
            self.schema_prefix = this_config.model.schema_prefix
            self.output_path = this_config.output_path

    def fetch_model_ids(self):
        results = run_query(
            """
                SELECT
                    REPLACE(table_name, 'preds_of_', '') AS model_id
                FROM information_schema."tables"
                WHERE table_name LIKE 'preds_of_%%'
                AND table_schema = %s
            """,
            (f"{self.schema_prefix}_model_predictions",),
        )
        return [r["model_id"] for r in results]

    def get_prediction_query(self):
        return "\nUNION ALL\n".join(
            [
                f'''SELECT * FROM
                {self.schema_prefix}_model_predictions."preds_of_{model_id}"'''
                for model_id in self.fetch_model_ids()
            ]
        )

    def fetch_aggregated_preds(self):
        results = run_query(
            f"""
            SELECT
                t.blocklot,
                MIN(score) AS min_score,
                MAX(score) AS max_score,
                AVG(score) AS mean_score,
                STDDEV_POP(score) AS std_score
            FROM ({self.get_prediction_query()}) AS t
            GROUP BY t.blocklot"""
        )
        return pd.DataFrame(results, columns=results[0].keys()).set_index("blocklot")

    def create(self):
        preds = self.fetch_aggregated_preds()
        preds["codemap"] = pd.Series(Reporter.codemap_urls(preds.index))
        preds["codemap_ext"] = pd.Series(Reporter.codemap_ext_urls(preds.index))
        preds["pictometry"] = pd.Series(Reporter.pictometry_urls(preds.index))
        return preds

    def write_scores(self):
        results = self.create()
        logging.info("Writing to %s", self.output_path)
        results.to_csv(self.output_path)


if __name__ == "__main__":
    creator = ListCreator()
    creator.write_scores()
