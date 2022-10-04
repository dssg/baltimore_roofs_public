from collections import defaultdict
from datetime import timedelta
import hashlib
from pathlib import Path

import pandas as pd
from tqdm.auto import tqdm

from src.config import experiment_config as config
from src.data.image_parser import fetch_image
from src.db import run_query, run
from src.models.image_baseline import DarkImageBaseline
from src.pipeline.evaluator import Evaluator
from src.pipeline.predictor import Predictor


class MatrixCreator:
    def __init__(self, config, splits_table):
        self.radii_of_311 = config.features.radii_of_311
        self.inspection_note_words = config.features.inspection_note_words
        self.max_date = config.max_date
        self.dark_pixel_thresholds = config.features.dark_pixel_thresholds
        self.transfer_learned_score = config.features.transfer_learned_score
        self.splits_table = splits_table
        self.disabled_features = set(config.disabled_features)
        self.use_cache = config.use_cache
        self.dark_cache = defaultdict(dict)
        self.year_built_imputed = config.features.get("year_built_imputed", None)

    @staticmethod
    def _split_to_id(kind, name):
        return run_query(
            f"""SELECT id FROM {config.schema_prefix}_model_prep.split_kinds
            WHERE kind = %s AND split = %s""",
            (kind, name),
        )[0][0]

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
            CREATE TABLE IF NOT EXISTS
                {config.schema_prefix}_model_prep.feature_groups (
                id integer PRIMARY KEY GENERATED ALWAYS AS IDENTITY,
                name integer
            )
        """
        )
        run(
            f"""
            CREATE TABLE IF NOT EXISTS
                {config.schema_prefix}_model_prep.feature_matrices (
                id integer PRIMARY KEY GENERATED ALWAYS AS IDENTITY,
                from_split integer
                REFERENCES {config.schema_prefix}_model_prep.split_kinds (id) UNIQUE,
                filepath varchar(300)
            )
        """
        )

    @staticmethod
    def save_matrix_to_disk(matrix, id):
        filename = f"{id}.pkl"
        directory = Path(config.matrix_creator.matrix_dir) / config.schema_prefix
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / filename
        matrix.to_pickle(path)
        return path

    @classmethod
    def save_matrix_to_db(cls, matrix, split_id):
        cls._create_tables()
        results = run_query(
            f"""
            INSERT INTO {config.schema_prefix}_model_prep.feature_matrices (from_split)
            VALUES (%s) RETURNING id
            """,
            (split_id,),
        )
        return results[0]["id"]

    @staticmethod
    def save_filepath(matrix_id, filepath):
        run(
            f"""UPDATE {config.schema_prefix}_model_prep.feature_matrices
            SET filepath = %s WHERE id = %s""",
            (str(filepath), matrix_id),
        )

    @classmethod
    def save_matrix(cls, matrix, split_id):
        id = cls.save_matrix_to_db(matrix, split_id)
        filepath = cls.save_matrix_to_disk(matrix, id)
        cls.save_filepath(id, filepath)

    def _splits(self):
        return run_query(f"SELECT kind, split, id FROM {self.splits_table}")

    @staticmethod
    def _blocklots_for_split(split_id):
        results = run_query(
            f"""
            SELECT DISTINCT(blocklot) FROM {config.schema_prefix}_model_prep.splits
            WHERE split_kind_id = %s
        """,
            (split_id,),
        )
        return [row["blocklot"] for row in results]

    def _build_311_radius(self, blocklots, radius, max_date):
        query = """
            SELECT
                blocklot,
                COUNT(DISTINCT service_request_number) AS n_calls
            FROM processed.tax_parcel_address AS tp
            LEFT JOIN processed.data_311 AS d
                ON ST_DWithin(d.shape, tp.wkb_geometry, %s)
                AND d.longitude IS NOT null
                AND created_date <= %s
            WHERE blocklot IN %s
            GROUP BY blocklot
        """
        key = f"calls_to_311_{radius}ft"
        output = {
            key: {
                row["blocklot"]: row["n_calls"]
                for row in run_query(query, (radius, max_date, tuple(blocklots)))
            }
        }
        for blocklot in blocklots:
            if blocklot not in output[key]:
                output[key][blocklot] = 0
        return output

    def _build_311_type_features(self, blocklots, max_date):
        types = [
            "HCD-Illegal Dumping",
            "HCD-Vacant Building",
            "HCD-Maintenance Structure",
            "HCD-Rodents",
            "HCD-Trees and Shrubs",
            "HCD-Abandoned Vehicle",
            "HCD-CCE Building Permit Complaint",
        ]
        features = {}
        for radius in self.radii_of_311:
            for type in types:
                results = run_query(
                    """
                    SELECT
                        blocklot,
                        COUNT(DISTINCT service_request_number) AS n_calls
                    FROM processed.tax_parcel_address AS tp
                    LEFT JOIN processed.data_311 AS d
                        ON ST_DWithin(d.shape, tp.wkb_geometry, %s)
                        AND d.longitude IS NOT NULL
                        AND created_date <= %s
                        AND sr_type = %s
                    WHERE blocklot IN %s
                    GROUP BY blocklot
                """,
                    (radius, max_date, type, tuple(blocklots)),
                )
                type_name = type.replace("HCD-", "").lower().replace(" ", "_")
                key = f"calls_to_311_for_{type_name}_{radius}ft"
                features[key] = {row["blocklot"]: row["n_calls"] for row in results}
        return features

    @classmethod
    def _median_year_built(cls, split_kind, split_id):
        # split_id = cls._split_to_id(split_kind, split_name)
        query = f"""
            SELECT percentile_disc(0.5) WITHIN GROUP (ORDER BY year_build) AS year
            FROM processed.tax_parcel_address tpa2
            JOIN {config.schema_prefix}_model_prep.splits sp
                ON tpa2.blocklot = sp.blocklot
            WHERE sp.split_kind_id = %s
        """
        return run_query(query, (split_id,))[0]["year"]

    def _build_year_built(self, blocklots, median_year_built, max_date):
        results = run_query(
            """
            SELECT
                blocklot,
                CASE
                    WHEN year_build = 0 THEN %s
                    WHEN year_build > extract('year' FROM %s) THEN %s
                    ELSE year_build
                END::int AS year_built,
                (year_build = 0)
                    OR (year_build > extract('year' FROM %s)) AS year_built_unknown
            FROM processed.tax_parcel_address tpa
            WHERE blocklot IN %s
        """,
            (
                median_year_built,
                max_date,
                median_year_built,
                max_date,
                tuple(blocklots),
            ),
        )
        return {
            "year_built": {row["blocklot"]: row["year_built"] for row in results},
            "year_built_unknown": {
                row["blocklot"]: row["year_built_unknown"] for row in results
            },
        }

    def _build_pct_dark(self, blocklots):
        features = defaultdict(dict)
        threshold_models = {}
        for threshold in self.dark_pixel_thresholds:
            threshold_models[threshold] = DarkImageBaseline(threshold)

        for blocklot in tqdm(blocklots, smoothing=0, desc="Loading images"):
            if blocklot in self.dark_cache:
                for threshold in self.dark_pixel_thresholds:
                    feature_name = f"pct_pixels_darker_than_{threshold}"
                    score = self.dark_cache[blocklot][threshold]
                    features[feature_name][blocklot] = score
                continue

            image = fetch_image(blocklot)
            for threshold in self.dark_pixel_thresholds:
                feature_name = f"pct_pixels_darker_than_{threshold}"
                model = threshold_models[threshold]
                score = model.predict_proba(image)
                if score is None:
                    score = 0.0
                self.dark_cache[blocklot][threshold] = score
                features[feature_name][blocklot] = score
        return features

    def _first_vbn_notice_date(self):
        return run_query(
            """
            SELECT min(created_date)
            FROM processed.all_vacant_building_notices
            """
        )[0][0]

    def _build_vbn_features(self, blocklots, max_date, interpolation_date=None):
        if interpolation_date is None:
            interpolation_date = self._first_vbn_notice_date() - timedelta(days=365)
        results = run_query(
            """
            SELECT
                tpa.blocklot,
                EXTRACT(
                    EPOCH FROM (%s - COALESCE(min(created_date), %s)))
                    AS secs_since_first_created_date,
                EXTRACT(
                    EPOCH FROM (%s - COALESCE(max(created_date), %s)))
                    AS secs_since_last_created_date,
                COUNT(distinct vbn."NoticeNum") AS n_vbns
                FROM processed.tax_parcel_address tpa
            LEFT JOIN processed.all_vacant_building_notices as vbn
                ON vbn.blocklot = tpa.blocklot
                AND vbn.created_date <= %s
                AND vbn."FileType" = 'New Violation Notice'
            WHERE tpa.blocklot IN %s
            GROUP BY tpa.blocklot;
        """,
            (
                max_date,
                interpolation_date,
                max_date,
                interpolation_date,
                max_date,
                tuple(blocklots),
            ),
        )
        return {
            "n_vbns": {row["blocklot"]: row["n_vbns"] for row in results},
            "secs_since_first_vbn": {
                row["blocklot"]: row["secs_since_first_created_date"] for row in results
            },
            "secs_since_last_vbn": {
                row["blocklot"]: row["secs_since_last_created_date"] for row in results
            },
        }

    def load_matrix_from_cache(self, hash):
        if not self.use_cache:
            return
        path = self.cache_filename_for_hashed(hash)
        if path.is_file():
            return pd.read_pickle(path)

    def cache_filename_for_hashed(self, hash):
        directory = Path(config.matrix_creator.matrix_cache_dir) / config.schema_prefix
        directory.mkdir(parents=True, exist_ok=True)
        filename = f"{hash}.df.pkl"
        return directory / filename

    def save_matrix_to_cache(self, matrix, hash):
        if not self.use_cache:
            return
        path = self.cache_filename_for_hashed(hash)
        matrix.to_pickle(path)

    def _build_code_violations_features(self, blocklots, max_date):
        results = run_query(
            """
            SELECT
                tpa.blocklot,
                count(distinct cva.noticenum) AS n_code_violations
            FROM processed.tax_parcel_address tpa
            LEFT JOIN processed.code_violations_after_2017 cva
                ON tpa.blocklot = cva.blocklot
                AND cva.datecreate <= %s
                AND cva.statusorig = 'NOTICE APPROVED'
            WHERE tpa.blocklot IN %s
            GROUP BY tpa.blocklot;
        """,
            (max_date, tuple(blocklots)),
        )
        return {
            "n_code_violations": {
                row["blocklot"]: row["n_code_violations"] for row in results
            },
        }

    def _build_inspection_note_features(self, blocklots, max_date, words):
        features = {}
        for word in words:
            results = run_query(
                f"""
                SELECT tpa.blocklot, COUNT(DISTINCT lowered_detail) AS n_mentions
                FROM processed.tax_parcel_address tpa
                LEFT JOIN processed.inspection_notes insp
                    ON tpa.blocklot = insp.blocklot
                    AND insp.created_date <= %s
                    AND insp.lowered_detail LIKE '%%{word}%%'
                WHERE tpa.blocklot IN %s
                GROUP BY tpa.blocklot
            """,
                (max_date, tuple(blocklots)),
            )
            features[f"n_insp_note_mentions_of_{word}"] = {
                r["blocklot"]: r["n_mentions"] for r in results
            }
        return features

    def _build_redlining_features(self, blocklots):
        redline_classes = ["A", "B", "C", "D", "AUD", "BUD"]
        results = run_query(
            """
            SELECT tpa.blocklot, red."class" AS redline_class
            FROM processed.tax_parcel_address tpa
            LEFT JOIN raw.redlining as red
            ON ST_Contains(red.shape, tpa.wkb_geometry)
            WHERE tpa.blocklot IN %s
        """,
            (tuple(blocklots),),
        )
        features = {}
        for redline_class in redline_classes:
            features[f"in_redline_{redline_class}"] = {
                r["blocklot"]: (1 if r["redline_class"] == redline_class else 0)
                for r in results
            }
        return features

    def _build_construction_permit_features(self, blocklots, max_date):
        results = run_query(
            """
            SELECT
                tp.blocklot,
                COUNT(DISTINCT id_permit) AS n_permits
            FROM processed.tax_parcel_address AS tp
            LEFT JOIN raw.building_construction_permits bcp
                ON tp.blocklot = bcp.blocklot
                AND bcp.csm_issued_date <= %s
            WHERE
            tp.blocklot IN %s
            GROUP BY tp.blocklot
        """,
            (max_date, tuple(blocklots)),
        )
        return {
            "n_construction_permits": {r["blocklot"]: r["n_permits"] for r in results}
        }

    def _build_demolition_features(self, blocklots, max_date):
        results = run_query(
            """
                SELECT
                    tp.blocklot,
                    COUNT(distinct "ID_Demo_RFA") AS n_demos
                FROM processed.tax_parcel_address AS tp
                LEFT JOIN raw.demolitions_as_of_20220706 demo
                    ON tp.blocklot = demo."BlockLot"
                    AND demo."DateDemoFinish" <= %s
                WHERE
                tp.blocklot IN %s
                GROUP BY tp.blocklot
            """,
            (max_date, tuple(blocklots)),
        )
        return {"n_demolitions": {r["blocklot"]: r["n_demos"] for r in results}}

    def _build_real_estate_features(self, blocklots, max_date):
        results = run_query(
            """
            SELECT
                tp.blocklot,
                EXTRACT(EPOCH FROM(%s - COALESCE(LAST_VALUE(red.deed_date) OVER w,
                        '2011-09-01'::timestamp)))
                        AS secs_since_last_sale,
                COALESCE(LAST_VALUE(red.adjusted_price) OVER w, 0) AS last_sale_price,
                LAST_VALUE(red.adjusted_price) OVER w IS NULL AS last_sale_unknown
            FROM processed.tax_parcel_address AS tp
            LEFT JOIN processed.real_estate_data red
                ON tp.blocklot = red.blocklot
            AND red.deed_date <= %s
            WHERE tp.blocklot IN %s
            WINDOW w AS (PARTITION BY red.blocklot ORDER BY deed_date DESC)
            """,
            (max_date, max_date, tuple(blocklots)),
        )
        return {
            "secs_since_last_sale": {
                r["blocklot"]: r["secs_since_last_sale"] for r in results
            },
            "last_sale_price": {
                r["blocklot"]: int(r["last_sale_price"]) for r in results
            },
            "last_sale_unknown": {
                r["blocklot"]: r["last_sale_unknown"] for r in results
            },
        }

    def _build_transfer_learned_features(self, blocklots):
        if self.transfer_learned_score.model_group_id != "None":
            scores = Evaluator(config.evaluator).model_group_scores(
                self.transfer_learned_score.model_group_id,
                self.transfer_learned_score.schema_prefix,
            )
            blocklot_scores = scores.score.to_dict()
        elif self.transfer_learned_score.model_id:
            blocklot_scores = Predictor.load_all_preds(
                self.transfer_learned_score.model_id,
                self.transfer_learned_score.schema_prefix,
            )
        return {
            "transfer_learned_score": {
                b: blocklot_scores.get(b, 0.0) for b in blocklots
            },
            "transfer_learned_score_unknown": {
                b: int(b not in blocklot_scores) for b in blocklots
            },
        }

    def calc_feature_matrix(self, split_kind, split_id, max_date=None, blocklots=None):
        if blocklots is None:
            blocklots = self._blocklots_for_split(split_id)

        imputed_year = self.year_built_imputed
        if imputed_year is None:
            imputed_year = self._median_year_built(split_kind, split_id)

        if max_date is None:
            max_date = self.max_date
        return self.calc_feature_matrix_for_blocklots(blocklots, max_date, imputed_year)

    def calc_feature_matrix_for_blocklots(
        self, blocklots, max_date=None, imputed_year=None
    ):

        if imputed_year is None:
            imputed_year = self.year_built_imputed
        assert imputed_year is not None

        if max_date is None:
            max_date = self.max_date

        m = hashlib.sha256()
        m.update((" ".join(sorted(blocklots))).encode("utf-8"))
        m.update(str(imputed_year).encode("utf-8"))
        m.update(str(max_date).encode("utf-8"))
        hashed = m.hexdigest()

        if self.use_cache:
            loaded = self.load_matrix_from_cache(hashed)
            if loaded is not None:
                return loaded

        matrix = {}

        # image features
        if "dark_pixels" not in self.disabled_features:
            matrix.update(self._build_pct_dark(blocklots))

        # vbns
        if "vbns" not in self.disabled_features:
            matrix.update(self._build_vbn_features(blocklots, max_date))

        # 311 radii
        if "311" not in self.disabled_features:
            for radius in self.radii_of_311:
                matrix.update(self._build_311_radius(blocklots, radius, max_date))
            matrix.update(self._build_311_type_features(blocklots, max_date))

        # Year built
        if "year" not in self.disabled_features:
            matrix.update(self._build_year_built(blocklots, imputed_year, max_date))

        # Code violations
        if "code_violations" not in self.disabled_features:
            matrix.update(self._build_code_violations_features(blocklots, max_date))

        # Inspection notes
        if "inspection_notes" not in self.disabled_features:
            matrix.update(
                self._build_inspection_note_features(
                    blocklots, max_date, self.inspection_note_words
                )
            )

        # Redlining
        if "redlining" not in self.disabled_features:
            matrix.update(self._build_redlining_features(blocklots))

        # Construction permits
        if "construction_permits" not in self.disabled_features:
            matrix.update(self._build_construction_permit_features(blocklots, max_date))

        # Construction permits
        if "demos" not in self.disabled_features:
            matrix.update(self._build_demolition_features(blocklots, max_date))

        # Real estate
        if "real_estate" not in self.disabled_features:
            matrix.update(self._build_real_estate_features(blocklots, max_date))

        # Real estate
        if "transfer_learning" not in self.disabled_features:
            matrix.update(self._build_transfer_learned_features(blocklots))

        output = pd.DataFrame(matrix)
        if self.use_cache:
            self.save_matrix_to_cache(output, hashed)
        return output

    def write_feature_matrices(self, max_date, blocklots=None):
        for kind, _, split_id in tqdm(self._splits(), desc="Feature matrices"):
            matrix = self.calc_feature_matrix(kind, split_id, max_date, blocklots)
            self.save_matrix(matrix, split_id)


if __name__ == "__main__":
    sample_blocklots = [
        "0001 028",
        "0001 033C",
        "0001 043",
        "0002 043",
        "0002 059",
        "0003 001",
        "0003 002",
        "0003 062",
        "0003 067A",
        "0004 046",
        "0006 025",
        "0006 029B",
        "0006 032",
        "0007 001",
        "0008 019",
        "0008 032",
        "0008 062",
        "0008 064",
        "0009 007",
        "0009 024",
    ]
    creator = MatrixCreator(config.matrix_creator, "model_prep.split_kinds")
    creator.write_feature_matrices(creator.max_date, blocklots=sample_blocklots)
