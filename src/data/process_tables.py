import logging

from psycopg2 import sql

from src.config import experiment_config as config
from src.db import run, run_query

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

RAW_SCHEMA = "raw"
CLEAN_SCHEMA = "processed"


def create_schema(schema):
    query = sql.SQL(
        """
    CREATE SCHEMA IF NOT EXISTS {schema} AUTHORIZATION "baltimore-roofs-role";
    """
    ).format(schema=sql.Identifier(schema))
    run(query)


def alter_data_311_table_structure(schema):
    date_cols = [
        "created_date",
        "status_date",
        "due_date",
        "last_activity_date",
        "close_date",
    ]
    for col in date_cols:
        query = sql.SQL(
            """
            ALTER TABLE {schema}.data_311 ALTER COLUMN {col} TYPE timestamp
            USING {col}::timestamp;
        """
        ).format(schema=sql.Identifier(schema), col=sql.Identifier(col))
        run(query)


def copy_table_structure_to_schema(from_schema, from_table, to_schema, to_table):
    query = sql.SQL(
        """
    CREATE TABLE IF NOT EXISTS {to_schema}.{to_table}
    (LIKE {from_schema}.{from_table} INCLUDING ALL);
    """
    ).format(
        from_schema=sql.Identifier(from_schema),
        from_table=sql.Identifier(from_table),
        to_schema=sql.Identifier(to_schema),
        to_table=sql.Identifier(to_table),
    )
    run(query)


def process_tax_parcel_address(from_schema, to_schema):
    cohort_blocklots = [row["blocklot"] for row in run_query(config.cohort.query)]
    logger.info("Filtering tax_parcel_address to %s blocklots", len(cohort_blocklots))
    query = sql.SQL(
        """
        WITH blocklot_row AS (
            SELECT
                objectid,
                row_number()
            OVER (
                PARTITION BY blocklot
                ORDER BY shape_area desc)
            FROM raw.tax_parcel_address tpa
            WHERE blocklot IN %s
        )
        SELECT
            wkb_geometry,
            tpa.objectid,
            pin,
            pinrelate,
            tpa.blocklot,
            block,
            lot,
            ward,
            section,
            assessor,
            taxbase,
            bfcvland,
            bfcvimpr,
            landexmp,
            imprexmp,
            citycred,
            statcred,
            ccredamt,
            scredamt,
            permhome,
            assesgrp,
            lot_size,
            no_imprv,
            currland,
            currimpr,
            exmpland,
            exmpimpr,
            fullcash,
            exmptype,
            exmpcode,
            usegroup,
            zonecode,
            sdatcode,
            artaxbas,
            distswch,
            dist_id,
            statetax,
            city_tax,
            ar_owner,
            deedbook,
            deedpage,
            saledate,
            owner_abbr,
            owner_1,
            owner_2,
            owner_3,
            fulladdr,
            stdirpre,
            st_name,
            st_type,
            bldg_no,
            fraction,
            unit_num,
            span_num,
            spanfrac,
            zip_code,
            extd_zip,
            dhcduse1,
            dhcduse2,
            dhcduse3,
            dhcduse4,
            dwelunit,
            eff_unit,
            roomunit,
            rpdeltag,
            salepric,
            propdesc,
            neighbor,
            srvccntr,
            year_build,
            structarea,
            ldate,
            ownmde,
            grndrent,
            subtype_geodb,
            sdatlink,
            blockplat,
            mailtoadd,
            vacind,
            projdemo,
            respagcy,
            releasedate,
            vbn_issued,
            name,
            shape_length,
            shape_area
    INTO {to_schema}.tax_parcel_address
    FROM {from_schema}.tax_parcel_address AS tpa
    JOIN blocklot_row AS br
        ON br.objectid = tpa.objectid
    WHERE
        tpa.blocklot IN %s
        AND br.row_number = 1
            """
    ).format(
        from_schema=sql.Identifier(from_schema), to_schema=sql.Identifier(to_schema)
    )
    run(query, (tuple(cohort_blocklots), tuple(cohort_blocklots)))
    run(
        """CREATE INDEX tpa_wkb_geom_idx ON processed.tax_parcel_address
        USING gist (wkb_geometry)"""
    )
    run("CREATE UNIQUE INDEX blocklot_idx ON processed.tax_parcel_address (blocklot)")


def process_data_311(from_schema, to_schema):
    query = sql.SQL(
        """
    SELECT
        objectid,
        ST_Transform(ST_SetSRID(shape, 4326), 2248) AS shape,
        sr_id,
        service_request_number,
        sr_type,
        NULLIF(created_date, 'NULL')::timestamp AS created_date,
        sr_status,
        NULLIF(status_date, 'NULL')::timestamp AS status_date,
        priority,
        NULLIF(due_date, 'NULL')::timestamp AS due_date,
        week_number,
        last_activity,
        NULLIF(last_activity_date, 'NULL')::timestamp AS last_activity_date,
        outcome,
        method_received,
        source,
        street_address,
        zip_code,
        neighborhood,
        latitude,
        longitude,
        police_district,
        council_district,
        vri_focus_area,
        case_details,
        geo_census_tract,
        geo_bulk_pickup_route,
        geo_east_west,
        geo_fire_inspection_area,
        geo_hcd_inspection_district,
        geo_transportation_sector,
        geo_primary_snow_zone,
        geo_street_light_service_area,
        geo_mixed_refuse_schedule,
        geo_refuse_route_number,
        geo_tree_region,
        geo_water_area,
        geo_sw_quad,
        block_number_c,
        details,
        assigned_to,
        int_comments,
        NULLIF(close_date, 'NULL')::timestamp AS close_date,
        chip_id,
        sf_source,
        contact_name,
        contact_email,
        contact_primary_phone,
        flex_summary,
        borough,
        additional_comments,
        community_stat_area,
        sr_parent_id,
        sr_duplicate_id,
        sr_parent_id_transfer,
        hashedrecord,
        agency
    INTO {to_schema}.data_311
    FROM {from_schema}.data_311
    WHERE
        longitude IS NOT NULL
        AND latitude IS NOT NULL
        """
    ).format(
        to_schema=sql.Identifier(to_schema), from_schema=sql.Identifier(from_schema)
    )
    run(query)
    run(
        sql.SQL(
            """CREATE INDEX data_311_created_date_idx
                ON {to_schema}.data_311 (created_date)"""
        ).format(to_schema=sql.Identifier(to_schema))
    )
    run(
        sql.SQL(
            "CREATE INDEX data_311_shape_idx ON {to_schema}.data_311 USING gist (shape)"
        ).format(to_schema=sql.Identifier(to_schema))
    )


def process_all_vacant_building_notices(from_schema, to_schema):
    run(
        sql.SQL(
            """
        SELECT
            *,
            rpad("Block", 5) || "Lot" AS blocklot,
            "DateCreate"::timestamp AS created_date
        INTO {to_schema}.all_vacant_building_notices
        FROM {from_schema}.all_vacant_building_notices AS tpa
    """
        ).format(
            to_schema=sql.Identifier(to_schema), from_schema=sql.Identifier(from_schema)
        )
    )
    run(
        sql.SQL(
            """CREATE INDEX all_vbn_blocklot_idx
                ON {to_schema}.all_vacant_building_notices (blocklot)"""
        ).format(to_schema=sql.Identifier(to_schema))
    )


def process_code_violations_after_2017(from_schema, to_schema):
    run(
        sql.SQL(
            """
        SELECT
            *,
            rpad("block", 5) || "lot" AS blocklot
        INTO {to_schema}.code_violations_after_2017
        FROM {from_schema}.codeviolationdata_after2017
    """
        ).format(
            to_schema=sql.Identifier(to_schema), from_schema=sql.Identifier(from_schema)
        )
    )
    run(
        sql.SQL(
            """CREATE INDEX code_vio_blocklot_idx
                ON {to_schema}.code_violations_after_2017 (blocklot)"""
        ).format(to_schema=sql.Identifier(to_schema))
    )


def process_inspection_notes(from_schema, to_schema):
    run(
        sql.SQL(
            """
        SELECT
            *,
            datecreate::timestamp AS created_date,
            rpad("block", 5) || "lot" AS blocklot,
            lower(detail) AS lowered_detail
        INTO {to_schema}.inspection_notes
        FROM {from_schema}.inspection_notes
    """
        ).format(
            to_schema=sql.Identifier(to_schema), from_schema=sql.Identifier(from_schema)
        )
    )
    run(
        sql.SQL(
            """CREATE INDEX inspection_notes_blocklot_idx
                ON {to_schema}.inspection_notes (blocklot)"""
        ).format(to_schema=sql.Identifier(to_schema))
    )


def process_real_estate_data(from_schema, to_schema):
    run(
        sql.SQL(
            """
        SELECT
            *,
            '1899-12-31'::timestamp + (date_of_deed || ' days')::interval AS deed_date
        INTO {to_schema}.real_estate_data
        FROM {from_schema}.real_estate_data
    """
        ).format(
            to_schema=sql.Identifier(to_schema), from_schema=sql.Identifier(from_schema)
        )
    )
    run(
        sql.SQL(
            """CREATE INDEX real_estate_blocklot_idx
                ON {to_schema}.real_estate_data (blocklot)"""
        ).format(to_schema=sql.Identifier(to_schema))
    )


def process_tables():
    create_schema(CLEAN_SCHEMA)
    # copy_table_structure_to_schema(RAW_SCHEMA, "data_311", CLEAN_SCHEMA, "data_311")
    # alter_data_311_table_structure(CLEAN_SCHEMA)
    # process_data_311(RAW_SCHEMA, CLEAN_SCHEMA)
    # process_tax_parcel_address(RAW_SCHEMA, CLEAN_SCHEMA)
    # process_all_vacant_building_notices(RAW_SCHEMA, CLEAN_SCHEMA)
    # process_code_violations_after_2017(RAW_SCHEMA, CLEAN_SCHEMA)
    # process_inspection_notes(RAW_SCHEMA, CLEAN_SCHEMA)
    process_real_estate_data(RAW_SCHEMA, CLEAN_SCHEMA)


if __name__ == "__main__":
    process_tables()
