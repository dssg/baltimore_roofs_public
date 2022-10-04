import click
import logging

import censusdata
import geopandas as gpd
import pandas as pd
import ohio.ext.pandas  # noqa: F401
from us import states
from sqlalchemy import create_engine

from src import db

pd.set_option("display.expand_frame_repr", False)
pd.set_option("display.precision", 2)

# Baltimore City county specific constants
COUNTY_FIPS = "510"
ESPG_PROJECTION = 2248
baltimore_census_geo = censusdata.censusgeo(
    [("state", states.MD.fips), ("county", COUNTY_FIPS), ("block group", "*")]
)


def get_baltimore_census_data(
    src: str, year: int, attributes_to_download_from_census: dict
) -> pd.DataFrame:
    """
    Get relevant census data (and update in DB?)

    Args:
        censusgeo (censusgeo): censusgeo of interest
        src (str): The name of census data source (e.g. acs5 for ACS 1-year estimates)
        attributes_to_download_from_census (dict): Map of table name to column name

    """

    # Get relevant census data
    census_df = censusdata.download(
        src,
        year,
        baltimore_census_geo,
        list(attributes_to_download_from_census.keys()),
    )

    census_df.rename(
        columns=attributes_to_download_from_census,
        index=map_to_geoid(census_df),
        inplace=True,
    )
    return census_df


def get_blockgroup_shapefile(
    year: int, state: str, espg_projection: int
) -> pd.DataFrame:
    MD_TIGER_URL = (
        f"https://www2.census.gov/geo/tiger/TIGER{year}/BG/tl_{year}_{state}_bg.zip"
    )
    geo_df = gpd.read_file(MD_TIGER_URL)
    # We could filter to baltimore county: geo_df = geo_df[geo_df["COUNTYFP"] == COUNTY_FIPS]
    geo_df.to_crs(epsg=espg_projection, inplace=True)
    return pd.DataFrame(geo_df.set_index(geo_df["GEOID"])["geometry"])


def map_to_geoid(census_df: pd.DataFrame) -> dict:
    # Note: hack to get geoid - could not find in existing library
    map_censusgeo_to_geoid = {}
    censusgeos = list(census_df.index)
    for censusgeo in censusgeos:
        params = censusgeo.params()
        geoid = ""
        # params example: (('state', 24), ('county', 510)...)
        for param in params:
            geoid += param[1]
        map_censusgeo_to_geoid[censusgeo] = geoid
    return map_censusgeo_to_geoid


def get_attributes_to_download_from_census() -> dict:
    attributes = {}
    with open("src/data/attributes_to_download_from_census.txt") as f:
        for line in f:
            (key, val) = line.split(":")
            attributes[key.strip()] = val.strip()
    return attributes


def process_census_tables(schema):
    """
    df.pg_copy_to uploads geometry as string.
    Process the table to convert this to geometry type.
    """
    db.run_query(
        f"""
        alter table {schema}.blockgroup_shapefiles_2020 add column shape geometry;

        update
            census.blockgroup_shapefiles_2020
        set
            shape = ST_GeomFromText(geometry,
                2248);

        create index bg_geom_id
        on
        census.blockgroup_shapefiles_2020
            using GIST (shape);

        alter table {schema}.blockgroup_shapefiles_2020 drop column geometry;
        alter table {schema}.blockgroup_shapefiles_2020 rename column shape to geometry;
        """
    )

    # db.run_query(
    #     """
    #     update
    #         census.blockgroup_shapefiles_2020
    #     set
    #         shape = ST_GeomFromText(geometry,
    #             2248);
    #     """
    # )
    # db.run_query(
    #     """
    #     create index bg_geom_id
    #     on
    #     {schema}.blockgroup_shapefiles_2020
    #         using GIST (shape);
    #     """
    # )
    # db.run_query(
    #     """
    #     alter table {schema}.blockgroup_shapefiles_2020 drop column geometry;
    #     """
    # )

    # db.run_query(
    #     """
    #     alter table {schema}.blockgroup_shapefiles_2020 rename column shape to geometry;
    #     """
    # )


@click.command()
@click.option("--schema", default="census_process", type=str, help="Schema name")
@click.option("--year", default=2020, type=int, help="Year of interest")
@click.option("--src", default="acs5", type=str, help="Census data source")
def load_data(schema, year, src):
    """Load data from baltimore census data (blockgroup level) into the DB.
    demographics table contains attributes of interest (index: census geoid)
    blockgroup_shapefiles table contains blockgroup shapefiles (index: census geoid)

    This function assumes you have all the requisite database credentials
    as environment variables already.
    """

    engine = create_engine(db.connection_string())
    logger = logging.getLogger(__name__)

    table_demographics = f"demographics_{year}"
    table_shapefiles = f"blockgroup_shapefiles_{year}"
    census_df = get_baltimore_census_data(
        src, year, get_attributes_to_download_from_census()
    )
    db.run_query(
        f"""
        drop table if exists {schema}.{table_demographics};
        drop table if exists {schema}.{table_shapefiles};
        """
    )
    census_df.pg_copy_to(table_demographics, engine, schema=schema)
    logger.info("Loaded census data with %s rows", census_df.shape[0])

    # table = f"blockgroup_shapefiles_{year}"
    geo_df = get_blockgroup_shapefile(year, states.MD.fips, ESPG_PROJECTION)
    geo_df.pg_copy_to(table_shapefiles, engine, schema=schema)
    logger.info("Loaded geo shapefiles with %s rows", geo_df.shape[0])
    process_census_tables(schema)


if __name__ == "__main__":
    load_data()
