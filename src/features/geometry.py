import rasterio
import shapely.wkb
from shapely.geometry.polygon import Polygon

from src.db import run_query


class RecordNotFoundError(Exception):
    """We couldn't find a corresponding record in the database."""

    pass


def tile_to_bounds(tile: rasterio.DatasetReader) -> Polygon:
    """Return the bounds of the image tile as a Shapely shape.

    Args:
        tile: A dataset (e.g. GeoTIFF) read in with rasterio.

    Returns:
        The geographic bounds of the dataset as a polygon
    """
    b = tile.bounds
    return Polygon(
        [(b.left, b.top), (b.right, b.top), (b.right, b.bottom), (b.left, b.bottom)]
    )


def wkb_to_shape(
    wkb: str, buffer: int = 0, simplify: bool = True, hull: bool = True
) -> Polygon:
    """Turn a well-known-binary represenation of a geometry into a shape

    Args:
        wkb: A Well-Known-Binary representation of a geometry
        buffer: Make the shape a this many units bigger than the geometry all the way
            around. The unit is defined by the coordinate system of the geometry.
        simplify: Whether or not to simplify the resulting shape
        hull: Whether or not to return just the hull of the shape, i.e. remove the holes

    Returns:
        The Shapely polygon of the geometry
    """
    shape = shapely.wkb.loads(wkb, hex=True)
    # Buffering even with 0 helps clean up shape
    shape = shape.buffer(buffer)
    if simplify:
        shape = shape.simplify(2)
    if hull:
        if isinstance(shape, shapely.geometry.multipolygon.MultiPolygon):
            shape = shape.convex_hull
        else:
            shape = shapely.geometry.polygon.Polygon(shape.exterior)
    return shape


def fetch_blocklot_geometry(blocklot: str) -> str:
    """Get the geometry of the blocklot from the database.

    Args:
        blocklot: The block lot in "bbbb lll" form.

    Returns:
        The geometry of the block as a well-known-binary hex string.

        The database stores these geometries in EPSG:2248, which is the
        projection for Maryland, and uses feet as its reference unit.
    """
    query = "SELECT wkb_geometry FROM raw.tax_parcel_address WHERE blocklot = %s"
    results = run_query(query, (blocklot,))
    if len(results) == 0:
        raise RecordNotFoundError("No records found for blocklot {}".format(blocklot))
    return results[0][0]


def blocklot_to_shape(blocklot: str, *to_shape_args, **to_shape_kwargs) -> Polygon:
    """Return the shape of a given blocklot

    Args:
        blocklot: The block lot in "bbbb lll" form.
        *to_shape_args: Arguments to pass to shape creation (`wkb_to_shape`)
        **to_shape_kwargs: Keyword arguments to pass to shape creation (`wkb_to_shape`)

    Returns:
        The geometry of the block as a Shapely polygon
    """
    return wkb_to_shape(
        fetch_blocklot_geometry(blocklot), *to_shape_args, **to_shape_kwargs
    )


def building_to_shape(blocklot: str, *to_shape_args, **to_shape_kwargs) -> Polygon:
    """Return the shape of the buildings on a blocklot

    This assumes that there's only a single building on the blocklot.

    Args:
        blocklot: The block lot in "bbbb lll" form.
        *to_shape_args: Arguments to pass to shape creation (`wkb_to_shape`)
        **to_shape_kwargs: Keyword arguments to pass to shape creation (`wkb_to_shape`)

    Returns:
        The geometry of the building as a Shapely polygon
    """
    building_outline = run_query(
        """
        SELECT ST_Intersection(tp.wkb_geometry, bo.shape)
        FROM raw.buildingoutline_2010 AS bo
        JOIN raw.tax_parcel_address AS tp
            ON ST_Intersects(tp.wkb_geometry, bo.shape)
        WHERE blocklot = %s
        """,
        (blocklot,),
    )

    return wkb_to_shape(building_outline[0][0], *to_shape_args, **to_shape_kwargs)
