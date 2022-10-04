import glob
import logging
from pathlib import Path
from typing import Iterable, Union

import click
import h5py
import numpy as np
import pyproj
import rasterio
import rasterio.crs
import rasterio.errors
import rasterio.mask
import rasterio.warp
from shapely.geometry.polygon import Polygon
from shapely.geometry.multipolygon import MultiPolygon
from shapely.ops import transform
from tqdm.auto import tqdm

from src.config import experiment_config as config
from src.db import run_query
from src.features.geometry import tile_to_bounds, blocklot_to_shape, building_to_shape

logger = logging.getLogger(__name__)

hdf5_file = None


class SimpleImageParser:
    def __init__(self, root_path: str, file_ext="jpg"):
        self.root_path = Path(root_path)
        self._filenames = []
        self._file_bounds = {}
        self._file_ext = file_ext
        self._load_images()

    def _reproject_shape(self, shape, src_crs, dst_crs):
        src_proj = pyproj.CRS(src_crs)
        dst_proj = pyproj.CRS(dst_crs)

        project = pyproj.Transformer.from_crs(
            src_proj, dst_proj, always_xy=True
        ).transform
        return transform(project, shape)

    def _load_images(self, src_crs="EPSG:4326", dst_crs="EPSG:2248"):
        """Load image filename and bounds from disk into memory.

        This does not store the actual images in memory (so we don't run out of memory).
        """
        self._filenames = self._get_image_filenames(self.root_path, self._file_ext)

        self._file_bounds = {
            filename: self._reproject_shape(
                tile_to_bounds(rasterio.open(filename)), src_crs, dst_crs
            )
            for filename in tqdm(self._filenames, desc="Reading bounds", smoothing=0)
        }

    @staticmethod
    def _get_image_filenames(path, file_ext):
        return glob.glob(str(path / f"MDCBAL*Ortho*.{file_ext}"))

    def _find_images_for_shape(
        self, shape: Union[Polygon, MultiPolygon]
    ) -> list[rasterio.DatasetReader]:
        """Find all images for a given shape

        Args:
            shape (Union[Polygon, MultiPolygon]): The Shapely shape for which to find
                images.

        Returns:
            A list of rasterio images
        """
        shape_images = []
        for image, bounds in self._file_bounds.items():
            if bounds.contains(shape):
                shape_images.append(rasterio.open(image))
        return shape_images

    def images_for_blocklot(self, blocklot: str) -> list[rasterio.DatasetReader]:
        """Return image files for a given blocklot.

        Args:
            blocklot (str): The blocklot id of the tax parcel

        Returns:
            A list of rasterio datasets (images) that cover the given blocklot
        """
        return self._find_images_for_shape(blocklot_to_shape(blocklot))

    def pixels_for_blocklot(
        self, blocklot: str, tiles_crs, blocklot_crs, *to_shape_args, **to_shape_kwargs
    ) -> np.ndarray:
        """Return the pixel values for an aerial image of a blocklot.

        Args:
            blocklot (str): The blocklot id of the tax parcel
            year (int): The year of aerial data

        Returns:
            np.ndarray: The pixel values for the first aerial image.
        """ """"""
        shape = blocklot_to_shape(blocklot, *to_shape_args, **to_shape_kwargs)
        shape_tiles = self.images_for_blocklot(blocklot)
        if len(shape_tiles) == 0:
            logging.warning("Shape for blocklot {} not found in tiles".format(blocklot))
            return np.ndarray([])
        tile = shape_tiles[0]
        return self._mask_image_to_shape(
            tile, self._reproject_shape(shape, blocklot_crs, tiles_crs)
        )

    def _mask_image_to_shape(
        self, tile: rasterio.DatasetReader, shape: Polygon
    ) -> np.ndarray:
        """Given an image and a shape, return just the pixels for the shape.

        Args:
            tile (rasterio.DatasetReader): The image
            shape (Polygon): The shape

        Returns:
            np.ndarray: The pixel values for the masked region
        """
        try:
            mask, out_transform, window = rasterio.mask.raster_geometry_mask(
                tile, [shape], crop=True
            )
        except ValueError as e:
            logging.warning("Error pulling out shape from image: %s", e)
            return np.ndarray([])
        image = tile.read([1, 2, 3], window=window)
        i = image.astype(np.float16)
        i[:, mask] = np.nan
        return np.transpose(
            i,
            [1, 2, 0],
        )


class ImageParser:
    """An easy way to get at images of block lots and buildings"""

    def __init__(
        self, root_path: str, years: Iterable[int] = [2015, 2018, 2019, 2020, 2022]
    ):
        """Create a new ImageLoader.

        Args:
            root_path (str): The directory where all the annual images are stored.
            years (Iterable[int], optional): The years for which data exists in the
                image directory. Defaults to [2015, 2018, 2019, 2020, 2022].
        """
        self.root_path = Path(root_path)
        self.years = years
        self._yearly_image_filenames: dict[int, list[str]] = {}
        self._yearly_image_bounds: dict[int, dict[str, Polygon]] = {}
        self._load_images()

    def _get_image_files_for_year(self, year: int) -> list[str]:
        """Find all the image filesnames for a given year.

        Not all years currently adhere to this format, so we need to do some cleaning.

        Args:
            year (int): The year of data to fetch.

        Returns:
            list[str]: The filenames of images for that year.
        """
        if year == 2022:
            return glob.glob(str(self.root_path / "MDCBAL*Ortho*.jpg"))

        return glob.glob(
            str(
                self.root_path / f"{year}/{year}/Ortho Mosaic Tiles"
                f"/MDCBAL{year%100}-TIF-TILES-4INCH/MDCBAL*OrthoSectorTile*.tif"
            )
        )

    def _load_images(self):
        """Load image filename and bounds from disk into memory.

        This does not store the actual images in memory (so we don't run out of memory).
        """
        for year in self.years:
            self._yearly_image_filenames[year] = self._get_image_files_for_year(year)
            self._yearly_image_bounds[year] = {
                filename: tile_to_bounds(rasterio.open(filename))
                for filename in self._yearly_image_filenames[year]
            }

    def _find_images_for_shape(
        self, shape: Union[Polygon, MultiPolygon], year: int
    ) -> list[rasterio.DatasetReader]:
        """Find all images for a given shape in a given year.

        Args:
            shape (Union[Polygon, MultiPolygon]): The Shapely shape for which to find
                images.
            year (int): The year of the image to fetch

        Returns:
            A list of rasterio images
        """
        shape_images = []
        for image, bounds in self._yearly_image_bounds[year].items():
            if bounds.contains(shape):
                shape_images.append(rasterio.open(image))
        return shape_images

    def images_for_blocklot(
        self, blocklot: str, year: int
    ) -> list[rasterio.DatasetReader]:
        """Return image files for a given blocklot and year.

        Args:
            blocklot (str): The blocklot id of the tax parcel
            year (int): The year of aerial data

        Returns:
            A list of rasterio datasets (images) that cover the given blocklot
        """
        return self._find_images_for_shape(blocklot_to_shape(blocklot), year)

    def images_for_building(self, blocklot: str, year: int) -> rasterio.DatasetReader:
        """Return image files for a given building and year.

        Args:
            blocklot (str): The blocklot id of the tax parcel on which the building
                stands.
            year (int): The year of aerial data.

        Returns:
            A list of rasterio datasets (images) that cover the given blocklot's
                building.
        """
        return self._find_images_for_shape(building_to_shape(blocklot), year)

    def _mask_image_to_shape(
        self, tile: rasterio.DatasetReader, shape: Polygon
    ) -> np.ndarray:
        """Given an image and a shape, return just the pixels for the shape.

        Args:
            tile (rasterio.DatasetReader): The image
            shape (Polygon): The shape

        Returns:
            np.ndarray: The pixel values for the masked region
        """
        try:
            mask, out_transform, window = rasterio.mask.raster_geometry_mask(
                tile, [shape], crop=True
            )
        except ValueError:
            return np.ndarray([])
        image = tile.read([1, 2, 3], window=window)
        i = image.astype(np.float16)
        i[:, mask] = np.nan
        return np.transpose(
            i,
            [1, 2, 0],
        )

    def pixels_for_blocklot(
        self, blocklot: str, year: int, *to_shape_args, **to_shape_kwargs
    ) -> np.ndarray:
        """Return the pixel values for an aerial image of a blocklot.

        Args:
            blocklot (str): The blocklot id of the tax parcel
            year (int): The year of aerial data

        Returns:
            np.ndarray: The pixel values for the first aerial image.
        """ """"""
        shape = blocklot_to_shape(blocklot, *to_shape_args, **to_shape_kwargs)
        shape_tiles = self.images_for_blocklot(blocklot, year)
        if len(shape_tiles) == 0:
            logging.warning("Shape for blocklot {} not found in tiles".format(blocklot))
            return np.ndarray([])
        tile = shape_tiles[0]
        return self._mask_image_to_shape(tile, shape)

    def pixels_for_building(
        self, blocklot: str, year: int, *to_shape_args, **to_shape_kwargs
    ) -> np.ndarray:
        """Return the pixel values for an aerial image of a blocklot.

        Args:
            blocklot (str): The blocklot id of the tax parcel
            year (int): The year of aerial data

        Returns:
            np.ndarray: The pixel values for the first aerial image.
        """ """"""
        shape = building_to_shape(blocklot, *to_shape_args, **to_shape_kwargs)
        shape_tiles = self.images_for_building(blocklot, year)
        tile = shape_tiles[0]
        return self._mask_image_to_shape(tile, shape)


def blocklot_to_path(blocklot):
    block = blocklot[:5].strip()
    lot = blocklot[5:].strip()
    return Path(block) / Path(f"{lot}.npy")


def all_blocklots():
    query = f"""
        SELECT DISTINCT(blocklot)
        FROM {config.schema_prefix}_model_prep.labels
        ORDER BY blocklot"""
    results = run_query(query)
    return [row["blocklot"] for row in results]


def fetch_image_from_disk(blocklot, base_dir, buffer=20):
    block, lot = split_blocklot(blocklot)
    img_f = Path(base_dir) / block / f"{lot}.npy"
    try:
        return np.load(img_f)
    except Exception:
        logging.warning(f"Missing {img_f}")
    return None


def split_blocklot(blocklot):
    block = blocklot[:5].strip()
    lot = blocklot[5:].strip()
    return block, lot


def fetch_image_from_hdf5(blocklot, file, buffer=20):
    global hdf5_file
    if hdf5_file is None:
        logger.info("Loading hdf5 image file")
        hdf5_file = h5py.File(file, "r")
    block, lot = split_blocklot(blocklot)
    hdf5_data = hdf5_file[f"{block}/{lot}"]
    arr = np.empty_like(hdf5_data)
    hdf5_data.read_direct(arr)
    return arr


def fetch_image(blocklot, base_dir=config.blocklot_image_path, buffer=20):
    if base_dir.endswith(".hdf5"):
        return fetch_image_from_hdf5(blocklot, base_dir, buffer)
    return fetch_image_from_disk(blocklot, base_dir, buffer)


@click.command()
@click.argument("image_root", type=click.Path(exists=True))
@click.argument(
    "output_root", type=click.Path(writable=True, file_okay=False, path_type=Path)
)
@click.argument("year", type=int)
@click.option("--blocklot", "-b", type=str, multiple=True, default=[])
def parse_blocklots(image_root: str, output_root: Path, year: int, blocklot: list[str]):
    parser = ImageParser(image_root, [year])
    if len(blocklot) == 0:
        blocklot = all_blocklots()
    for lot in tqdm(blocklot, desc="Blocklots", smoothing=0):
        path = output_root / blocklot_to_path(lot)
        if path.exists():
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        pixels = parser.pixels_for_blocklot(lot, year)
        np.save(str(path), pixels)


@click.command()
@click.argument("image_root", type=click.Path(exists=True))
@click.argument("output_root", type=click.Path(writable=True))
@click.option("--blocklot", "-b", type=str, multiple=True, default=[])
def simple_parse_blocklots(image_root: str, output_root: Path, blocklot: list[str]):
    parser = SimpleImageParser(image_root)
    f = h5py.File(str(output_root), "a")

    if len(blocklot) == 0:
        blocklot = all_blocklots()
    for this_blocklot in tqdm(blocklot, desc="Blocklots", smoothing=0):
        block, lot = split_blocklot(this_blocklot)
        key = f"{block}/{lot}"
        if key not in f:
            pixels = parser.pixels_for_blocklot(this_blocklot, "EPSG:4326", "EPSG:2248")
            f.create_dataset(key, data=pixels)
    f.close()


if __name__ == "__main__":
    simple_parse_blocklots()
