from __future__ import annotations

import enum
import logging
import pathlib as pl
from collections import defaultdict
from typing import TYPE_CHECKING, Any, TypeAlias, TypeVar

import fsspec
import numpy as np
import xarray as xr

if TYPE_CHECKING:
    import geopandas as gpd_t
    import shapely.geometry as shg_t

LOGGER = logging.getLogger(__name__)
T = TypeVar("T", bound=enum.Enum)

PolygonLike: TypeAlias = (
    "str | pl.Path | gpd_t.GeoDataFrame | shg_t.Polygon"
    "| tuple[float, float, float, float]"
)


def normalize_polygon(polygon: PolygonLike) -> gpd_t.GeoDataFrame:
    """Normalize provided polygon to a GeoDataFrame.

    Parameters
    ----------
    polygon
        Polygon as a string (path), a GeoDataFrame or a Polygon,
        or a bounding box as a tuple of floats as (lon_min, lat_min, lon_max, lat_max)

    Returns
    -------
    :
        Normalized polygon.
    """
    LOGGER.debug("Normalizing polygon.")

    import geopandas as gpd
    import shapely.geometry as shg

    match polygon:
        case tuple():
            minx, miny, maxx, maxy = polygon
            polygon = shg.box(minx, miny, maxx, maxy)
            gdf = gpd.GeoDataFrame(index=[0], crs="epsg:4326", geometry=[polygon])
        case str() | pl.Path():
            gdf = gpd.read_file(polygon)
        case shg.Polygon():
            gdf = gpd.GeoDataFrame(index=[0], crs="epsg:4326", geometry=[polygon])
        case gpd.GeoDataFrame() | gpd.GeoSeries():
            gdf = gpd.GeoDataFrame(polygon)
        case _:
            msg = f"Provide polygon type is invalid: {type(polygon)}"
            raise TypeError(msg)

    return gdf


def polygon_bounding_box(
    polygon: gpd_t.GeoDataFrame | None,
) -> tuple[float, float, float, float] | None:
    """Extract the bounding box from provided polygon.

    Parameters
    ----------
    polygon
        Polygon from which to extract the bounding box.

    Returns
    -------
    :
        Polygon's bounding box (min_lon, min_lat, max_lon, max_lat).
    """
    if polygon is None:
        return None

    polygon_norm = normalize_polygon(polygon=polygon)

    bounds: dict[str, list[float]] = {
        "min_lon": [],
        "min_lat": [],
        "max_lon": [],
        "max_lat": [],
    }

    for v in polygon_norm.geometry.values:
        geo_bounds = v.bounds
        bounds["min_lon"].append(geo_bounds[0])
        bounds["min_lat"].append(geo_bounds[1])
        bounds["max_lon"].append(geo_bounds[2])
        bounds["max_lat"].append(geo_bounds[3])

    min_lon = min(bounds["min_lon"])
    min_lat = min(bounds["min_lat"])
    max_lon = max(bounds["max_lon"])
    max_lat = max(bounds["max_lat"])

    return min_lon, min_lat, max_lon, max_lat


def restrict_to_box(
    data: xr.Dataset,
    box: tuple[float, float, float, float],
    index: str,
    longitude: str,
    latitude: str,
) -> xr.Dataset:
    """Restrict the data to the provided bounding box.

    Parameters
    ----------
    data
        Data from which to select data.
    box
        Bounding box (min_lon, min_lat, max_lon, max_lat).
    index
        Data's index name.
    longitude
        Data's longitude name.
    latitude
        Data's latitude name.

    Returns
    -------
    :
        Dataset reduced to the bounding box constraints.
    """
    LOGGER.debug("Applying bounding box selection.")
    min_lon, min_lat, max_lon, max_lat = box

    mask: xr.DataArray = (
        (data[longitude] >= min_lon)
        & (data[longitude] <= max_lon)
        & (data[latitude] >= min_lat)
        & (data[latitude] <= max_lat)
    )

    try:
        data = data.sel({index: mask.values})
    except (ValueError, IndexError):
        # For multi-dimensional dimensions
        data = data.where(mask, drop=True)

    return data


def restrict_to_polygon(
    data: xr.Dataset,
    polygon: PolygonLike,
    index: str,
    longitude: str,
    latitude: str,
) -> xr.Dataset:
    """Restrict the data to the provided polygon.

    Parameters
    ----------
    data
        Data from which to select data.
    polygon
        Selection polygon on which to reduce the data.
    index
        Data's index name.
    longitude
        Data's longitude name.
    latitude
        Data's latitude name.

    Returns
    -------
    :
        Dataset reduced to the polygon constraints.
    """
    import geopandas as gpd

    polygon = normalize_polygon(polygon=polygon)
    LOGGER.debug("Applying polygon selection.")

    df = data[[longitude, latitude]].to_dataframe()
    df = gpd.GeoDataFrame(
        df,
        crs="epsg:4326",
        geometry=gpd.points_from_xy(df[longitude], df[latitude]),
    )

    df = gpd.sjoin(df, polygon, how="left", predicate="within")
    # Removing potential duplicate (if contained by multiple shapes)
    df = df[~df.index.duplicated(keep="first")]
    mask = df["index_right"].notnull().values
    mask = mask.reshape(data[longitude].shape)

    if len(data[longitude].dims) == 1:
        data = data.sel({index: mask})
    else:
        data = _multi_dim_sel(data=data, mask=mask, dims=data[longitude].dims)

    return data


def _multi_dim_sel(
    data: xr.Dataset, mask: np.ndarray, dims: tuple[str, ...]
) -> xr.Dataset:
    """Applying xarray where while avoiding to broadcast variables that do not
    need to be broadcasted.

    Parameters
    ----------
    data
        Data on which to apply the selection.
    mask
        Selection mask.
    dims
        Selection's dimensions.

    Returns
    -------
    :
        Selected data.
    """
    mask_arr = xr.DataArray(mask.astype(bool), dims=dims)
    # Splitting variables according to their dimensions
    dims_sel = set(dims)
    full_sel = []
    no_sel = []
    single_sel = defaultdict(list)

    for n, v in data.variables.items():
        dims_c = dims_sel.intersection(v.dims)
        if dims_c == dims_sel or len(dims_c) > 1:
            # Letting where do its job on these variables
            full_sel.append(n)
        elif not dims_c:
            # No selection on these variables
            no_sel.append(n)
        else:
            # Single common dimension
            single_sel[dims_c.pop()].append(n)

    data_x = data[full_sel].where(mask_arr, drop=True)

    for n in no_sel:
        data_x[n] = data[n]

    for d, variables in single_sel.items():
        # Eliminating index fully masked over the other dimensions
        mask_d = (~mask_arr).all(dim=[dx for dx in mask_arr.dims if dx != d])
        data_x = xr.merge([data_x, data[variables].isel({d: ~mask_d})])

    return data_x


def missing_dependency_class(dependency: str, error: str | None = None) -> type:
    """Generate a placeholder class raising an ImportError when used.

    Parameters
    ----------
    dependency
        Name of the missing dependency.
    error
        Error message raised when trying to import the dependency.

    Returns
    -------
    :
        Placeholder class.
    """
    message = f"Optional module '{dependency}' is missing."

    if error is not None:
        message += f" Error: {error}"

    class _OptionalDependency:
        def __init__(self, *args, **kwargs):
            raise ImportError(message)

    return _OptionalDependency


def normalize_enum(itype: None | str | T, kls: type[T]) -> T:
    """Normalize input parameters by making it an instance of the provided
    class.

    Parameters
    ----------
    itype
        Type to normalize as string or instance of kls.
    kls
        Class to which we want to normalize.

    Returns
    -------
    :
        Instance of the provided class.

    Raises
    ------
    ValueError
        If the provided type is invalid.
    """
    if itype is None:
        itype = str(itype)

    if type(itype) is str:
        try:
            return kls[itype.upper()]
        except KeyError as e:
            msg = f"{itype} is not a valid {kls.__name__}"
            raise ValueError(msg) from e

    return kls(itype)


def normalize_file_system(
    fs: fsspec.AbstractFileSystem | str | dict[str, Any] | None,
) -> fsspec.AbstractFileSystem:
    """Normalize the provided file system reference.

    Parameters
    ----------
    fs
        File system or path.

    Returns
    -------
    :
        Normalized fsspec file system.
    """
    protocol: str
    options: dict[str, Any]

    match fs:
        case None:
            protocol = "file"
            options = {}
        case str():
            protocol = fs
            options = {}
        case dict():
            protocol = fs.get("protocol", "file")
            options = fs.get("options", {})
        case _:
            return fs

    return fsspec.filesystem(protocol=protocol, **options)
