import enum

import fsspec.implementations.local as fs_local
import fsspec.implementations.memory as fs_mem
import geopandas as gpd
import numpy as np
import pytest
import shapely as shp
import xarray as xr

from cnes_alti_reader.utilities import (
    missing_dependency_class,
    normalize_enum,
    normalize_file_system,
    normalize_polygon,
    polygon_bounding_box,
    restrict_to_box,
    restrict_to_polygon,
)
from tests.conftest import INDEX, LATITUDE, LONGITUDE

DATASET_2D = xr.Dataset(
    {
        INDEX: (INDEX, np.arange(6)),
        "pix": ("pix", np.arange(3)),
        LONGITUDE: ((INDEX, "pix"), np.arange(-180, 180, 20).reshape((6, 3))),
        LATITUDE: ((INDEX, "pix"), np.arange(-90, 90, 10).reshape((6, 3))),
    }
)


def test_normalize_polygon(tmp_path):
    with pytest.raises(TypeError, match="Provide polygon type is invalid"):
        normalize_polygon(polygon=1)

    polygon = shp.Polygon([(0, 0), (0, 10), (10, 10), (10, 0), (0, 0)])

    polygon_norm = normalize_polygon(polygon=polygon)

    assert isinstance(polygon_norm, gpd.GeoDataFrame)

    polygon_2 = normalize_polygon(polygon=polygon_norm)

    assert isinstance(polygon_2, gpd.GeoDataFrame)
    assert polygon_2.equals(polygon_norm)

    shape_file = tmp_path / "polygon.shp"
    polygon_norm.to_file(shape_file)

    polygon_3 = normalize_polygon(polygon=shape_file)

    assert isinstance(polygon_3, gpd.GeoDataFrame)
    assert polygon_3.geometry.equals(polygon_norm.geometry)


@pytest.mark.parametrize(
    ("geometry", "res"),
    [
        (None, None),
        (
            [
                shp.Point(0, 0),
                shp.Point(0, 10),
                shp.Point(10, 10),
                shp.Point(10, 0),
                shp.Point(0, 0),
            ],
            (0, 0, 10, 10),
        ),
        (
            [
                shp.Point(-50, 0),
                shp.Point(0, 10),
                shp.Point(100, 10),
                shp.Point(10, 0),
                shp.Point(-50, 0),
            ],
            (-50, 0, 100, 10),
        ),
        (
            [
                shp.Point(-180, -90),
                shp.Point(0, 10),
                shp.Point(10, 10),
                shp.Point(10, 0),
                shp.Point(0, 0),
            ],
            (-180, -90, 10, 10),
        ),
    ],
)
def test_polygon_bounding_box(geometry, res):
    if geometry is None:
        polygon = None
    else:
        polygon = gpd.GeoDataFrame(data={"geometry": geometry}, geometry="geometry")

    assert polygon_bounding_box(polygon=polygon) == res


@pytest.mark.parametrize(
    ("box", "res"),
    [
        ((0, 0, 10, 10), [[0], [0]]),
        ((0, 0, 180, 10), [[0], [0]]),
        ((0, 0, 180, 40), [[0, 80], [0, 40]]),
        ((-180, -60, 180, 40), [[-80, 0, 80], [-40, 0, 40]]),
    ],
)
def test_restrict_to_box(dataset, box, res):
    data = restrict_to_box(
        data=dataset, box=box, index=INDEX, longitude=LONGITUDE, latitude=LATITUDE
    )

    assert np.array_equal(data[LONGITUDE].values, res[0])
    assert np.array_equal(data[LATITUDE].values, res[1])


@pytest.mark.parametrize(
    ("box", "res"),
    [
        ((0, 0, 10, 10), [[[0]], [[0]]]),
        ((0, 0, 40, 10), [[[0, 20]], [[0, 10]]]),
        (
            (-40, -30, 100, 45),
            [
                [[np.nan, -40, -20], [0, 20, 40], [60, 80, np.nan]],
                [[np.nan, -20, -10], [0, 10, 20], [30, 40, np.nan]],
            ],
        ),
    ],
)
def test_polygon_bounding_box_2d(box, res):
    data = restrict_to_box(
        data=DATASET_2D, box=box, index=INDEX, longitude=LONGITUDE, latitude=LATITUDE
    )

    assert np.array_equal(data[LONGITUDE].values, res[0], equal_nan=True)
    assert np.array_equal(data[LATITUDE].values, res[1], equal_nan=True)


@pytest.mark.parametrize(
    ("geometry", "res"),
    [
        (
            [
                (-0.05, -0.05),
                (-0.05, 10.05),
                (10.05, 10.05),
                (10.05, -0.05),
                (-0.05, -0.05),
            ],
            [[0], [0]],
        ),
        (
            [
                (-0.05, -0.05),
                (-0.05, 10.05),
                (180.05, 10.05),
                (180.05, -0.05),
                (-0.05, -0.05),
            ],
            [[0], [0]],
        ),
        (
            [
                (-0.05, -0.05),
                (-0.05, 40.05),
                (180.05, 40.05),
                (180.05, -0.05),
                (-0.05, -0.05),
            ],
            [[0, 80], [0, 40]],
        ),
        (
            [
                (-180.05, -60.05),
                (-180.05, 40.05),
                (180.05, 40.05),
                (180.05, -60.05),
                (-180.05, -60.05),
            ],
            [[-80, 0, 80], [-40, 0, 40]],
        ),
    ],
)
def test_restrict_to_polygon(dataset, geometry, res):
    polygon = shp.Polygon(geometry)
    data = restrict_to_polygon(
        data=dataset,
        polygon=polygon,
        index=INDEX,
        longitude=LONGITUDE,
        latitude=LATITUDE,
    )

    assert np.array_equal(data[LONGITUDE].values, res[0])
    assert np.array_equal(data[LATITUDE].values, res[1])


@pytest.mark.parametrize(
    ("geometry", "res"),
    [
        (
            [
                (-0.05, -0.05),
                (-0.05, 10.05),
                (10.05, 10.05),
                (10.05, -0.05),
                (-0.05, -0.05),
            ],
            [[[0]], [[0]]],
        ),
        (
            [
                (-0.05, -0.05),
                (-0.05, 10.05),
                (40.05, 10.05),
                (40.05, -0.05),
                (-0.05, -0.05),
            ],
            [[[0, 20]], [[0, 10]]],
        ),
        (
            [
                (-40.05, -30.05),
                (-40.05, 45.05),
                (100.05, 45.05),
                (100.05, -30.05),
                (-40.05, -30.05),
            ],
            [
                [[np.nan, -40, -20], [0, 20, 40], [60, 80, np.nan]],
                [[np.nan, -20, -10], [0, 10, 20], [30, 40, np.nan]],
            ],
        ),
    ],
)
def test_restrict_to_polygon_2d(geometry, res):
    dataset = DATASET_2D.copy()

    size_idx = dataset.sizes[INDEX]
    size_pix = dataset.sizes["pix"]

    # Adding some custom variables to test edge cases
    dataset["var_pix"] = ("pix", np.arange(3))
    dataset["var_a"] = ("a", np.arange(2))
    dataset["var_ab"] = (("a", "b"), np.arange(4).reshape((2, 2)))
    dataset["var_tab"] = (
        (INDEX, "a", "b"),
        np.arange(size_idx * 4).reshape((size_idx, 2, 2)),
    )
    dataset["var_ipab"] = (
        (INDEX, "pix", "a", "b"),
        np.arange(size_idx * size_pix * 4).reshape((size_idx, size_pix, 2, 2)),
    )

    polygon = shp.Polygon(geometry)
    data = restrict_to_polygon(
        data=dataset,
        polygon=polygon,
        index=INDEX,
        longitude=LONGITUDE,
        latitude=LATITUDE,
    )

    assert np.array_equal(data[LONGITUDE].values, res[0], equal_nan=True)
    assert np.array_equal(data[LATITUDE].values, res[1], equal_nan=True)

    # Checking that each variable kept its original dimensions
    assert data["var_pix"].dims == dataset["var_pix"].dims
    assert data["var_a"].dims == dataset["var_a"].dims
    assert data["var_ab"].dims == dataset["var_ab"].dims
    assert data["var_tab"].dims == dataset["var_tab"].dims
    assert data["var_ipab"].dims == dataset["var_ipab"].dims


def test_missing_dependency_class():
    msg = "Missing dependency: invalid_module"
    md = missing_dependency_class(dependency="invalid_module", error=msg)

    with pytest.raises(ImportError, match=msg):
        md(a=1, b=2)


def test_normalize_enum():
    class _TestEnum(enum.Enum):
        NONE = enum.auto()
        V1 = enum.auto()
        V2 = enum.auto()

    assert normalize_enum(None, _TestEnum) == _TestEnum.NONE
    assert normalize_enum("none", _TestEnum) == _TestEnum.NONE
    assert normalize_enum("v1", _TestEnum) == _TestEnum.V1
    assert normalize_enum("v2", _TestEnum) == _TestEnum.V2

    with pytest.raises(ValueError, match="is not a valid"):
        normalize_enum("v3", _TestEnum)

    class _TestEnum2(enum.Enum):
        NONE = 1
        V1 = 2
        V2 = 3
        V3 = 3

    assert normalize_enum("v2", _TestEnum2) == normalize_enum("v3", _TestEnum2)
    assert normalize_enum("v3", _TestEnum2) == _TestEnum2.V2
    assert normalize_enum("v2", _TestEnum2) == _TestEnum2.V2
    assert normalize_enum(_TestEnum2.V2, _TestEnum2) == _TestEnum2.V2


@pytest.mark.parametrize(
    ("fs", "expected"),
    [
        (None, fs_local.LocalFileSystem),
        (fs_local.LocalFileSystem(), fs_local.LocalFileSystem),
        ("file", fs_local.LocalFileSystem),
        ({"protocol": "file"}, fs_local.LocalFileSystem),
        ({"protocol": "memory"}, fs_mem.MemoryFileSystem),
    ],
)
def test_normalize_file_system(fs, expected):
    assert isinstance(normalize_file_system(fs=fs), expected)
