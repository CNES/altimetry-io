import pathlib as pl
from datetime import datetime

import geopandas as gpd_t
import numpy as np
import pytest
import shapely.geometry as shg_t
import xarray as xr

TIME = "time"
LONGITUDE = "longitude"
LATITUDE = "latitude"
VAR1 = "ssha_noiseless"
VAR2 = "ssha2_noiseless"
CYCLE_NUMBER = "cycle_number"
PASS_NUMBER = "pass_number"
NUM_CYCLE = 1
NUM_PASS = 1
NUM_PASS2 = 2
DATE_START = np.datetime64("2024-01-01")
DATE_END = np.datetime64("2024-01-02")
DATE_START2 = np.datetime64("2024-01-03")
DATE_END2 = np.datetime64("2024-01-04")


@pytest.fixture(scope="session")
def l3_lr_ssh_basic_1():
    time = np.arange("2024-01-01T12", "2024-01-01T15", dtype="M8[h]").astype("M8[ns]")
    lon = [[1, 3, 5], [2, 4, 6], [3, 5, 7]]
    lat = [[4, 3, 2], [6, 5, 4], [8, 7, 6]]
    var1 = np.random.random((3, 3))
    var2 = np.random.random((3, 3))
    cycle_nb = np.full(3, NUM_CYCLE, dtype=np.uint16)
    pass_nb = np.full(3, NUM_PASS, dtype=np.uint16)

    return xr.Dataset(
        data_vars={
            VAR1: (("num_lines", "num_pixels"), var1),
            VAR2: (("num_lines", "num_pixels"), var2),
            CYCLE_NUMBER: (("num_lines"), cycle_nb),
            PASS_NUMBER: (("num_lines"), pass_nb),
        },
        coords={
            TIME: ("num_lines", time.astype("M8[ns]")),
            LONGITUDE: (("num_lines", "num_pixels"), lon),
            LATITUDE: (("num_lines", "num_pixels"), lat),
        },
    )


@pytest.fixture(scope="session")
def l3_lr_ssh_basic_2():
    time = np.arange("2024-01-03T12", "2024-01-03T15", dtype="M8[h]").astype("M8[ns]")
    lon = [[12, 14, 16], [13, 15, 16], [13, 15, 17]]
    lat = [[14, 13, 12], [16, 15, 14], [18, 17, 16]]
    var1 = np.random.random((3, 3))
    var2 = np.random.random((3, 3))
    cycle_nb = np.full(3, NUM_CYCLE, dtype=np.uint16)
    pass_nb = np.full(3, NUM_PASS2, dtype=np.uint16)

    return xr.Dataset(
        data_vars={
            VAR1: (("num_lines", "num_pixels"), var1),
            VAR2: (("num_lines", "num_pixels"), var2),
            CYCLE_NUMBER: (("num_lines"), cycle_nb),
            PASS_NUMBER: (("num_lines"), pass_nb),
        },
        coords={
            TIME: ("num_lines", time.astype("M8[ns]")),
            LONGITUDE: (("num_lines", "num_pixels"), lon),
            LATITUDE: (("num_lines", "num_pixels"), lat),
        },
    )


@pytest.fixture(scope="session")
def data_dir(l3_lr_ssh_basic_1, l3_lr_ssh_basic_2, tmpdir_factory):
    """The test folder will contain multiple netcdf."""
    data_dir = pl.Path(tmpdir_factory.mktemp("swot_data"))

    l3_lr_ssh_basic_1.to_netcdf(
        data_dir.joinpath(
            f"SWOT_L3_LR_SSH_Basic_{NUM_CYCLE:03d}_{NUM_PASS:03d}_"
            f"{DATE_START.astype('datetime64[s]').astype(datetime).strftime('%Y%m%dT%H%M%S')}_"
            f"{DATE_END.astype('datetime64[s]').astype(datetime).strftime('%Y%m%dT%H%M%S')}"
            f"_v0.3.nc"
        )
    )
    l3_lr_ssh_basic_2.to_netcdf(
        data_dir.joinpath(
            f"SWOT_L3_LR_SSH_Basic_{NUM_CYCLE:03d}_{NUM_PASS2:03d}_"
            f"{DATE_START2.astype('datetime64[s]').astype(datetime).strftime('%Y%m%dT%H%M%S')}_"
            f"{DATE_END2.astype('datetime64[s]').astype(datetime).strftime('%Y%m%dT%H%M%S')}"
            f"_v0.3.nc"
        )
    )

    return data_dir


@pytest.fixture(scope="session")
def polygon_gdf() -> gpd_t.GeoDataFrame:
    # Polygon selecting 2 lines (6 points)
    polygon = shg_t.Polygon(
        [
            (0, 0),
            (0, 6),
            (6, 6),
            (6, 0),
            (0, 0),
        ]
    )
    return gpd_t.GeoDataFrame(geometry=[polygon], crs="EPSG:4326")


@pytest.fixture(scope="session")
def polygon_shp(data_dir, polygon_gdf) -> pl.Path:
    file = data_dir / "my_polygon.shp"
    polygon_gdf.to_file(file)
    return file
