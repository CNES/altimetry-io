import logging

import numpy as np
import pytest
import shapely as shp
import xarray as xr

from altimetry.io.sources import FileCollectionSource
from altimetry.io.utilities import restrict_to_polygon

from .conftest import (
    CYCLE_NUMBER,
    DATE_END,
    DATE_END2,
    DATE_START,
    LATITUDE,
    LONGITUDE,
    NUM_CYCLE,
    NUM_PASS,
    NUM_PASS2,
    PASS_NUMBER,
    TIME,
    VAR1,
    VAR2,
)

try:
    import fcollections.implementations as fc_impl

    have_fc = True
except ImportError:
    fc_impl = None
    have_fc = False

if not have_fc:
    pytest.skip("Skipping fcollections tests", allow_module_level=True)


@pytest.fixture(scope="session")
def fc_source(data_dir) -> FileCollectionSource:
    """Get the file collection."""
    return FileCollectionSource(path=data_dir, ftype="SWOT_L3_LR_SSH", subset="Basic")


@pytest.fixture(scope="session")
def data_ref(fc_source) -> xr.Dataset:
    return fc_source.query_date(start=DATE_START, end=DATE_END)


def test_handler(fc_source):
    assert isinstance(fc_source.handler, fc_impl.NetcdfFilesDatabaseSwotLRL3)


def test_variables(fc_source, data_dir):
    collection_ref = fc_impl.NetcdfFilesDatabaseSwotLRL3(path=data_dir)

    assert fc_source.path == data_dir

    assert fc_source == FileCollectionSource(
        path=data_dir, ftype="SWOT_L3_LR_SSH", subset="Basic"
    )
    fields = {TIME, LONGITUDE, LATITUDE, VAR1, VAR2, CYCLE_NUMBER, PASS_NUMBER}

    assert set(fc_source.variables()) == set(
        v.name
        for v in collection_ref.variables_info(**fc_source._request_kwargs()).variables
    )
    # Validating the fields caching
    assert set(fc_source.variables()) == set(fields)


def test_period(fc_source):
    assert fc_source.period() == (DATE_START, DATE_END2)


def test_half_orbit_periods(fc_source):
    ho_periods = fc_source.half_orbit_periods()

    assert np.array_equal(
        ho_periods[["cycle_number", "pass_number"]].to_numpy(),
        np.array([[NUM_CYCLE, NUM_PASS], [NUM_CYCLE, NUM_PASS2]]),
    )


def test_query_date(fc_source, l3_lr_ssh_basic_1):
    data = fc_source.query_date(start=DATE_START, end=DATE_END)

    assert data[TIME].values[0] >= DATE_START
    assert data[TIME].values[-1] <= DATE_END
    for field in fc_source.variables().values():
        assert field.name in data
        assert np.array_equal(
            data[field.name].values, l3_lr_ssh_basic_1[field.name].values
        )

    fields = {TIME, LONGITUDE, LATITUDE, VAR1, VAR2}
    data = fc_source.query_date(start=DATE_START, end=DATE_END, variables=fields)
    assert not (set(fields) - set(data.variables))


def test_query_date_nadir_swath(fc_source, caplog):
    backend_kwargs = {
        "nadir": True,
        "swath": False,
    }
    caplog.set_level(logging.WARNING)

    fc_source.query_date(start=DATE_START, end=DATE_END, backend_kwargs=backend_kwargs)

    assert any(
        "The nadir/swath parameters cannot be applied to this collection"
        in record.message
        for record in caplog.records
    )

    assert "nadir" not in backend_kwargs
    assert "swath" not in backend_kwargs


def test_query_date_polygon(fc_source, data_ref):
    # Polygon not matching data
    polygon = shp.Polygon(
        [
            (10, 10),
            (10, 50),
            (50, 50),
            (50, 10),
            (10, 10),
        ]
    )
    data = fc_source.query_date(start=DATE_START, end=DATE_END, polygon=polygon)
    data_ref_restricted = restrict_to_polygon(
        data=data_ref,
        polygon=polygon,
        index=fc_source.index,
        longitude=fc_source.longitude,
        latitude=fc_source.latitude,
    )
    assert data.equals(data_ref_restricted)

    # Polygon matching data
    polygon = shp.Polygon(
        [
            (1, 1),
            (1, 5),
            (5, 5),
            (5, 1),
            (1, 1),
        ]
    )
    data = fc_source.query_date(start=DATE_START, end=DATE_END, polygon=polygon)
    data_ref_restricted = restrict_to_polygon(
        data=data_ref,
        polygon=polygon,
        index=fc_source.index,
        longitude=fc_source.longitude,
        latitude=fc_source.latitude,
    )
    assert data.equals(data_ref_restricted)


def test_query_date_bbox(fc_source, data_ref):
    # Box not matching data
    bbox = (10, 10, 50, 50)
    data = fc_source.query_date(start=DATE_START, end=DATE_END, polygon=bbox)
    # If the bbox doesn't match the data, FCollections returns
    # a dataset with coords and data_vars but empty data
    assert data[LONGITUDE].size == 0
    assert data[LATITUDE].size == 0

    # Very large box
    bbox = (0, 0, 100, 100)
    data = fc_source.query_date(start=DATE_START, end=DATE_END, polygon=bbox)
    assert data.equals(data_ref)

    # Box selecting 2 lines (6 points)
    bbox = (0, 0, 6, 6)
    data = fc_source.query_date(start=DATE_START, end=DATE_END, polygon=bbox)
    assert data[LONGITUDE].size == 6
    assert data[LATITUDE].size == 6


def test_query_orbit(fc_source, l3_lr_ssh_basic_1):
    data = fc_source.query_orbit(cycle_number=NUM_CYCLE, pass_number=NUM_PASS)

    assert np.all(data[PASS_NUMBER].values == NUM_PASS)
    assert np.all(data[CYCLE_NUMBER].values == NUM_CYCLE)
    for field in fc_source.variables().values():
        assert field.name in data
        assert np.array_equal(
            data[field.name].values, l3_lr_ssh_basic_1[field.name].values
        )

    data = fc_source.query_orbit(
        cycle_number=NUM_CYCLE, pass_number=[NUM_PASS, NUM_PASS2]
    )

    assert np.array_equal(
        data[PASS_NUMBER].values,
        [NUM_PASS, NUM_PASS, NUM_PASS, NUM_PASS2, NUM_PASS2, NUM_PASS2],
    )
    assert np.all(data[CYCLE_NUMBER].values == NUM_CYCLE)

    fields = {TIME, LONGITUDE, LATITUDE, VAR1, VAR2}
    data = fc_source.query_orbit(
        cycle_number=NUM_CYCLE, pass_number=NUM_PASS, variables=fields
    )
    assert not (set(fields) - set(data.variables))


def test_query_orbit_concat_false(fc_source, l3_lr_ssh_basic_1, l3_lr_ssh_basic_2):
    data = fc_source.query_orbit(
        cycle_number=NUM_CYCLE, pass_number=[NUM_PASS, NUM_PASS2], concat=False
    )

    assert isinstance(data, list)
    assert len(data) == 2

    assert np.all(data[0][PASS_NUMBER].values == NUM_PASS)
    assert np.all(data[0][CYCLE_NUMBER].values == NUM_CYCLE)
    assert np.all(data[1][PASS_NUMBER].values == NUM_PASS2)
    assert np.all(data[1][CYCLE_NUMBER].values == NUM_CYCLE)

    for field in fc_source.variables().values():
        assert field.name in data[0]
        assert np.array_equal(
            data[0][field.name].values, l3_lr_ssh_basic_1[field.name].values
        )
        assert field.name in data[1]
        assert np.array_equal(
            data[1][field.name].values, l3_lr_ssh_basic_2[field.name].values
        )


def test_query_orbit_nadir_swath(fc_source, caplog):
    backend_kwargs = {
        "nadir": True,
        "swath": False,
    }
    caplog.set_level(logging.WARNING)

    fc_source.query_orbit(
        cycle_number=NUM_CYCLE, pass_number=NUM_PASS, backend_kwargs=backend_kwargs
    )

    assert any(
        "The nadir/swath parameters cannot be applied to this collection"
        in record.message
        for record in caplog.records
    )

    assert "nadir" not in backend_kwargs
    assert "swath" not in backend_kwargs


def test_query_orbit_polygon(fc_source, data_ref):
    # Polygon not matching data
    polygon = shp.Polygon(
        [
            (10, 10),
            (10, 50),
            (50, 50),
            (50, 10),
            (10, 10),
        ]
    )
    data = fc_source.query_orbit(
        cycle_number=NUM_CYCLE, pass_number=NUM_PASS, polygon=polygon
    )
    data_ref_restricted = restrict_to_polygon(
        data=data_ref,
        polygon=polygon,
        index=fc_source.index,
        longitude=fc_source.longitude,
        latitude=fc_source.latitude,
    )
    assert data.equals(data_ref_restricted)

    # Polygon matching data
    polygon = shp.Polygon(
        [
            (1, 1),
            (1, 5),
            (5, 5),
            (5, 1),
            (1, 1),
        ]
    )
    data = fc_source.query_orbit(
        cycle_number=NUM_CYCLE, pass_number=NUM_PASS, polygon=polygon
    )
    data_ref_restricted = restrict_to_polygon(
        data=data_ref,
        polygon=polygon,
        index=fc_source.index,
        longitude=fc_source.longitude,
        latitude=fc_source.latitude,
    )
    assert data.equals(data_ref_restricted)


def test_query_orbit_bbox(fc_source, data_ref):
    # Box not matching data
    bbox = (10, 10, 50, 50)
    data = fc_source.query_orbit(
        cycle_number=NUM_CYCLE, pass_number=NUM_PASS, polygon=bbox
    )
    # If the bbox doesn't match the data, FCollections returns
    # a dataset with coords and data_vars but empty data
    assert data[LONGITUDE].size == 0
    assert data[LATITUDE].size == 0

    # Very large box
    bbox = (0, 0, 100, 100)
    data = fc_source.query_orbit(
        cycle_number=NUM_CYCLE, pass_number=NUM_PASS, polygon=bbox
    )
    assert data.equals(data_ref)

    # Box selecting 2 lines (6 points)
    bbox = (0, 0, 6, 6)
    data = fc_source.query_orbit(
        cycle_number=NUM_CYCLE, pass_number=NUM_PASS, polygon=bbox
    )
    assert data[LONGITUDE].size == 6
    assert data[LATITUDE].size == 6


def test_query_date_gdf(fc_source, polygon_gdf):
    data = fc_source.query_date(start=DATE_START, end=DATE_END, polygon=polygon_gdf)
    assert data[LONGITUDE].size == 6
    assert data[LATITUDE].size == 6


def test_query_orbit_gdf(fc_source, polygon_gdf):
    data = fc_source.query_orbit(
        cycle_number=NUM_CYCLE, pass_number=NUM_PASS, polygon=polygon_gdf
    )
    assert data[LONGITUDE].size == 6
    assert data[LATITUDE].size == 6


def test_query_date_shapefile(fc_source, polygon_shp):
    data = fc_source.query_date(start=DATE_START, end=DATE_END, polygon=polygon_shp)
    assert data[LONGITUDE].size == 6
    assert data[LATITUDE].size == 6
    data = fc_source.query_date(
        start=DATE_START, end=DATE_END, polygon=str(polygon_shp)
    )
    assert data[LONGITUDE].size == 6
    assert data[LATITUDE].size == 6


def test_query_orbit_shapefile(fc_source, polygon_shp):
    data = fc_source.query_orbit(
        cycle_number=NUM_CYCLE, pass_number=NUM_PASS, polygon=polygon_shp
    )
    assert data[LONGITUDE].size == 6
    assert data[LATITUDE].size == 6
    data = fc_source.query_orbit(
        cycle_number=NUM_CYCLE, pass_number=NUM_PASS, polygon=str(polygon_shp)
    )
    assert data[LONGITUDE].size == 6
    assert data[LATITUDE].size == 6
