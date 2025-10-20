import pathlib as pl

import numpy as np
import pandas as pd
import pytest
import shapely as shp

from cnes_alti_reader.sources import ScCollectionSource
from cnes_alti_reader.utilities import restrict_to_polygon

try:
    import swot_calval as sc
    import swot_calval.io as sc_io
    import swot_calval.natural_earth as sc_ne

    have_sc = True
except ImportError:
    have_sc = False
    sc = None
    sc_io = None
    sc_ne = None

if not have_sc:
    pytest.skip("Skipping swot_calval related tests", allow_module_level=True)


@pytest.fixture(scope="session")
def swot_collection() -> pl.Path:
    """Get the SWOT collection."""

    sc_root = pl.Path(sc.__file__).parent
    sc_data = sc_root / "tests" / "data"
    sc_test_collection = sc_data / "collection"

    return sc_test_collection


def test_handler(swot_collection):
    source = ScCollectionSource(path=swot_collection)

    assert isinstance(source.handler, sc_io.Collection)


def test_variables(swot_collection):
    source = ScCollectionSource(path=swot_collection)
    collection = sc_io.open_collection(folder=swot_collection)

    assert source.path == str(swot_collection)

    assert source == ScCollectionSource(path=swot_collection)

    assert set(source.variables()) == set(collection.variables())
    # Validating the fields caching
    assert set(source.variables()) == set(collection.variables())


def test_period(swot_collection):
    source = ScCollectionSource(path=swot_collection)
    collection = sc_io.open_collection(folder=swot_collection)

    assert source.period() == collection.period()


def test_half_orbit_periods(swot_collection):
    source = ScCollectionSource(path=swot_collection)
    collection = sc_io.open_collection(folder=swot_collection)

    ho_ref = pd.DataFrame(collection.half_orbit_periods())

    ho_periods = source.half_orbit_periods()

    assert ho_periods.equals(ho_ref)

    ho_periods = source.half_orbit_periods(ho_min=(1, 3))
    assert ho_periods.equals(ho_ref[2:].reset_index(drop=True))

    ho_periods = source.half_orbit_periods(ho_max=(1, 3))
    assert ho_periods.equals(ho_ref[:3].reset_index(drop=True))

    ho_periods = source.half_orbit_periods(ho_min=(2, 1))
    assert len(ho_periods) == 0

    ho_periods = source.half_orbit_periods(ho_max=(1, 0))
    assert len(ho_periods) == 0

    ho_periods = source.half_orbit_periods(ho_min=(1, 3), ho_max=(1, 3))
    assert ho_periods.equals(ho_ref[2:3].reset_index(drop=True))


def test_query_date(swot_collection):
    source = ScCollectionSource(path=swot_collection)
    collection = sc_io.open_collection(folder=swot_collection)

    fields = [source.time, source.longitude, source.latitude]
    start, end = collection.period()

    data = source.query_date(start=start, end=end, variables=fields)

    assert not (set(fields) - set(data.variables))
    assert data[source.time].values[0] == start
    assert data[source.time].values[-1] == end

    start = start + np.timedelta64(1, "m")
    end = end - np.timedelta64(1, "m")

    data = source.query_date(start=start, end=end, variables=fields)

    assert not (set(fields) - set(data.variables))
    assert data[source.time].values[0] >= start
    assert data[source.time].values[-1] <= end

    data = source.query_date(
        start=np.datetime64("1960"), end=np.datetime64("1961"), variables=fields
    )
    assert data.sizes[source.index] == 0


@pytest.mark.parametrize("cycles", [0, 1, 2, [1, 2]])
def test_query_cycle(swot_collection, cycles):
    source = ScCollectionSource(path=swot_collection)
    collection = sc_io.open_collection(folder=swot_collection)

    fields = [source.time, source.longitude, source.latitude]

    data_ref = collection.query(cycle_numbers=cycles, selected_variables=fields)
    data = source.query_orbit(cycles_nb=cycles, variables=fields)

    if data_ref is None:
        assert data.sizes[source.index] == 0
    else:
        assert data.equals(data_ref.to_xarray())


@pytest.mark.parametrize(
    ("cycles", "passes"),
    [(0, 0), (1, 0), (1, [0, 2, 4]), ([1, 2], [0, 2, 4]), (2, 1)],
)
def test_query_cycle_pass(swot_collection, cycles, passes):
    source = ScCollectionSource(path=swot_collection)
    collection = sc_io.open_collection(folder=swot_collection)

    fields = [source.time, source.longitude, source.latitude]

    data_ref = collection.query(
        cycle_numbers=cycles, pass_numbers=passes, selected_variables=fields
    )
    data = source.query_orbit(cycles_nb=cycles, passes_nb=passes, variables=fields)

    if data_ref is None:
        assert data.sizes[source.index] == 0
    else:
        assert data.equals(data_ref.to_xarray())


def test_query_polygon(swot_collection):
    source = ScCollectionSource(path=swot_collection)

    fields = [source.time, source.longitude, source.latitude]
    polygon = shp.Polygon(
        [
            (10, 10),
            (10, 50),
            (50, 50),
            (50, 10),
            (10, 10),
        ]
    )

    data_restricted = restrict_to_polygon(
        data=source.query_orbit(cycles_nb=1, variables=fields),
        polygon=polygon,
        index=source.index,
        longitude=source.longitude,
        latitude=source.latitude,
    )
    data = source.query_orbit(cycles_nb=1, variables=fields, polygon=polygon)

    assert data.equals(data_restricted)
