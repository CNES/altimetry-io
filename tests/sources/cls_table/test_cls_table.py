import logging

import numpy as np
import pytest

from cnes_alti_reader.sources import ClsTableSource
from tests.conftest import (
    DATE_END,
    DATE_START,
    DATE_STEP,
    DELTA_1_US,
    FIELD_1,
    FIELD_2,
    FIELD_3,
    INDEX,
    LATITUDE,
    LONGITUDE,
)

try:
    import cls_tables

    have_cls_tables = True
except ImportError:  # pragma: no cover
    cls_tables = None
    have_cls_tables = False

if not have_cls_tables:  # pragma: no cover
    pytest.skip("Skipping CLSTable reader tests", allow_module_level=True)


def test_handler(table_name):
    source = ClsTableSource(name=table_name)

    assert isinstance(source.handler, cls_tables.TableMeasure)


def test_variables(table_name):
    source = ClsTableSource(name=table_name)

    assert source.name == table_name

    assert source == ClsTableSource(name=table_name)

    table_fields = {INDEX, LONGITUDE, LATITUDE, FIELD_1, FIELD_2, FIELD_3}

    assert set(source.variables()) == set(table_fields)
    # Validating the fields caching
    assert set(source.variables()) == set(table_fields)


def test_period(dataset, table_name):
    times = dataset[INDEX].values
    source = ClsTableSource(name=table_name)

    assert source.period() == (times[0], times[-1])


def test_half_orbit_periods(table_name, orf_name):
    source = ClsTableSource(name=table_name, orf=orf_name)

    periods = source.half_orbit_periods()
    assert np.array_equal(
        periods[["cycle_number", "pass_number"]].to_numpy(),
        np.array([[1, 1], [1, 2], [3, 1], [3, 3]]),
    )

    periods = source.half_orbit_periods(ho_min=(3, 2))
    assert np.array_equal(
        periods[["cycle_number", "pass_number"]].to_numpy(), np.array([[3, 3]])
    )

    periods = source.half_orbit_periods(ho_max=(3, 2))
    assert np.array_equal(
        periods[["cycle_number", "pass_number"]].to_numpy(),
        np.array([[1, 1], [1, 2], [3, 1]]),
    )

    periods = source.half_orbit_periods(ho_min=(2, 1), ho_max=(3, 3))
    assert np.array_equal(
        periods[["cycle_number", "pass_number"]].to_numpy(), np.array([[3, 1], [3, 3]])
    )


@pytest.mark.parametrize(
    ("cycle_nb", "pass_nb", "method", "exp_res"),
    [
        (1, 1, "equal", (1, 1, 0)),
        (1, 1000, "before", (1, 2, 2)),
        (1, 100, "equal", None),
    ],
)
def test_pass_from_indices(
    dataset, table_name, orf_name, cycle_nb, pass_nb, method, exp_res
):
    source = ClsTableSource(name=table_name, orf=orf_name)

    res = source.pass_from_indices(cycle_nb=cycle_nb, pass_nb=pass_nb, method=method)

    if exp_res is None:
        assert res is None
        return

    cn, pn, start, _ = res

    assert cn == exp_res[0]
    assert pn == exp_res[1]
    assert start == dataset[INDEX].values[exp_res[2]]


@pytest.mark.parametrize(
    ("date", "method", "exp_res"),
    [
        (DATE_START, "equal", (1, 1)),
        (DATE_START, "before", None),
        (DATE_START + DATE_STEP, "equal", (1, 1)),
        (DATE_START + 2 * DATE_STEP, "equal", (1, 2)),
    ],
)
def test_pass_from_date(table_name, orf_name, date, method, exp_res):
    source = ClsTableSource(name=table_name, orf=orf_name)

    res = source.pass_from_date(date=date, method=method)

    if exp_res is None:
        assert res is None
        return

    cn, pn, _, _ = res

    assert cn == exp_res[0]
    assert pn == exp_res[1]


def test_query_dates(dataset, table_name):
    source = ClsTableSource(name=table_name)
    data = source.query_date(start=DATE_START, end=DATE_END)

    assert set(data.variables) == set(source.variables())

    for field in source.variables().values():
        assert field.name in data

        if field.name == INDEX:
            assert np.array_equal(data[field.name].values, dataset[field.name].values)
        else:
            assert np.allclose(
                data[field.name].values,
                dataset[field.name].values,
                equal_nan=True,
                rtol=0,
                atol=0,
            )

    data = source.query_periods(periods=[(DATE_START, DATE_END)], variables=[INDEX])
    assert np.array_equal(data[INDEX].values, dataset[INDEX].values)

    data = source.query_periods(
        periods=[
            (DATE_START, DATE_START + 2 * DATE_STEP),
            (DATE_START + 2 * DATE_STEP + DELTA_1_US, DATE_START + 3 * DATE_STEP),
            (DATE_START + 3 * DATE_STEP + DELTA_1_US, DATE_END),
        ],
        variables=[INDEX],
    )
    assert np.array_equal(data[INDEX].values, dataset[INDEX].values)


def test_query_orbits(caplog, dataset, table_name, orf_name):
    caplog.set_level(logging.WARNING)
    source = ClsTableSource(name=table_name)

    with pytest.raises(ValueError, match="An orf must be set"):
        source.query_orbit(cycles_nb=1)

    source = ClsTableSource(name=table_name, orf=orf_name)

    data = source.query_orbit(cycles_nb=1, variables=[INDEX])
    assert np.array_equal(data[INDEX].values, dataset[INDEX].values[:3])

    caplog.clear()

    data = source.query_orbit(cycles_nb=2, variables=[INDEX])
    assert "Cycle 2 not found in" in caplog.text
    assert data.sizes[INDEX] == 0

    caplog.clear()

    data = source.query_orbit(cycles_nb=[1, 2, 3, 4], variables=[INDEX])
    assert "Cycle 2 not found in" in caplog.text
    assert np.array_equal(data[INDEX].values, dataset[INDEX].values)

    data = source.query_orbit(cycles_nb=1, passes_nb=1, variables=[INDEX])
    assert np.array_equal(data[INDEX].values, dataset[INDEX].values[:2])

    caplog.clear()

    data = source.query_orbit(cycles_nb=[1, 6], passes_nb=[1, 3], variables=[INDEX])
    assert "Cycle 1, pass 3 not found in" in caplog.text
    assert "Cycle 6, pass 1 not found in" in caplog.text
    assert "Cycle 6, pass 3 not found in" in caplog.text
    assert np.array_equal(data[INDEX].values, dataset[INDEX].values[:2])
