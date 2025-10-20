import dask.array as da
import numpy as np
import pytest
import xarray as xr

from cnes_alti_reader.utilities import (
    data_slice,
    data_slice_include_end,
    dataset_select_field_1d,
)


@pytest.mark.parametrize("array_gen", [np.array, da.from_array])
@pytest.mark.parametrize(
    ("data", "val_range", "res_range"),
    [
        ([0, 1, 2], (0, 1), slice(0, 1)),
        ([0, 1, 2], (0, 1.2), slice(0, 2)),
        ([0, 1, 2], (None, None), slice(None, None)),
        ([0, 1, 2], (None, 2), slice(None, 2)),
        ([0, 1, 2], (1, None), slice(1, None)),
        ([0, 1, 2], (-10, 15), slice(0, 3)),
    ],
)
def test_data_slice(array_gen, data, val_range, res_range):
    res = data_slice(values=array_gen(data), val_min=val_range[0], val_max=val_range[1])
    r = res_range

    assert res == r


@pytest.mark.parametrize("array_gen", [np.array, da.from_array])
@pytest.mark.parametrize(
    ("data", "val_range", "res_range"),
    [
        ([0, 1, 2], (0, 1), slice(0, 2)),
        ([0, 1, 2], (0, 1.2), slice(0, 2)),
        ([0, 1, 2], (None, None), slice(None, None)),
        ([0, 1, 2], (None, 2), slice(None, 3)),
        ([0, 1, 2], (1, None), slice(1, None)),
        ([0, 1, 2], (-10, 15), slice(0, 3)),
    ],
)
def test_data_slice_include_end(array_gen, data, val_range, res_range):
    res = data_slice_include_end(
        values=array_gen(data), val_min=val_range[0], val_max=val_range[1]
    )
    r = res_range

    assert res == r


@pytest.mark.parametrize("array_gen", [np.array, da.from_array])
@pytest.mark.parametrize("include_end", [False, True])
@pytest.mark.parametrize(
    ("data", "dims", "val_range", "res_range", "res_range_end"),
    [
        ([0, 1, 2], "x", (0, 1), (0, 0), (0, 1)),
        ([0, 1, 2], "x", (0, 1.2), (0, 1), (0, 1)),
        ([0, 1, 2], "x", (None, None), (0, 2), (0, 2)),
        ([0, 1, 2], "x", (None, 2), (0, 1), (0, 2)),
        ([0, 1, 2], "x", (1, None), (1, 2), (1, 2)),
        ([0, 1, 2], "x", (-10, 15), (0, 2), (0, 2)),
    ],
)
def test_dataset_select_field_1d(
    array_gen, include_end, data, dims, val_range, res_range, res_range_end
):
    idx = "field"
    ds = xr.Dataset(data_vars={idx: (dims, array_gen(data))})

    res = dataset_select_field_1d(
        data=ds,
        field=idx,
        values=val_range,
        include_end=include_end,
    )
    values = res[idx].values

    if include_end:
        r = res_range_end
    else:
        r = res_range

    assert values[0] == r[0]
    assert values[-1] == r[1]


@pytest.mark.parametrize("include_end", [False, True])
@pytest.mark.parametrize(
    ("data", "dims", "val_range", "error"),
    [
        ([0, 1, 2], "x", (3, 1), "cannot be greater than end"),
        (
            [[0, 1, 2], [0, 1, 2]],
            ("x", "y"),
            (0, None),
            "Cannot select on multiple dimensions",
        ),
    ],
)
def test_dataset_select_field_1d_error(include_end, data, dims, val_range, error):
    idx = "field"
    ds = xr.Dataset(data_vars={idx: (dims, np.array(data))})

    with pytest.raises(ValueError, match=error):
        dataset_select_field_1d(
            data=ds, field=idx, values=val_range, include_end=include_end
        )
