from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    import dask.array as da_t
    import xarray as xr_t


def data_slice(
    values: np.ndarray | da_t.Array, val_min: Any | None, val_max: Any | None
) -> slice:
    """Compute a slice corresponding to values where val_min <= values <
    val_max.

    Creates a slice object for subsetting an array of sorted values based on the
    given minimum and maximum bounds. The upper bound is exclusive.

    Parameters
    ----------
    values
        Sorted array from which to compute the slice
    val_min
        Minimum value (inclusive) or None for no lower bound
    val_max
        Maximum value (exclusive) or None for no upper bound

    Returns
    -------
    ;
        Slice object corresponding to the subset of values

    Raises
    ------
    ValueError
        If val_min is greater than val_max
    """
    # Nothing to do
    if val_min is None and val_max is None:
        return slice(None, None)

    if val_min is None:
        val_ar = [val_max]
    elif val_max is None:
        val_ar = [val_min]
    else:
        if val_max < val_min:
            msg = f"start ({val_min}) cannot be greater than end ({val_max})"
            raise ValueError(msg)

        val_ar = [val_min, val_max]

    if isinstance(values, np.ndarray):
        idx = list(np.searchsorted(values, np.array(val_ar)))
    else:
        import dask.array as da

        idx = list(da.searchsorted(values, da.array(val_ar)).compute())

    if val_min is None:
        idx = [None, idx[0]]
    elif val_max is None:
        idx = [idx[0], None]

    return slice(*idx)


def data_slice_include_end(
    values: np.ndarray | da_t.Array,
    val_min: Any | None,
    val_max: Any | None,
) -> slice:
    """Compute a slice corresponding to values where val_min <= values <=
    val_max.

    Creates a slice object for subsetting an array of sorted values based on the
    given minimum and maximum bounds. Both bounds are inclusive.

    Parameters
    ----------
    values
        Sorted array from which to compute the slice
    val_min
        Minimum value (inclusive) or None for no lower bound
    val_max
        Maximum value (inclusive) or None for no upper bound

    Returns
    -------
    :
        Slice object corresponding to the subset of values

    Raises
    ------
    ValueError
        If val_min is greater than val_max
    """
    # Nothing to do
    if val_min is None and val_max is None:
        return slice(None, None)

    if val_min is not None and val_max is not None and val_max < val_min:
        msg = f"start ({val_min}) cannot be greater than end ({val_max})"
        raise ValueError(msg)

    if isinstance(values, np.ndarray):
        fun = np.searchsorted
        array = np.array
        compute = False
    else:
        import dask.array as da

        fun = da.searchsorted
        array = da.array
        compute = True

    idx_1 = idx_2 = None

    if val_min is not None:
        idx_1 = fun(values, array([val_min]), side="left")[0]

        if compute:
            idx_1 = idx_1.compute()

    if val_max is not None:
        idx_2 = fun(values, array([val_max]), side="right")[0]

        if compute:
            idx_2 = idx_2.compute()

    return slice(idx_1, idx_2)


def dataset_select_field_1d(
    data: xr_t.Dataset, field: str, values: tuple[Any, Any], include_end: bool = False
) -> xr_t.Dataset:
    """Select a dataset subset based on value range in a one-dimensional field.

    Filters the dataset to include only elements where field values are within the
    specified range. The field must be one-dimensional and sorted. Selection follows
    the pattern: min <= field < max (or min <= field <= max if include_end is True).

    Parameters
    ----------
    data
        Dataset to filter.
    field
        Name of the field to use for selection (must be sorted and one-dimensional).
    values
        Tuple containing (min_value, max_value).
    include_end
        If True, include the upper bound in the selection.

    Returns
    -------
    data
        Filtered dataset subset.
    """
    start, end = values

    # Nothing to do
    if start is None and end is None:
        return data

    dims = data[field].dims

    if len(dims) > 1:
        msg = f"Cannot select on multiple dimensions: {dims}"
        raise ValueError(msg)

    index = dims[0]

    f_data = data[field].data

    if include_end:
        idx_slice = data_slice_include_end(values=f_data, val_min=start, val_max=end)
    else:
        idx_slice = data_slice(values=f_data, val_min=start, val_max=end)

    data_sel = data.isel({index: idx_slice})

    return data_sel
