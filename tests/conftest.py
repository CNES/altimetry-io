import logging
import warnings
import weakref

import dask
import distributed
import numpy as np
import pytest
import xarray as xr

INDEX = "time"
LONGITUDE = "LONGITUDE"
LATITUDE = "LATITUDE"
FIELD_1 = "FIELD_1"
FIELD_2 = "FIELD_2"
FIELD_3 = "FIELD_3"

DATE_START = np.datetime64("2020-01-01")
DATE_END = np.datetime64("2020-01-05")
DATE_STEP = np.timedelta64(1, "D")

DELTA_1_NS = np.timedelta64(1, "ns")
DELTA_1_US = np.timedelta64(1, "us")

F1_VAL = np.array([10, 20, 30, 40, 50])
F2_VAL = np.arange(start=10, stop=10.5, step=0.1, dtype=np.float32)
F3_VAL = np.array([10, np.nan, 30, 40, 50])


@pytest.fixture(scope="session")
def dataset():
    ds = xr.Dataset(
        data_vars={
            FIELD_1: xr.DataArray(data=F1_VAL, dims=INDEX),
            FIELD_2: xr.DataArray(data=F2_VAL, dims=INDEX),
            FIELD_3: xr.DataArray(data=F3_VAL, dims=INDEX),
            LONGITUDE: xr.DataArray(data=[-160, -80, 0, 80, 160], dims=INDEX),
            LATITUDE: xr.DataArray(data=[-80, -40, 0, 40, 80], dims=INDEX),
            INDEX: xr.DataArray(
                data=np.arange(
                    start=DATE_START,
                    stop=DATE_END + DATE_STEP,
                    step=DATE_STEP,
                    dtype="datetime64[D]",
                ).astype("datetime64[ns]"),
                dims=INDEX,
            ),
        }
    )
    ds = ds.chunk({INDEX: 3})

    return ds


@pytest.fixture(scope="session")
def scheduler_file(tmp_path_factory) -> str:
    """Start a dask cluster and save its information into a scheduler file."""

    dir_tmp = tmp_path_factory.mktemp("scheduler")
    sc_file = dir_tmp / "scheduler.json"

    if sc_file.exists():
        return str(sc_file)

    # Use the root path of the test session for the dask worker space
    dask_worker = dir_tmp / "dask_worker_space"
    dask.config.set(temporary_directory=str(dask_worker))

    logging.debug("Dask local cluster starting")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)

        cluster = distributed.LocalCluster(
            protocol="tcp://",
            n_workers=1,
            threads_per_worker=1,
            processes=False,
            dashboard_address=None,
        )

    def teardown():
        """Stop the cluster and remove the scheduler file."""
        logging.debug("Dask local cluster ending")
        cluster.close()
        logging.debug("Dask local cluster ended")
        if sc_file.exists():
            sc_file.unlink(missing_ok=True)

    weakref.finalize(cluster, teardown)

    # Making sure we can connect to the cluster.
    with distributed.Client(cluster) as client:
        client.write_scheduler_file(sc_file)
        client.wait_for_workers(1)

    logging.debug("Dask local cluster started")
    return str(sc_file)


@pytest.fixture
def dask_client(scheduler_file) -> distributed.Client:
    """Connect a Dask client to the cluster."""
    with distributed.Client(scheduler_file=scheduler_file) as client:
        yield client
