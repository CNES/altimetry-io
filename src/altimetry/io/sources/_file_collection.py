from __future__ import annotations

import dataclasses as dc
import enum
import logging
from typing import Any

import fcollections.core as fc_core
import fcollections.implementations as fc_impl
import fcollections.missions as fc_mis
import fcollections.time as fc_time
import fsspec
import numpy as np
import pandas as pd
import xarray as xr

from ..utilities import (
    PolygonLike,
    normalize_enum,
    normalize_file_system,
    polygon_bounding_box,
)
from ._model import (
    CONST_CYCLE_NUMBER,
    CONST_END_TIME,
    CONST_PASS_NUMBER,
    CONST_START_TIME,
    DOC_PARAMETERS_ALTI_SOURCE,
    HALF_ORBIT_DTYPE,
    AltimetrySource,
    AltimetryVariable,
)

FC_CONST_TIME = "time"
FC_CONST_CYCLE_NUMBER = "cycle_number"
FC_CONST_PASS_NUMBER = "pass_number"

LOGGER = logging.getLogger(__name__)


class FCollectionType(enum.Enum):
    """Different types of file's collections."""

    SWOT_L2_LR_SSH = enum.auto()
    SWOT_L3_LR_SSH = enum.auto()
    SWOT_L3_LR_WIND_WAVE = enum.auto()
    NADIR_L2 = enum.auto()
    NADIR_L3 = enum.auto()

    def fc_database(self) -> type[fc_core.FilesDatabase]:
        known_collections = {
            FCollectionType.SWOT_L2_LR_SSH: fc_impl.NetcdfFilesDatabaseSwotLRL2,
            FCollectionType.SWOT_L3_LR_SSH: fc_impl.NetcdfFilesDatabaseSwotLRL3,
            FCollectionType.SWOT_L3_LR_WIND_WAVE: fc_impl.NetcdfFilesDatabaseSwotLRWW,
            FCollectionType.NADIR_L2: fc_impl.NetcdfFilesDatabaseL2Nadir,
            FCollectionType.NADIR_L3: fc_impl.NetcdfFilesDatabaseL3Nadir,
        }

        return known_collections[self]


@dc.dataclass(kw_only=True)
class FileCollectionSource(AltimetrySource[fc_core.FilesDatabase]):
    __doc__ = f"""Source implementation for sets of files.

    Parameters
    ----------
    path
        File collection's path.

    {DOC_PARAMETERS_ALTI_SOURCE}
    """

    path: str
    fs: fsspec.AbstractFileSystem | str | None = dc.field(default=None, compare=False)
    ftype: FCollectionType

    time: str = "time"
    longitude: str = "longitude"
    latitude: str = "latitude"
    # TODO: Do not set this?
    index: str = "num_lines"

    subset: fc_impl.ProductSubset | str | None = None
    mission: fc_mis.MissionsPhases | str | None = None
    version: str | None = None

    _database: fc_core.FilesDatabase = dc.field(repr=False, init=False, compare=False)
    _initialized: bool = dc.field(repr=False, init=False, compare=False, default=False)

    _with_ho: bool = dc.field(repr=False, init=False, compare=False, default=False)
    _data: pd.DataFrame = dc.field(repr=False, init=False, compare=False)

    def __post_init__(self):
        self.fs = normalize_file_system(fs=self.fs)
        self.ftype = normalize_enum(self.ftype, FCollectionType)

        self._database = self.ftype.fc_database()(path=self.path, fs=self.fs)

    @property
    def handler(self) -> fc_core.FilesDatabase:
        return self._database

    def _request_kwargs(self) -> dict[str, Any]:
        request_kw = {}

        if self.subset is not None:
            request_kw["subset"] = self.subset

        if self.version is not None:
            request_kw["version"] = self.version

        if self.mission is not None:
            request_kw["mission"] = self.mission

        return request_kw

    def _initialize(self) -> None:
        if self._initialized:
            return

        pd_files: pd.DataFrame = self._database.list_files(
            sort=True, **self._request_kwargs()
        )

        periods = np.array([[p.start, p.stop] for p in pd_files[FC_CONST_TIME].values])

        if FC_CONST_CYCLE_NUMBER in pd_files.columns:
            self._with_ho = True

            ho_data = pd_files[[FC_CONST_CYCLE_NUMBER, FC_CONST_PASS_NUMBER]].values

            data = np.empty(ho_data.shape[0], dtype=HALF_ORBIT_DTYPE)
            data[CONST_CYCLE_NUMBER] = ho_data[:, 0]
            data[CONST_PASS_NUMBER] = ho_data[:, 1]
            data[CONST_START_TIME] = periods[:, 0]
            data[CONST_END_TIME] = periods[:, 1]

            self._data = pd.DataFrame(data)
        else:
            self._with_ho = False

            data = np.empty(
                periods.shape[0],
                dtype=HALF_ORBIT_DTYPE[[CONST_START_TIME, CONST_END_TIME]],
            )
            data[CONST_START_TIME] = periods[:, 0]
            data[CONST_END_TIME] = periods[:, 1]

            self._data = pd.DataFrame(data)

        self._initialized = True

    def variables(self) -> dict[str, AltimetryVariable]:
        if self._fields is not None:
            return self._fields

        self._fields = {}

        info: fc_core.GroupMetadata = self._database.variables_info(
            **self._request_kwargs()
        )

        if info.subgroups:
            msg = "This collection contains subgroups. This is not supported yet."
            raise ValueError(msg)

        for variable in info.variables:
            self._fields[variable.name] = AltimetryVariable(
                name=variable.name,
                units=variable.attributes.get("units", ""),
                description=variable.attributes.get("comment", ""),
            )

        return self._fields

    def period(self) -> tuple[np.datetime64, np.datetime64]:
        self._initialize()

        return (
            np.min(self._data[CONST_START_TIME].values),
            np.max(self._data[CONST_END_TIME].values),
        )

    def half_orbit_periods(
        self,
        half_orbit_min: tuple[int, int] | None = None,
        half_orbit_max: tuple[int, int] | None = None,
    ) -> pd.DataFrame:
        self._check_orf()

        return self._data

    def query_date(
        self,
        start: np.datetime64,
        end: np.datetime64,
        variables: list[str] | None = None,
        polygon: PolygonLike | None = None,
        backend_kwargs: dict[str, Any] | None = None,
    ) -> xr.Dataset:
        backend_kwargs = backend_kwargs or {}

        if "nadir" in backend_kwargs or "swath" in backend_kwargs:
            LOGGER.warning(
                "The nadir/swath parameters cannot be applied to this collection."
                " Please open a NADIR_L2/NADIR_L3 collection to query nadir data."
            )
            backend_kwargs.pop("nadir", None)
            backend_kwargs.pop("swath", None)

        request_kwargs = {
            **self._request_kwargs(),
            **backend_kwargs,
        }
        is_bbox = isinstance(polygon, tuple)

        polygon = polygon_bounding_box(polygon=polygon)

        # FCollections doesn't allow bbox=None as kwarg
        if polygon is not None:
            request_kwargs["bbox"] = polygon

        data = self._database.query(
            time=fc_time.Period(start=start, stop=end),
            selected_variables=variables,
            **request_kwargs,
        )
        if data is None:
            return self._empty_dataset()

        # Deactivate polygon restriction if the polygon was a bbox
        # -> the selection was done by fcollections
        if is_bbox:
            polygon = None

        return self.restrict_to_polygon(data=data, polygon=polygon)

    def query_orbit(
        self,
        cycle_number: int | list[int],
        pass_number: int | list[int] | None = None,
        variables: list[str] | None = None,
        polygon: PolygonLike | None = None,
        backend_kwargs: dict[str, Any] | None = None,
        concat: bool = True,
    ) -> xr.Dataset:
        backend_kwargs = backend_kwargs or {}

        if "nadir" in backend_kwargs or "swath" in backend_kwargs:
            LOGGER.warning(
                "The nadir/swath parameters cannot be applied to this collection."
                " Please open a NADIR_L2/NADIR_L3 collection to query nadir data."
            )
            backend_kwargs.pop("nadir", None)
            backend_kwargs.pop("swath", None)

        self._check_orf()

        request_kwargs = {
            **self._request_kwargs(),
            **backend_kwargs,
        }
        is_bbox = isinstance(polygon, tuple)

        polygon = polygon_bounding_box(polygon=polygon)

        # FCollections doesn't allow bbox=None as kwarg
        if polygon is not None:
            request_kwargs["bbox"] = polygon

        data = self._database.query(
            cycle_number=cycle_number,
            pass_number=pass_number,
            selected_variables=variables,
            **request_kwargs,
        )
        if data is None:
            return self._empty_dataset()

        # Deactivate polygon restriction if the polygon was a bbox
        # -> the selection was done by fcollections
        if is_bbox:
            polygon = None

        return self.restrict_to_polygon(data=data, polygon=polygon)

    def _check_orf(self):
        self._initialize()

        if not self._with_ho:
            msg = "This collection does not contain half-orbit information."
            raise ValueError(msg)
