"""CLS Table sources."""

from __future__ import annotations

import dataclasses as dc
import logging
from contextlib import AbstractContextManager
from functools import lru_cache
from typing import Any

import cls_tables as cls_t
import numpy as np
import pandas as pd
import xarray as xr

from ..utilities import PolygonLike
from ._model import (
    DOC_PARAMETERS_ALTI_SOURCE,
    HALF_ORBIT_DTYPE,
    AltimetrySource,
    AltimetryVariable,
)

LOGGER = logging.getLogger(__name__)


@lru_cache(maxsize=10000)
def _pass_from_indices(
    orf: str, cycle_number: int, pass_number: int, method: str
) -> tuple[
    tuple[np.datetime64, float, float],
    tuple[np.datetime64, float, float],
    tuple[np.datetime64, float, float],
    int,
    int,
]:
    """Caching mechanism allowing to retrieve pass information from their cycle
    and pass number.

    Parameters
    ----------
    orf
        Name of the orf.
    cycle_nb
        Cycle number.
    pass_nb
        Pass number.
    method
        Searching method.

    Returns
    -------
    :
        Pass's information as a tuple containing the following elements:

        * Starting position
        * Equator position
        * Ending position
        * Cycle number
        * Pass number
    """
    with OrfContext(name=orf) as orf_h:
        if pass_number > orf_h.passes_per_cycle and method in [
            "before",
            "before_or_equal",
        ]:
            pass_number = orf_h.passes_per_cycle

        return orf_h.find_track_info_from_indices(
            cycle_number, pass_number, extrapolate=False, method=method
        )


@lru_cache(maxsize=10000)
def _pass_from_date(
    orf: str, date: np.datetime64, method: str
) -> tuple[
    tuple[np.datetime64, float, float],
    tuple[np.datetime64, float, float],
    tuple[np.datetime64, float, float],
    int,
    int,
]:
    """Caching mechanism allowing to retrieve pass information from their date.

    Parameters
    ----------
    orf
        Name of the orf.
    date
        Date to search for.
    method
        Searching method.

    Returns
    -------
    :
        Pass's information as a tuple containing the following elements:

        * Starting position
        * Equator position
        * Ending position
        * Cycle number
        * Pass number
    """
    with OrfContext(name=orf) as orf_h:
        return orf_h.find_track_info_from_date(date, extrapolate=False, method=method)


@dc.dataclass
class OrfContext(AbstractContextManager):
    """Context manager allowing to interact with an CLS Orf."""

    name: str | None = None

    _orf: cls_t.Orf | None = None

    def __enter__(self) -> cls_t.Orf:
        # noinspection PyArgumentList
        self._orf = cls_t.Orf(self.name, mode="r")

        return self._orf

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._orf.close()

        return True


@dc.dataclass
class TableContext(AbstractContextManager):
    """Context manager allowing to interact with a CLS Table."""

    name: str | None = None

    _table: cls_t.TableMeasure | None = None

    def __enter__(self) -> cls_t.TableMeasure:
        self._table = cls_t.TableMeasure(self.name)

        return self._table

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._table.close()

        return True


@dc.dataclass(kw_only=True)
class ClsTableSource(AltimetrySource[cls_t.TableMeasure]):
    __doc__ = f"""Source implementation for CLS tables.

    Parameters
    ----------
    name
        Table's name.
    orf
        ORF's name.

    {DOC_PARAMETERS_ALTI_SOURCE}
    """

    name: str
    orf: str | None = None

    time: str = "time"
    longitude: str = "LONGITUDE"
    latitude: str = "LATITUDE"
    index: str = "time"

    _orf_first_cycle: int = dc.field(init=False, default=0, repr=False)
    _orf_last_cycle: int = dc.field(init=False, default=0, repr=False)
    _orf_passes_per_cycle: int = dc.field(init=False, default=0, repr=False)

    @property
    def handler(self) -> cls_t.TableMeasure:
        return cls_t.TableMeasure(self.name)

    def variables(self) -> dict[str, AltimetryVariable]:
        if self._fields is not None:
            return self._fields

        with TableContext(name=self.name) as table:
            self._fields = {
                f.name: AltimetryVariable(
                    name=f.name, units=f.unit, description=f.description
                )
                for f in table.fields
            }
        self._fields[self.time] = AltimetryVariable(
            name=self.time, units="", description=self.time
        )

        return self._fields

    def period(self) -> tuple[np.datetime64, np.datetime64]:
        with TableContext(name=self.name) as table:
            first_date = table.find_next_date(
                cls_t.round_vanilla_datetime(date=np.datetime64("1900"))
            )
            last_date = table.find_previous_date(
                cls_t.round_vanilla_datetime(date=np.datetime64("2200"))
            )

        return np.datetime64(first_date), np.datetime64(last_date)

    def half_orbit_periods(
        self,
        half_orbit_min: tuple[int, int] | None = None,
        half_orbit_max: tuple[int, int] | None = None,
    ) -> pd.DataFrame:
        self._check_orf()
        self._set_orf_info()

        half_orbit_min_t: tuple[int, int] = half_orbit_min or (self._orf_first_cycle, 1)
        half_orbit_max_t: tuple[int, int] = half_orbit_max or (
            self._orf_last_cycle,
            self._orf_passes_per_cycle,
        )

        cycles_list = []
        pass_info = self.pass_from_indices(
            cycle_number=half_orbit_min_t[0],
            pass_number=half_orbit_min_t[1],
            method="after_or_equal",
        )

        while pass_info is not None and (
            pass_info[0] < half_orbit_max_t[0]
            or (
                pass_info[0] == half_orbit_max_t[0]
                and pass_info[1] <= half_orbit_max_t[1]
            )
        ):
            cycles_list.append((pass_info[0], pass_info[1], pass_info[2], pass_info[3]))
            pass_info = self.pass_from_indices(
                cycle_number=pass_info[0],
                pass_number=pass_info[1],
                method="after",
            )
        return pd.DataFrame(np.array(cycles_list, dtype=HALF_ORBIT_DTYPE))

    def query_date(
        self,
        start: np.datetime64,
        end: np.datetime64,
        variables: list[str] | None = None,
        polygon: PolygonLike | None = None,
        backend_kwargs: dict[str, Any] | None = None,
    ) -> xr.Dataset:
        variables = variables or list(self.variables())

        if backend_kwargs is None:
            backend_kwargs = {}

        with TableContext(name=self.name) as table:
            data = table.read_values_as_dataset(
                variables, start, end, include_end=True, **backend_kwargs
            )

        return self.restrict_to_polygon(data=data, polygon=polygon)

    def _query_cycle(
        self,
        cycle_number: list[int],
        variables: list[str] | None = None,
        polygon: PolygonLike | None = None,
        backend_kwargs: dict[str, Any] | None = None,
        concat: bool = True,
    ) -> xr.Dataset:
        self._check_orf()

        data = []
        for cycle_nb in cycle_number:
            pass_start_info = self.pass_from_indices(
                cycle_number=cycle_nb,
                pass_number=1,
                method="after_or_equal",
            )
            pass_end_info = self.pass_from_indices(
                cycle_number=cycle_nb + 1,
                pass_number=1,
                method="before",
            )

            if (
                pass_start_info is None
                or pass_end_info is None
                or pass_start_info[0] != cycle_nb
                or pass_end_info[0] != cycle_nb
            ):
                LOGGER.warning("Cycle %s not found in %s.", cycle_nb, self.orf)
                data.append(self._empty_dataset())
                continue

            data.append(
                self.query_date(
                    start=pass_start_info[2],
                    end=pass_end_info[3],
                    variables=variables,
                    polygon=polygon,
                    backend_kwargs=backend_kwargs,
                )
            )

        if not concat:
            return [self.restrict_to_polygon(data=ds, polygon=polygon) for ds in data]

        return self.restrict_to_polygon(
            data=xr.concat(data, dim=self.index), polygon=polygon
        )

    def query_orbit(
        self,
        cycle_number: int | list[int],
        pass_number: int | list[int] | None = None,
        variables: list[str] | None = None,
        polygon: PolygonLike | None = None,
        backend_kwargs: dict[str, Any] | None = None,
        concat: bool = True,
    ) -> xr.Dataset:
        self._check_orf()

        if isinstance(cycle_number, int):
            cycle_number = [cycle_number]

        if pass_number is None:
            return self._query_cycle(
                cycle_number=cycle_number,
                variables=variables,
                polygon=polygon,
                backend_kwargs=backend_kwargs,
                concat=concat,
            )

        if isinstance(pass_number, int):
            pass_number = [pass_number]

        data = []
        for cycle_nb in cycle_number:
            for pass_nb in pass_number:
                pass_info = self.pass_from_indices(
                    cycle_number=cycle_nb,
                    pass_number=pass_nb,
                    method="equal",
                )

                if pass_info is None:
                    LOGGER.warning(
                        "Cycle %s, pass %s not found in %s.",
                        cycle_nb,
                        pass_nb,
                        self.orf,
                    )
                    data.append(self._empty_dataset())
                    continue

                data.append(
                    self.query_date(
                        start=pass_info[2],
                        end=pass_info[3],
                        variables=variables,
                        polygon=polygon,
                        backend_kwargs=backend_kwargs,
                    )
                )

        if not concat:
            return [self.restrict_to_polygon(data=ds, polygon=polygon) for ds in data]

        return self.restrict_to_polygon(
            data=xr.concat(data, dim=self.index), polygon=polygon
        )

    def _check_orf(self):
        if self.orf is None:
            msg = "An orf must be set to use this function."
            raise ValueError(msg)

    def _set_orf_info(self):
        with OrfContext(name=self.orf) as orf:
            self._orf_first_cycle = orf.first_cycle
            self._orf_last_cycle = orf.last_cycle
            self._orf_passes_per_cycle = orf.passes_per_cycle

    def pass_from_indices(
        self, cycle_number: int, pass_number: int, method: str
    ) -> tuple[int, int, np.datetime64, np.datetime64] | None:
        self._check_orf()

        pass_info = _pass_from_indices(
            orf=self.orf,
            cycle_number=cycle_number,
            pass_number=pass_number,
            method=method,
        )

        if pass_info is None:
            return None

        first, _, last, cn, pn = pass_info

        return cn, pn, first[0], last[0]

    def pass_from_date(
        self, date: np.datetime64, method: str
    ) -> tuple[int, int, np.datetime64, np.datetime64] | None:
        self._check_orf()

        pass_info = _pass_from_date(orf=self.orf, date=date, method=method)

        if pass_info is None:
            return None

        first, _, last, cn, pn = pass_info

        return cn, pn, first[0], last[0]
