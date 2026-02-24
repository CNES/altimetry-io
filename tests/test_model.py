from __future__ import annotations

import dataclasses as dc
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import shapely as shp
import xarray as xr

from altimetry.io import AltimetryData
from altimetry.io.sources import AltimetrySource, AltimetryVariable
from altimetry.io.utilities import PolygonLike
from tests.conftest import DATE_END, DATE_START, INDEX, LATITUDE, LONGITUDE

if TYPE_CHECKING:
    pass


@dc.dataclass(kw_only=True)
class FakeSource(AltimetrySource[int]):
    index: str = dc.field(init=False)

    def __post_init__(self):
        self.index = self.time

    @property
    def handler(self) -> int:
        return 1

    def variables(self) -> dict[str, AltimetryVariable]:
        return {
            "a": AltimetryVariable(name="a"),
            "b": AltimetryVariable(name="b"),
            self.time: AltimetryVariable(name=self.time, description="Time coordinate"),
            self.longitude: AltimetryVariable(
                name=self.longitude, description="Longitude coordinate"
            ),
            self.latitude: AltimetryVariable(
                name=self.latitude, description="Latitude coordinate"
            ),
        }

    def period(self) -> tuple[np.datetime64, np.datetime64]:
        return np.datetime64("2020-01-01"), np.datetime64("2020-01-02")

    def half_orbit_periods(
        self,
        half_orbit_min: tuple[int, int] | None = None,
        half_orbit_max: tuple[int, int] | None = None,
    ) -> pd.DataFrame:
        return pd.DataFrame([1, 2, 3])

    def query_date(
        self,
        start: np.datetime64,
        end: np.datetime64,
        variables: list[str] | None = None,
        polygon: PolygonLike | None = None,
        backend_kwargs: dict[str, Any] | None = None,
    ) -> xr.Dataset:
        return xr.Dataset()

    def query_orbit(
        self,
        cycle_number: int | list[int],
        pass_number: int | list[int] | None = None,
        variables: list[str] | None = None,
        polygon: PolygonLike | None = None,
        backend_kwargs: dict[str, Any] | None = None,
        concat: bool = True,
    ) -> xr.Dataset:
        if not concat:
            return [xr.Dataset()]
        return xr.Dataset()


SOURCE = FakeSource(time=INDEX, longitude=LONGITUDE, latitude=LATITUDE)


def test_handler():
    alti_data = AltimetryData(source=SOURCE)

    assert alti_data.handler == 1


def test_variables():
    alti_data = AltimetryData(source=SOURCE)

    assert alti_data.variables() == SOURCE.variables()


def test_show_variables():
    alti_data = AltimetryData(source=SOURCE)

    assert set(alti_data.show_variables()["name"]) == set(SOURCE.variables())

    assert set(alti_data.show_variables(containing="LoNgItUde")["name"]) == {
        SOURCE.longitude
    }

    assert set(alti_data.show_variables(containing="coord")["name"]) == {
        SOURCE.time,
        SOURCE.longitude,
        SOURCE.latitude,
    }

    assert set(alti_data.show_variables(containing="___")["name"]) == set()


def test_periods():
    data = AltimetryData(source=SOURCE)
    assert data.period() == SOURCE.period()


def test_half_orbit_periods():
    data = AltimetryData(source=SOURCE)
    assert data.half_orbit_periods().equals(SOURCE.half_orbit_periods())


def test_query():
    alti_data = AltimetryData(source=SOURCE)
    assert alti_data.query(periods=(DATE_START, DATE_END)).equals(
        SOURCE.query(periods=(DATE_START, DATE_END))
    )
    assert alti_data.query(periods=[(DATE_START, DATE_END)]).equals(
        SOURCE.query(periods=[(DATE_START, DATE_END)])
    )
    assert alti_data.query(
        periods=[(DATE_START, DATE_END), (DATE_START, DATE_END)]
    ).equals(SOURCE.query(periods=[(DATE_START, DATE_END), (DATE_START, DATE_END)]))
    # testing with concat = False
    data_ref = alti_data.query(
        periods=[(DATE_START, DATE_END), (DATE_START, DATE_END)], concat=False
    )
    data = SOURCE.query(
        periods=[(DATE_START, DATE_END), (DATE_START, DATE_END)], concat=False
    )
    assert isinstance(data, list)
    assert all(d1.equals(d2) for d1, d2 in zip(data, data_ref, strict=True))


def test_query_date():
    alti_data = AltimetryData(source=SOURCE)
    assert alti_data.query(periods=[(DATE_START, DATE_END)]).equals(
        SOURCE.query_date(start=DATE_START, end=DATE_END)
    )


def test_query_orbit():
    alti_data = AltimetryData(source=SOURCE)
    assert alti_data.query_orbit(cycle_number=3).equals(
        SOURCE.query_orbit(cycle_number=3)
    )

    assert alti_data.query_orbit(cycle_number=3, pass_number=1).equals(
        SOURCE.query_orbit(cycle_number=3, pass_number=1)
    )
    # testing with concat = False
    data_ref = alti_data.query_orbit(cycle_number=3, concat=False)
    data = SOURCE.query_orbit(cycle_number=3, concat=False)
    assert isinstance(data, list)
    assert all(d1.equals(d2) for d1, d2 in zip(data, data_ref, strict=True))


def test_restrict_to_polygon(dataset):
    polygon = shp.Polygon(
        [
            (-180.05, -60.05),
            (-180.05, 40.05),
            (180.05, 40.05),
            (180.05, -60.05),
            (-180.05, -60.05),
        ]
    )
    data = SOURCE.restrict_to_polygon(data=dataset, polygon=polygon)

    assert np.array_equal(data[LONGITUDE].values, [-80, 0, 80])
    assert np.array_equal(data[LATITUDE].values, [-40, 0, 40])
