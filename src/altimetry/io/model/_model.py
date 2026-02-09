from __future__ import annotations

import dataclasses as dc
import logging
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import xarray as xr

from altimetry.io.sources import AltimetrySource, AltimetryVariable

from ..utilities import PolygonLike

if TYPE_CHECKING:
    pass


LOGGER = logging.getLogger(__name__)


@dc.dataclass
class AltimetryData:
    source: AltimetrySource

    @property
    def handler(self) -> Any:
        return self.source.handler

    def variables(self) -> dict[str, AltimetryVariable]:
        """Variables contained in this altimetric data source."""
        return self.source.variables()

    def show_variables(self, containing: str | None = None) -> pd.DataFrame:
        """Display variables containing a given string as a DataFrame.

        Parameters
        ----------
        containing
            String contained in variable names or descriptions.

        Returns
        -------
        :
            List of variables as a DataFrame.
        """
        if containing is not None:
            containing = containing.upper()
            variables = {
                v.name: v
                for v in self.variables().values()
                if (containing in v.name.upper())
                or (containing in v.description.upper())
            }
        else:
            variables = self.variables()

        if not variables:
            data = pd.DataFrame([], columns=["name", "description", "units"])
        else:
            data = pd.DataFrame(
                np.array(
                    [[v.name, v.description, v.units] for v in variables.values()]
                ),
                columns=["name", "description", "units"],
            )

        return data

    def period(self) -> tuple[np.datetime64, np.datetime64]:
        """Period covered by this altimetric data source."""
        return self.source.period()

    def half_orbit_periods(
        self,
        half_orbit_min: tuple[int, int] | None = None,
        half_orbit_max: tuple[int, int] | None = None,
    ) -> pd.DataFrame:
        """Half orbit periods covered by this altimetric data source.

        Parameters
        ----------
        half_orbit_min
            Tuple of (cycle_number, pass_number) for the minimum half orbit.
        half_orbit_max
            Tuple of (cycle_number, pass_number) for the maximum half orbit.

        Returns
        -------
        :
            Set of half orbit periods.
        """
        return self.source.half_orbit_periods(
            half_orbit_min=half_orbit_min, half_orbit_max=half_orbit_max
        )

    def query(
        self,
        periods: tuple[np.datetime64, np.datetime64]
        | list[tuple[np.datetime64, np.datetime64]],
        variables: list[str] | None = None,
        polygon: PolygonLike | None = None,
        backend_kwargs: dict[str, Any] | None = None,
    ) -> xr.Dataset:
        """Query data between two dates.

        Parameters
        ----------
        periods
            Period or list of periods to query.
        variables
            Set of variables to query.
        polygon
            Selection polygon on which to reduce the data.
            Can be represented as a string or a path to a shapefile, a geopandas
            GeoDataFrame, a shapely Polygon, or a bounding box representing by a
            tuple of floats as (lon_min, lat_min, lon_max, lat_max).
        backend_kwargs
            Additional parameters to pass to the underlying data source.

        Returns
        -------
        :
            Dataset respecting the query constraints.
        """

        return self.source.query(
            periods=periods,
            variables=variables,
            polygon=polygon,
            backend_kwargs=backend_kwargs,
        )

    def query_orbit(
        self,
        cycles_nb: int | list[int],
        passes_nb: int | list[int] | None = None,
        variables: list[str] | None = None,
        polygon: PolygonLike | None = None,
        backend_kwargs: dict[str, Any] | None = None,
    ) -> xr.Dataset:
        """Query data for a set of cycles and passes.

        Parameters
        ----------
        cycles_nb
            Cycle number or list of cycle numbers.
        passes_nb
            Passes number or list of passes numbers.
        variables
            Set of variables to query.
        polygon
            Selection polygon on which to reduce the data.
            Can be represented as a string or a path to a shapefile, a geopandas
            GeoDataFrame, a shapely Polygon, or a bounding box representing by a
            tuple of floats as (lon_min, lat_min, lon_max, lat_max).
        backend_kwargs
            Additional parameters to pass to the underlying data source.

        Returns
        -------
        :
            Dataset respecting the query constraints.
        """
        return self.source.query_orbit(
            cycles_nb=cycles_nb,
            passes_nb=passes_nb,
            variables=variables,
            polygon=polygon,
            backend_kwargs=backend_kwargs,
        )
