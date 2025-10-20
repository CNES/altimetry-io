from __future__ import annotations

import abc
import dataclasses as dc
import enum
import os
import pathlib as pl
from collections.abc import Mapping
from typing import Any, TypeVar

import fsspec
import numpy as np
import pandas as pd
import ruamel.yaml

from cnes_alti_reader.model import CnesAltiData
from cnes_alti_reader.sources import (
    ClsTableSource,
    CnesAltiSource,
    FCollectionType,
    FileCollectionSource,
    ScCollectionSource,
)
from cnes_alti_reader.utilities import normalize_enum, normalize_file_system

CST_CATALOG_ENV = "CNES_ALTI_DATA_CATALOG"


class SourceType(enum.Enum):
    """Different types of collection's sources."""

    CLS_TABLE = enum.auto()
    SC_COLLECTION = enum.auto()
    FILE_COLLECTION = enum.auto()


@dc.dataclass(kw_only=True)
class FileSystemConfig:
    """File system configuration.

    Parameters
    ----------
    protocol
        File system protocol (file, s3, ...)
    options
        Additional file system options.
    """

    protocol: str = "file"
    options: dict[str, Any] = dc.field(default_factory=dict)
    fs: fsspec.AbstractFileSystem = dc.field(init=False)

    def __post_init__(self):
        self.fs = fsspec.filesystem(protocol=self.protocol, **self.options)


@dc.dataclass(kw_only=True)
class CnesAltiCatalogEntry(abc.ABC):
    """Common catalog entry for a CnesAlti source."""

    mission: str
    description: str
    fs: fsspec.AbstractFileSystem | str | None = None

    @classmethod
    def from_config(
        cls: type[CnesAltiCatalogEntryType], config: dict[str, Any]
    ) -> CnesAltiCatalogEntryType:
        """Instantiate this class from the provided configuration.

        Parameters
        ----------
        config
            Catalog configuration.

        Returns
        -------
        :
            Catalog entry.
        """
        config = config.copy()

        if "fs" in config:
            fs = FileSystemConfig(**config.pop("fs")).fs
        else:
            fs = None

        return cls(**config, fs=fs)

    @abc.abstractmethod
    def open_source(self, name: str) -> CnesAltiSource[Any]:
        """Open the source associated to this entry.

        Parameters
        ----------
        name
            Entry name.

        Returns
        -------
        :
            Instantiated source.
        """

    def match_key(self, key: str | None) -> bool:
        """Check if the mission or description matches the provided key."""
        if key is None:
            return True

        key = key.upper()

        if key in self.mission.upper() or key in self.description.upper():
            return True

        return False


CnesAltiCatalogEntryType = TypeVar(
    "CnesAltiCatalogEntryType", bound=CnesAltiCatalogEntry
)


@dc.dataclass(kw_only=True)
class CnesAltiCatalogCLSTables(CnesAltiCatalogEntry):
    """Catalog entry for a CLS Table."""

    orf: str | None = None

    def open_source(self, name: str) -> CnesAltiSource[Any]:
        return ClsTableSource(name=name, orf=self.orf)


@dc.dataclass(kw_only=True)
class CnesAltiCatalogScCollection(CnesAltiCatalogEntry):
    """Catalog entry for a Swot Calval collection."""

    path: str

    def open_source(self, name: str) -> CnesAltiSource[Any]:
        return ScCollectionSource(path=self.path, fs=self.fs)


@dc.dataclass(kw_only=True)
class CnesAltiCatalogFileCollection(CnesAltiCatalogEntry):
    """Catalog entry for a file collection."""

    path: str
    ftype: str

    # TODO: Do not set this?
    time: str = "time"
    longitude: str = "longitude"
    latitude: str = "latitude"
    index: str = "num_lines"

    mission_phase: str | None = None
    subset: str | None = None
    version: str | None = None

    def open_source(self, name: str) -> CnesAltiSource[Any]:
        return FileCollectionSource(
            path=self.path,
            fs=self.fs,
            ftype=normalize_enum(self.ftype, FCollectionType),
            index=self.index,
            time=self.time,
            longitude=self.longitude,
            latitude=self.latitude,
            subset=self.subset,
            version=self.version,
        )


@dc.dataclass(kw_only=True)
class CnesAltiCatalog:
    """Catalog of CnesAlti sources."""

    environment: dict[str, str] = dc.field(default_factory=dict)
    cls_tables: dict[str, CnesAltiCatalogCLSTables]
    sc_collections: dict[str, CnesAltiCatalogScCollection]
    file_collections: dict[str, CnesAltiCatalogFileCollection]
    fs: fsspec.AbstractFileSystem | str | None = dc.field(default=None, compare=False)

    def __post_init__(self):
        self.fs = normalize_file_system(fs=self.fs)

        for k, v in self.environment.items():
            os.environ[k] = v

    @classmethod
    def load(cls, path: str | None = None) -> CnesAltiCatalog:
        """Load a catalog from a YAML file.

        Parameters
        ----------
        path
            Catalog file path.
            If None, the environment variable CNES_ALTI_DATA_CATALOG is used.
            If no environment variable is set, the default catalog is used.

        Returns
        -------
        :
            A catalog.
        """

        yaml = ruamel.yaml.YAML()
        path_norm = cls._catalog_path(path=path)

        try:
            config = yaml.load(os.path.expandvars(path_norm.read_text()))

            return CnesAltiCatalog.from_config(config=config)
        except Exception as error:
            msg = f"Invalid JSON file {path_norm!r}: {error}"

            raise RuntimeError(msg) from error

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> CnesAltiCatalog:
        """Instantiate this class from the provided configuration.

        Parameters
        ----------
        config
            Catalog configuration.
        fs
            Filesystem configuration used to load datasets.

        Returns
        -------
        :
            Catalog.
        """
        environment = config.get("environment", {})
        cls_tables: dict[str, dict] = config.get("cls_tables", {})
        sc_collections = config.get("sc_collections", {})
        file_collections = config.get("file_collections", {})

        return cls(
            environment=environment,
            cls_tables={
                name: CnesAltiCatalogCLSTables.from_config(config=cfg)
                for name, cfg in cls_tables.items()
            },
            sc_collections={
                name: CnesAltiCatalogScCollection.from_config(config=cfg)
                for name, cfg in sc_collections.items()
            },
            file_collections={
                name: CnesAltiCatalogFileCollection.from_config(config=cfg)
                for name, cfg in file_collections.items()
            },
        )

    def open_data(self, dtype: SourceType | str, name: str) -> CnesAltiData:
        """Open a dataset.

        Parameters
        ----------
        dtype
            Type of dataset.
        name
            Name of the dataset.

        Returns
        -------
        :
            Opened dataset.
        """
        dtype = normalize_enum(dtype, SourceType)
        source_info: CnesAltiCatalogEntry

        match dtype:
            case SourceType.CLS_TABLE:
                source_info = self.cls_tables[name]
            case SourceType.SC_COLLECTION:
                source_info = self.sc_collections[name]
            case SourceType.FILE_COLLECTION:
                source_info = self.file_collections[name]
            case _:
                msg = f"Unknown source type: {dtype}"

                raise ValueError(msg)

        return CnesAltiData(source=source_info.open_source(name=name))

    def show_dataset(
        self,
        dtypes: list[SourceType | str] | SourceType | str | None = None,
        containing: str | None = None,
    ) -> pd.DataFrame:
        """Display dataset containing a given string as a DataFrame.

        Parameters
        ----------
        dtypes
            Types of dataset (single one, list of types or None for all types).
        containing
            String contained in dataset's mission name or descriptions.

        Returns
        -------
        :
            List of matching datasets.
        """
        dtypes_norm: list[SourceType] = []

        match dtypes:
            case None:
                dtypes_norm = [
                    SourceType.CLS_TABLE,
                    SourceType.SC_COLLECTION,
                    SourceType.FILE_COLLECTION,
                ]
            case SourceType() | str():
                dtypes_norm = [normalize_enum(dtypes, SourceType)]
            case list():
                dtypes_norm = [normalize_enum(dtype, SourceType) for dtype in dtypes]

        datasets: dict[tuple[str, str], CnesAltiCatalogEntry] = {
            (name, xx.name): ds
            for xx in dtypes_norm
            for name, ds in self._dtype_entries(dtype=xx).items()
            if ds.match_key(key=containing)
        }

        if not datasets:
            data = pd.DataFrame([], columns=["name", "type", "mission", "description"])
        else:
            data = pd.DataFrame(
                np.array(
                    [
                        [name, dtype, ds.mission, ds.description]
                        for (name, dtype), ds in datasets.items()
                    ]
                ),
                columns=["name", "type", "mission", "description"],
            )

        return data

    @staticmethod
    def _catalog_path(path: str | None = None) -> pl.Path:
        """Return the catalog path with the following priority:

        * Provided path
        * Environment variable CNES_ALTI_DATA_CATALOG
        * Default catalog path
        """
        if path is not None:
            return pl.Path(path)

        if CST_CATALOG_ENV in os.environ:
            path_norm = pl.Path(os.environ[CST_CATALOG_ENV])
        else:
            path_norm = pl.Path(__file__).parent.parent / "resources" / "catalog.yaml"

        return path_norm

    def _dtype_entries(
        self, dtype: SourceType | str
    ) -> Mapping[str, CnesAltiCatalogEntry]:
        """Return the entries of a given type."""
        dtype = normalize_enum(dtype, SourceType)

        match dtype:
            case SourceType.CLS_TABLE:
                return self.cls_tables
            case SourceType.SC_COLLECTION:
                return self.sc_collections
            case SourceType.FILE_COLLECTION:
                return self.file_collections
