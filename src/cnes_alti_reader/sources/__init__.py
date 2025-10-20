"""Altimetric reader's sources implementation."""

__all__ = [
    "DOC_PARAMETERS_ALTI_SOURCE",
    "HALF_ORBIT_DTYPE",
    "ClsTableSource",
    "CnesAltiSource",
    "CnesAltiVariable",
    "FCollectionType",
    "FileCollectionSource",
    "ScCollectionSource",
]

import logging

from cnes_alti_reader.utilities import missing_dependency_class

from ._model import (
    DOC_PARAMETERS_ALTI_SOURCE,
    HALF_ORBIT_DTYPE,
    CnesAltiSource,
    CnesAltiVariable,
)

LOGGER = logging.getLogger(__name__)

try:
    from ._file_collection import FCollectionType, FileCollectionSource
except ImportError as e:  # pragma: no cover
    LOGGER.debug("Unable to import FileCollectionSource: %s", e)
    FCollectionType = missing_dependency_class(  # type: ignore[assignment,misc]
        dependency="ocean_tools", error=str(e)
    )
    FileCollectionSource = missing_dependency_class(  # type: ignore[assignment,misc]
        dependency="ocean_tools", error=str(e)
    )


try:
    from ._cls_table import ClsTableSource
except ImportError as e:  # pragma: no cover
    LOGGER.debug("Unable to import ClsTableSource: %s", e)
    ClsTableSource = missing_dependency_class(  # type: ignore[assignment,misc]
        dependency="cls_tables", error=str(e)
    )


try:
    from ._sc_collection import ScCollectionSource
except ImportError as e:  # pragma: no cover
    LOGGER.debug("Unable to import ScCollectionSource: %s", e)
    ScCollectionSource = missing_dependency_class(  # type: ignore[assignment,misc]
        dependency="swot_calval", error=str(e)
    )
