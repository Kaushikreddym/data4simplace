"""Management-side derivations: the irrigated / rainfed cell classification."""

from data4simplace.management.irrigation import (
    ECIRA_CODES,
    MIRCA_SUBCROPS,
    SIMPLACE_CROP_GROUPS,
    SOURCE_ECIRA,
    SOURCE_MIRCA,
    SOURCE_NONE,
    IrrigationClassification,
    IrrigationClassifier,
    classify,
    conservative_regrid,
    irrigated_fraction,
    resolve_crop_group,
)

__all__ = [
    "ECIRA_CODES",
    "IrrigationClassification",
    "IrrigationClassifier",
    "MIRCA_SUBCROPS",
    "SIMPLACE_CROP_GROUPS",
    "SOURCE_ECIRA",
    "SOURCE_MIRCA",
    "SOURCE_NONE",
    "classify",
    "conservative_regrid",
    "irrigated_fraction",
    "resolve_crop_group",
]
