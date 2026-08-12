"""Per-cell site inputs: sowing calendar, altitude and the CO2 series.

These are the three inputs both SIMPLACE and torchcrop need and neither the
climate, soil nor NPK stage supplies. Before this stage each runner invented
them — a single sowing day-of-year for the whole continent, an altitude of zero
and a hard-coded CO2 table — so the values were assumptions buried in modelling
code rather than data in the export.
"""

from data4simplace.site.calendar import (
    GGCMI_CROPS,
    SAGE_CROPS,
    load_calendar,
    resolve_calendar_crop,
)
from data4simplace.site.co2 import FALLBACK_CO2_PPM, load_co2_series, write_co2_series
from data4simplace.site.elevation import load_elevation
from data4simplace.site.window import SowingWindow
from data4simplace.site.handler import (
    CALENDAR_SOURCE_CODES,
    SiteHandler,
    fill_calendar_gaps,
)

__all__ = [
    "CALENDAR_SOURCE_CODES",
    "FALLBACK_CO2_PPM",
    "GGCMI_CROPS",
    "SAGE_CROPS",
    "SiteHandler",
    "SowingWindow",
    "fill_calendar_gaps",
    "load_calendar",
    "load_co2_series",
    "load_elevation",
    "resolve_calendar_crop",
    "write_co2_series",
]
