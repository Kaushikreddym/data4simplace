"""Per-crop constants: one profile per simulated crop, selected by ``CM4EU_CROP``.

Every number the calibration and the evaluation used to fix for winter wheat
lives here, so a second crop changes one name rather than a dozen module
constants. The profile is chosen **once per process**, from the environment, and
the module-level constants of :mod:`cropmodelling4eu.evaluation.config` and
:mod:`cropmodelling4eu.calibration.config` are read from it at import time — so
a submit script exports ``CM4EU_CROP`` and nothing else has to know.

The wheat profile reproduces the previous constants exactly; a test pins it.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

__all__ = ["CROP_ENV", "CROP_PROFILE", "CROP_PROFILES", "CropProfile", "profile"]

#: Environment variable that selects the active profile.
CROP_ENV: str = "CM4EU_CROP"


@dataclass(frozen=True, slots=True)
class CropProfile:
    """Everything crop-specific the pipeline reads, by data source.

    Attributes:
        key: Profile name, also ``SeasonConfig.crop``.
        cybench: CyBench crop directory and file infix.
        clms: CLMS reducer crop label.
        clms_require_split: Gate on the reducer's winter/spring validation.
            Only a split crop can pass it; maize never does.
        gdhy: GDHY crop directory.
        sage: SAGE calendar name.
        simplace_block: ``Crop`` code of the SUSTAg ``LINTUL5_crop.xml`` block.
        pep725: Key into :data:`data4simplace.phenology.pep725.PEP_CROPS`.
        autumn_cut_doy: An autumn stage on or after this DOY belongs to the
            following harvest year. ``367`` means no stage ever moves, which is
            right for a crop sown and harvested in one calendar year.
        clms_harvest_offset_days: Days **added** to CLMS ``CPMCH``.
        clms_emergence_offset_days: Days **subtracted** from CLMS ``CPMCE``.
        clms_offsets_measured: Whether the two offsets were measured against
            PEP725. An unmeasured profile carries zeros, and says so.
        clms_harvest_usable: Whether CLMS harvest enters the calibration at all.
        yield_moisture_content: CyBench market moisture, for the dry-matter
            conversion.
        min_plausible_season_days: Shorter SIMPLACE seasons are treated as the
            mod-365 duration wrap. A spring crop never wraps.
        sim_sowing_doy: Fallback sowing DOY of the fixed-date runs.
        min_days_after_sowing: Default torchcrop season window after sowing.
        spinup_months: Default torchcrop crop-free spin-up before sowing. The
            sowing latch is ``doy >= idpl`` on a day-of-year, so a spin-up that
            crosses New Year sows on the window's first day: a spring crop
            cannot take wheat's six months.
        emergence_anomaly_weight: Default ``loss.emergence_anomaly_clms``.
        exclude_irrigated: Default ``data.exclude_irrigated``.
        stage_parameters: Parameter-spec file per calibration stage.
    """

    key: str
    cybench: str
    clms: str
    clms_require_split: bool
    gdhy: str
    sage: str
    simplace_block: str
    pep725: str
    autumn_cut_doy: int
    clms_harvest_offset_days: float
    clms_emergence_offset_days: float
    clms_offsets_measured: bool
    clms_harvest_usable: bool
    yield_moisture_content: float
    min_plausible_season_days: float
    sim_sowing_doy: int
    min_days_after_sowing: int
    spinup_months: int
    emergence_anomaly_weight: float
    exclude_irrigated: bool
    stage_parameters: dict[str, str] = field(default_factory=dict)


CROP_PROFILES: dict[str, CropProfile] = {
    "wheat": CropProfile(
        key="wheat",
        cybench="wheat",
        clms="wheat_winter",
        clms_require_split=True,
        gdhy="wheat_winter",
        sage="Wheat.Winter",
        simplace_block="WW",
        pep725="wheat",
        autumn_cut_doy=180,
        # Measured against PEP725: identical on both supports, constant year to
        # year, unchanged under 3x3 smoothing -- a product bias.
        clms_harvest_offset_days=18.7,
        clms_emergence_offset_days=6.0,
        clms_offsets_measured=True,
        clms_harvest_usable=True,
        yield_moisture_content=0.135,
        min_plausible_season_days=150.0,
        sim_sowing_doy=270,
        min_days_after_sowing=365,
        spinup_months=6,
        # CLMS emergence anomaly r 0.11 against PEP725: near zero, not off.
        emergence_anomaly_weight=0.05,
        exclude_irrigated=False,
        stage_parameters={
            "phenology": "params_phenology.yaml",
            "yield": "params_yield.yaml",
            "lai": "params_lai.yaml",
        },
    ),
    "maize": CropProfile(
        key="maize",
        cybench="maize",
        # CTY 1130 holds grain and silage in one class. Measured against PEP725
        # Zea 2017-2024 on the 10 km support (tmp_claude/clms_pep725_maize.py,
        # 2026-11-16): CPMCH - BBCH 100 = +4.6 d, RMSE 15, anomaly r 0.80 over
        # 2 738 pairs; CPMCE - BBCH 10 = +16.4 d, anomaly r 0.35. BBCH 85 is
        # 31 d off, so CPMCH is the harvest, not dough ripeness. 97 % of the
        # pairs are German; the 37 non-German harvest pairs read -14 d.
        clms="maize",
        clms_require_split=False,
        gdhy="maize",
        sage="Maize",
        simplace_block="MAIZ",
        pep725="maize",
        autumn_cut_doy=367,
        clms_harvest_offset_days=-4.6,
        clms_emergence_offset_days=16.4,
        clms_offsets_measured=True,
        clms_harvest_usable=True,
        # User decision 2026-11-16, not yet verified against CyBench/Eurostat.
        yield_moisture_content=0.14,
        min_plausible_season_days=60.0,
        sim_sowing_doy=120,
        # Sown April-June and mature within the calendar year; 365 would
        # simulate half a year of bare soil after harvest.
        min_days_after_sowing=240,
        # SIMPLACE sows from DOY 57; any spin-up of 2+ months crosses New Year.
        spinup_months=0,
        # Anomaly r 0.35 against PEP725; off by user decision 2026-11-16.
        emergence_anomaly_weight=0.0,
        # vIRR = 1 cells kept out of the pool (user decision 2026-11-16).
        exclude_irrigated=True,
        stage_parameters={
            "phenology": "params_phenology_maize.yaml",
            "yield": "params_yield_maize.yaml",
            "lai": "params_lai.yaml",
        },
    ),
}


def profile(name: str | None = None) -> CropProfile:
    """The named profile, or the one ``CM4EU_CROP`` selects (default wheat)."""
    key = (name or os.environ.get(CROP_ENV) or "wheat").strip().lower()
    if key not in CROP_PROFILES:
        raise ValueError(f"{CROP_ENV}={key!r}: known crops are {sorted(CROP_PROFILES)}")
    return CROP_PROFILES[key]


#: The profile this process runs on.
CROP_PROFILE: CropProfile = profile()
if not CROP_PROFILE.clms_offsets_measured:
    logger.warning(
        "%s: CLMS offsets are not measured against PEP725; CLMS dates are used "
        "uncorrected and CLMS harvest %s the calibration",
        CROP_PROFILE.key, "enters" if CROP_PROFILE.clms_harvest_usable else "is kept out of",
    )
