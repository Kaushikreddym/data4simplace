"""Shared fixtures: a miniature data4simplace export on disk."""

from __future__ import annotations

import gzip
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cropmodelling4eu.config import GridConfig, RunConfig

#: A 10 x 10 test grid. Small enough to write in full, large enough that
#: SimplaceID arithmetic is not trivially the identity.
TEST_GRID = {
    "resolution_deg": 0.1,
    "min_lon": 10.0,
    "max_lon": 11.0,
    "min_lat": 52.0,
    "max_lat": 53.0,
}

#: Cells the fixture export covers (row, col) -> SimplaceID with n_lon = 10.
TEST_CELLS = [1, 2, 12, 35]

#: Layer bottoms of the fixture soil profile [m].
LAYER_BOTTOMS_M = [0.1, 0.3, 0.5, 0.7, 1.0, 2.0]

_SOIL_STEMS = {
    "clay": 20.0,
    "sand": 40.0,
    "bulkdensity": 1.4,
    "carbon": 15.0,
    "PH": 6.5,
    "soilwater_fc": 0.31,
    "soilwater_wp": 0.14,
    "soilwater_sat": 0.42,
    "soilwater_init": 0.31,
    "soilwater_res": 0.1,
    "ammonium": 12.0,
    "nitrate": 28.0,
}


@pytest.fixture()
def grid() -> GridConfig:
    return GridConfig(**TEST_GRID)


def _write_weather(
    directory: Path, grid: GridConfig, ids: list[int], nested: bool = False
) -> None:
    """One gzipped, tab-separated weather file per cell, covering 1999-2002.

    ``nested`` writes ``<row>/<file>`` the way data4simplace does now; the
    default is the flat layout every export written before that change uses.
    Both are read, so both are fixtured.
    """
    directory.mkdir(parents=True, exist_ok=True)
    dates = pd.date_range("1999-01-01", "2002-12-31", freq="D")
    doy = dates.dayofyear.to_numpy()
    # A plausible mid-latitude annual cycle, so a season slice is not constant.
    seasonal = np.cos((doy - 200) / 365.0 * 2 * np.pi)
    frame = pd.DataFrame(
        {
            "Date": dates.strftime("%Y-%m-%d"),
            "Precipitation": np.round(1.5 + seasonal * 0.5, 2),
            "TempMin": np.round(4.0 + seasonal * 8.0, 2),
            "TempMean": np.round(9.0 + seasonal * 9.0, 2),
            "TempMax": np.round(14.0 + seasonal * 10.0, 2),
            "Radiation": np.round(120.0 + seasonal * 90.0, 2),
            "Windspeed": 2.5,
            "RelHumCalc": np.round(75.0 - seasonal * 10.0, 2),
        }
    )
    for simplace_id in ids:
        row, col = (simplace_id - 1) // grid.n_lon, (simplace_id - 1) % grid.n_lon
        parent = directory / str(row) if nested else directory
        parent.mkdir(parents=True, exist_ok=True)
        path = parent / f"daily_mean_RES1_C{col}R{row}.csv.gz"
        with gzip.open(path, "wt", encoding="utf-8") as handle:
            frame.to_csv(handle, sep="\t", index=False)


def _wide_soil(ids: list[int]) -> pd.DataFrame:
    """The wide profile: one row per location, ``<stem>_<N>`` columns."""
    data: dict[str, object] = {"location": ids}
    for stem, value in _SOIL_STEMS.items():
        for layer in range(1, 7):
            data[f"{stem}_{layer}"] = value
    for layer, bottom in enumerate(LAYER_BOTTOMS_M, start=1):
        data[f"SoilLayerDepth_{layer}"] = bottom
    return pd.DataFrame(data)


def _long_soil(ids: list[int]) -> pd.DataFrame:
    """The same profile in the EU SUSTAg long layout."""
    rows = []
    for simplace_id in ids:
        for bottom in LAYER_BOTTOMS_M:
            rows.append(
                {
                    "location": simplace_id,
                    "Depth": bottom,
                    "LL": _SOIL_STEMS["soilwater_wp"],
                    "DUL": _SOIL_STEMS["soilwater_fc"],
                    "SAT": _SOIL_STEMS["soilwater_sat"],
                    "BD": _SOIL_STEMS["bulkdensity"],
                    # g/100g where the wide layout writes g/kg.
                    "OC": _SOIL_STEMS["carbon"] / 10.0,
                    "ph": _SOIL_STEMS["PH"],
                    "sand": _SOIL_STEMS["sand"],
                    "clay": _SOIL_STEMS["clay"],
                    "soilwater_init": _SOIL_STEMS["soilwater_init"],
                    "soilwater_res": _SOIL_STEMS["soilwater_res"],
                }
            )
    return pd.DataFrame(rows)


def _fertilizer(ids: list[int]) -> pd.DataFrame:
    """A wide schedule: product amounts of KAS and the straight P/K carriers."""
    events = [("P", 0.001, 40.0), ("K", 0.001, 40.0), ("KAS", 0.25, 32.0),
              ("KAS", 0.4, 16.0), ("KAS", 0.9, 16.0)]
    rows = []
    for simplace_id in ids:
        for number, (vtype, dvs, amount) in enumerate(events, start=1):
            rows.append(
                {
                    "location": simplace_id,
                    "FertilizerScenario": 2,
                    "crop": "winter_wheat",
                    "Event": number,
                    "vType": vtype,
                    "DVS": dvs,
                    "Amount": amount,
                    # One irrigated cell, so the flag is not trivially constant.
                    "vIRR": int(simplace_id == TEST_CELLS[-1]),
                }
            )
    return pd.DataFrame(rows)


def _composition(path: Path) -> None:
    """A cut-down ``fertilizer_composition.xml``."""
    path.write_text(
        "<?xml version='1.0' encoding='UTF-8'?>\n"
        "<FertilizerData>\n"
        "  <fertilizer>\n"
        "    <parameter id='Fertilizertype'>KAS</parameter>\n"
        "    <parameter id='NitrateAndAmmonium'>0.27</parameter>\n"
        "    <parameter id='Phosphorus'>0.0</parameter>\n"
        "    <parameter id='Potassium'>0.0</parameter>\n"
        "  </fertilizer>\n"
        "  <fertilizer>\n"
        "    <parameter id='Fertilizertype'>P</parameter>\n"
        "    <parameter id='NitrateAndAmmonium'>0.0</parameter>\n"
        "    <parameter id='Phosphorus'>0.4364</parameter>\n"
        "    <parameter id='Potassium'>0.0</parameter>\n"
        "  </fertilizer>\n"
        "  <fertilizer>\n"
        "    <parameter id='Fertilizertype'>K</parameter>\n"
        "    <parameter id='NitrateAndAmmonium'>0.0</parameter>\n"
        "    <parameter id='Phosphorus'>0.0</parameter>\n"
        "    <parameter id='Potassium'>0.8302</parameter>\n"
        "  </fertilizer>\n"
        "</FertilizerData>\n",
        encoding="utf-8",
    )


def _site(ids: list[int]) -> pd.DataFrame:
    """A site table with a per-cell sowing date and altitude."""
    return pd.DataFrame(
        {
            "location": ids,
            "latitude": 52.5,
            "longitude": 10.5,
            "altitude_m": [10.0, 20.0, 300.0, 1200.0][: len(ids)],
            # Two distinct sowing dates, so batching by sowing DOY is exercised.
            "sowing_doy": [280, 280, 295, 295][: len(ids)],
            "sowing_start_doy": 270,
            "sowing_end_doy": 300,
            "harvest_doy": 220,
            "season_length_days": 300,
            "calendar_filled": 0,
            "calendar_source": "product",
            "calendar_product": "Wheat.Winter",
        }
    )


@pytest.fixture()
def export_dir(tmp_path) -> Path:
    """A miniature but complete export: weather, soil (both layouts), management, site."""
    root = tmp_path / "export"
    grid = GridConfig(**TEST_GRID)

    _write_weather(root / "weather", grid, TEST_CELLS)

    (root / "soil").mkdir(parents=True, exist_ok=True)
    _wide_soil(TEST_CELLS).to_csv(root / "soil" / "soil.csv", index=False)
    _long_soil(TEST_CELLS).to_csv(root / "soil" / "soil_long.csv", index=False)

    (root / "management").mkdir(parents=True, exist_ok=True)
    _fertilizer(TEST_CELLS).to_csv(
        root / "management" / "fertilizer_winter_wheat.csv", index=False
    )
    _composition(root / "management" / "fertilizer_composition.xml")

    (root / "site").mkdir(parents=True, exist_ok=True)
    _site(TEST_CELLS).to_csv(root / "site" / "site.csv", index=False)
    co2 = root / "site" / "co2.csv"
    with co2.open("w", encoding="utf-8") as handle:
        handle.write("# atmospheric CO2 [ppm], source: fallback\n")
        pd.DataFrame({"year": [2000, 2001, 2002], "co2_ppm": [369.6, 371.1, 373.3]}).to_csv(
            handle, index=False
        )

    (root / "_work").mkdir(parents=True, exist_ok=True)
    (root / "_work" / "config_run.yaml").write_text(
        "grid:\n"
        + "".join(f"  {k}: {v}\n" for k, v in TEST_GRID.items())
        + '  crs: "EPSG:4326"\n',
        encoding="utf-8",
    )
    return root


@pytest.fixture()
def run_config(export_dir) -> RunConfig:
    """A run config over the fixture export, covering the seasons it can run."""
    return RunConfig.model_validate(
        {
            "run_name": "test",
            "paths": {"export_dir": str(export_dir)},
            "grid": TEST_GRID,
            "season": {"start_year": 2000, "end_year": 2001, "crop": "wheat"},
        }
    )
