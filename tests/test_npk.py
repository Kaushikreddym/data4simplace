"""Tests for the NPKGRIDS reader and the fertilizer-composition parser."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from data4simplace.config import PipelineConfig
from data4simplace.npk.composition import (
    K2O_TO_K,
    P2O5_TO_P,
    default_composition_path,
    parse_fertilizer_composition,
)
from data4simplace.npk.npkgrids import NPKGridsHandler

COMPOSITION_XML = """<?xml version="1.0" encoding="UTF-8"?>
<fertilizerparameters>
    <fertilizer>
        <parameter id="Fertilizertype">KAS</parameter>
        <parameter id="Nitrate"> 0.135 </parameter>
        <parameter id="Ammonium"> 0.135 </parameter>
        <parameter id="Potassium"> 0 </parameter>
        <parameter id="Phosphorus"> 0 </parameter>
        <parameter id="NitrateAndAmmonium"> 0.27 </parameter>
        <parameter id="OrganicC"> 0.0</parameter>
        <parameter id="OrganicN"> 0.0</parameter>
    </fertilizer>
    <fertilizer>
        <parameter id="Fertilizertype">PK</parameter>
        <parameter id="Nitrate"> 0.0 </parameter>
        <parameter id="Ammonium"> 0.0 </parameter>
        <parameter id="Potassium"> 0.083 </parameter>
        <parameter id="Phosphorus"> 0.0792 </parameter>
        <parameter id="NitrateAndAmmonium"> 0.0 </parameter>
    </fertilizer>
    <fertilizer>
        <parameter id="Fertilizertype">P</parameter>
        <parameter id="Phosphorus"> 0.4364 </parameter>
    </fertilizer>
    <fertilizer>
        <parameter id="Fertilizertype">K</parameter>
        <parameter id="Potassium"> 0.8302 </parameter>
    </fertilizer>
    <fertilizer>
        <parameter id="Fertilizertype">Urea</parameter>
        <parameter id="Nitrate"> 0.23 </parameter>
        <parameter id="Ammonium"> 0.23 </parameter>
    </fertilizer>
</fertilizerparameters>
"""


# --------------------------------------------------------------------------- #
# fertilizer_composition.xml
# --------------------------------------------------------------------------- #
def test_parse_composition_reads_contents(tmp_path):
    path = tmp_path / "fertilizer_composition.xml"
    path.write_text(COMPOSITION_XML)

    comps = parse_fertilizer_composition(path)
    assert set(comps) == {"KAS", "PK", "P", "K", "Urea"}
    assert comps["KAS"].mineral_n == pytest.approx(0.27)
    assert comps["KAS"].carries_n and not comps["KAS"].carries_p
    assert comps["PK"].carries_p and comps["PK"].carries_k
    # The straight carriers are the pure oxides, to within the file's rounding.
    assert comps["P"].phosphorus == pytest.approx(P2O5_TO_P, abs=1e-4)
    assert comps["K"].potassium == pytest.approx(K2O_TO_K, abs=1e-4)


def test_parse_composition_recovers_missing_mineral_n_total(tmp_path):
    # Urea declares only the split forms; the total must be recovered from them.
    path = tmp_path / "fertilizer_composition.xml"
    path.write_text(COMPOSITION_XML)
    assert parse_fertilizer_composition(path)["Urea"].mineral_n == pytest.approx(0.46)


def test_default_composition_path_sits_next_to_the_reference(tmp_path):
    folder = tmp_path / "management"
    folder.mkdir()
    (folder / "fertilizer_composition.xml").write_text(COMPOSITION_XML)
    csv = folder / "fertilizer_winter_wheat.csv"
    csv.write_text("location,vType,DVS,Amount\n1,KAS,0.25,32\n")

    assert default_composition_path(csv) == folder / "fertilizer_composition.xml"
    assert default_composition_path(folder) == folder / "fertilizer_composition.xml"
    assert default_composition_path(tmp_path / "elsewhere.csv") is None


def test_parse_composition_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        parse_fertilizer_composition(tmp_path / "nope.xml")


# --------------------------------------------------------------------------- #
# NPKGRIDS reader
# --------------------------------------------------------------------------- #
def _write_npkgrids(directory, crop="wheat", nrate=None):
    """Write a miniature NPKGRIDS file on a 0.05 deg south-to-north grid."""
    lat = np.arange(51.975, 52.35, 0.05)  # ascending, as in the real files
    lon = np.arange(11.225, 11.55, 0.05)
    shape = (lat.size, lon.size)
    if nrate is None:
        nrate = np.full(shape, 100.0)
    data = {
        "Nrate": (("lat", "lon"), np.asarray(nrate, dtype="float32")),
        "P2O5rate": (("lat", "lon"), np.full(shape, 40.0, dtype="float32")),
        "K2Orate": (("lat", "lon"), np.full(shape, 60.0, dtype="float32")),
        "Nqual": (("lat", "lon"), np.full(shape, 0.8, dtype="float32")),
        "P2O5qual": (("lat", "lon"), np.full(shape, 0.8, dtype="float32")),
        "K2Oqual": (("lat", "lon"), np.full(shape, 0.8, dtype="float32")),
    }
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"NPKGRIDSv1.08_{crop}.nc"
    xr.Dataset(data, coords={"lat": lat, "lon": lon}).to_netcdf(path)
    return path


def _npk_config(tmp_path, npk_root, **npk):
    return PipelineConfig.model_validate(
        {
            "flags": {"run_npk_processing": True},
            "grid": {
                "resolution_deg": 0.1,
                "min_lon": 11.2,
                "max_lon": 11.5,
                "min_lat": 52.0,
                "max_lat": 52.3,
            },
            "time": {"start": "1979-01-01", "end": "1979-01-03"},
            "paths": {
                "mswx_root": str(tmp_path),
                "npk_root": str(npk_root),
                "output_dir": str(tmp_path / "out"),
            },
            "npk": {"crop": "wheat", **npk},
        }
    )


def test_npkgrids_regrids_rates_to_the_target_grid(tmp_path):
    root = tmp_path / "npkgrids"
    _write_npkgrids(root)
    result = NPKGridsHandler(_npk_config(tmp_path, root)).load()

    assert set(result.data_vars) == {"N", "P2O5", "K2O"}
    assert result["N"].shape == (3, 3)  # 0.3 deg / 0.1 deg
    assert np.allclose(result["N"].values, 100.0)
    assert result["N"].attrs["units"] == "kg-N/ha"
    # Quality is not carried unless asked for.
    assert "N_quality" not in result


def test_npkgrids_masks_ocean_and_zero_pixels(tmp_path):
    root = tmp_path / "npkgrids"
    rates = np.full((8, 7), 100.0)
    rates[:, 0] = -1.0   # ocean column -> never a rate
    rates[:, 1] = 0.0    # land the crop is not grown on
    rates[:2, :] = -1.0  # a wholly ocean target row
    _write_npkgrids(root, nrate=rates)

    result = NPKGridsHandler(_npk_config(tmp_path, root)).load()["N"].values
    # The masked pixels never drag a cell's mean below the rate of its
    # positive pixels, and a cell with no positive pixel is left NaN.
    finite = result[np.isfinite(result)]
    assert finite.size and np.allclose(finite, 100.0)
    assert np.isnan(result).any()


def test_npkgrids_include_zero_rate_averages_the_zeros_in(tmp_path):
    root = tmp_path / "npkgrids"
    rates = np.full((8, 7), 100.0)
    rates[:, 2:4] = 0.0  # one full target cell column of zero-rate land
    _write_npkgrids(root, nrate=rates)

    kept = NPKGridsHandler(
        _npk_config(tmp_path, root, include_zero_rate=True)
    ).load()["N"].values
    assert np.nanmin(kept) == pytest.approx(0.0)

    dropped = NPKGridsHandler(_npk_config(tmp_path, root)).load()["N"].values
    assert np.allclose(dropped[np.isfinite(dropped)], 100.0)


def test_npkgrids_keep_quality_and_min_quality(tmp_path):
    root = tmp_path / "npkgrids"
    _write_npkgrids(root)

    with_quality = NPKGridsHandler(
        _npk_config(tmp_path, root, keep_quality=True)
    ).load()
    assert np.allclose(with_quality["N_quality"].values, 0.8)

    # The fixture's score is 0.8, so a 0.9 floor removes every rate; the filter
    # must work even though keep_quality is off.
    filtered = NPKGridsHandler(_npk_config(tmp_path, root, min_quality=0.9)).load()
    assert "N_quality" not in filtered
    assert np.isnan(filtered["N"].values).all()


def test_npkgrids_unknown_crop_lists_what_is_available(tmp_path):
    root = tmp_path / "npkgrids"
    _write_npkgrids(root, crop="wheat")
    config = _npk_config(tmp_path, root, crop="quinoa")
    with pytest.raises(FileNotFoundError, match="wheat"):
        NPKGridsHandler(config).load()


def test_npkgrids_requires_a_root(tmp_path):
    config = PipelineConfig.model_validate(
        {
            "flags": {"run_npk_processing": True},
            "grid": {
                "resolution_deg": 0.1, "min_lon": 11.2, "max_lon": 11.5,
                "min_lat": 52.0, "max_lat": 52.3,
            },
            "time": {"start": "1979-01-01", "end": "1979-01-03"},
            "paths": {"mswx_root": str(tmp_path), "output_dir": str(tmp_path / "out")},
        }
    )
    with pytest.raises(ValueError, match="paths.npk_root"):
        NPKGridsHandler(config).load()
