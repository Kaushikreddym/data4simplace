"""Findings shown at the top of each notebook, written after an executed run.

These are static text: rerun and recheck them if the runs or the analysis change.
"""

FINDINGS: dict[str, str] = {
    "01_spatial_differences": """
## Findings

**Yield rises almost everywhere, but the gain comes mainly from the CO₂ parameter (notebook 05), and the pattern partly reflects the calibration.**

* **Future change, Europe vs 1985–2014:** +8.0 % [4.9, 10.7] in `ssp126` 2071–2100 and +28.7 % [22.9, 33.5] in `ssp370` 2071–2100, multi-model mean [model range]. Under `ssp370` the signal is robust on 86 % of cropland, and under `ssp126` on 58 %.
* **Losses are confined to the far south:** southern Spain and Portugal, Sicily, the southern Turkish coast and North Africa. Spain (+1 %) and Greece (+3 %) gain least under `ssp126`, and Denmark gains most (+13 %, and +37 % under `ssp370`).
* **Anchoring is good:** against the MSWX run over 2000–2014, the Europe bias is −7.4 % to +4.4 % and pattern r is 0.97–0.99.
* **Forcing:** over 1985–2014, `hist-aer` is about 1 t/ha below `historical`, but CO₂ is held at 1850 in that run, so this is not an aerosol effect. The clean climate contrast is `hist-GHG − historical` at −0.12 t/ha in Europe and −0.45 t/ha in Eastern Europe, from one run per model.

**Calibration artefacts (§9).** The straight-edged bands in the east are calibration-region boundaries. North–south jumps in baseline yield are 13× larger across a boundary (0.90 vs 0.07 t/ha). Only 43 % of cropland has a CyBench unit, with none in Eastern Europe or West-Central Asia, and region 20 (3.4 % of cropland) runs on the fallback crop. Fine spatial detail outside the EU-23 is not reliable, so prefer the EU-23 row: +9.1 % and +24.7 %.
""",
    "02_timeseries": """
## Findings

**Yield tracks CO₂ closely, and the two SSPs separate only after mid-century.**

* **Historical:** Europe's yield rises from about 3 t/ha in the 1850s to about 4 t/ha in 1985–2014. `hist-aer`, with CO₂ fixed at 1850, stays near 3 t/ha throughout, which is the clearest sign that CO₂ rather than climate drives the rise.
* **Future:** `ssp126` levels off about 8–10 % above the baseline as its CO₂ stabilises. `ssp370` keeps rising, to about +30 % by 2100.
* **Models:** ACCESS-ESM1-5 is the outlier, lowest in level and with `ssp126` falling back to about the baseline by 2100, while the other three hold near +10 %.
* **Join:** there is no large step across the missing 2015. Levels move by −0.16 to +0.19 t/ha between `historical` 2005–2014 and the SSPs in 2016–2025, which is within model differences.
* **MSWX:** its 4.06 t/ha in 2005–2014 lies within the model range of 3.83–4.30.
""",
    "03_trends": """
## Findings

**Trends follow CO₂: strongly significant where CO₂ rises fast, absent where it does not.**

* **1951–2014, Europe:** `historical` +0.12 to +0.19 t/ha per decade, `hist-GHG` +0.09 to +0.14, both significant in every model. `hist-aer` is −0.03 to +0.01 and significant in none.
* **Cropland with a significant rise, after false-discovery control:** 77 % in `historical`, 56 % in `hist-GHG` and 2 % in `hist-aer`, with almost no significant falls.
* **30 years is short:** over 1985–2014, only 2 of 4 models have a significant Europe trend in `historical`, and the cell maps disagree on sign in the east.
* **Future:** `ssp370` 2051–2100 rises +0.08 to +0.19 t/ha per decade. `ssp126` 2051–2100 is flat to slightly falling (−0.06 to +0.01), with 0 % of cropland significant. Its yield peaks in 2039–2078 depending on model, while `ssp370` is still rising at the end of the run in every model.
* **Emergence beyond one interannual standard deviation by 2090:** 73 % of cropland in `ssp370` and 22 % in `ssp126`.
* **Recent trend:** MSWX 2000–2024 is +0.12 t/ha per decade (p = 0.27), inside the models' −0.05 to +0.19, and no run is significant over that window.
* **Method check:** median lag-1 autocorrelation of detrended seasons is −0.04 to +0.03, so Mann–Kendall's independence assumption holds. Forcing trends roughly add up, with residuals of +0.00 to +0.10 t/ha per decade, largest in ACCESS-ESM1-5.
""",
    "04_extreme_yield_probability": """
## Findings

**As simulated, low-yield risk falls across most of Europe, but that fall depends on the prescribed CO₂ effect. Without it, far-future `ssp370` risk rises instead.**

| Europe, 1-in-10 low yield, risk ratio [model range] | As simulated | `COTB` removed (§10) |
|---|---|---|
| `ssp126` 2071–2100 | 0.72 [0.51, 0.96] | 1.13 [0.87, 1.34] |
| `ssp370` 2021–2050 | 0.51 [0.37, 0.74] | 0.91 [0.77, 1.16] |
| `ssp370` 2071–2100 | 0.42 [0.28, 0.58] | **1.81 [1.52, 2.30]** |
| `hist-aer` 1985–2014 | 4.15 [3.32, 5.11] | 1.18 [0.89, 1.69] |

* **The rescaling is an approximation, not a rerun.** It passes its test only roughly: `hist-aer` returns to a mean of −0.4 %, but its risk ratio of 1.18 has a range spanning 1. Treat the direction under `ssp370` far as the result, and the 1.81 as indicative.
* **Mean or variability:** as simulated, nearly all of the change comes from the shifting mean (Europe `ssp370` far: −7.2 points in total, −7.7 from the mean). Detrended CV falls by 2.4 points in Europe, and more in the Mediterranean (−5.7) and Eastern Europe (−6.1).
* **Where risk rises even as simulated:** the far south. Models agree on a rise on 5 % of cropland under `ssp370` far, against a fall on 84 %. Spain (+1.9 points) and Greece (+1.1) are the only countries with an increase in that scenario.
* **Synchronous shortfalls:** seasons with ≥ 20 % of cropland below its own 1-in-10 level fall from 18 % to 13 % (`ssp126` far) and 0.8 % (`ssp370` far, one season of 120).
* **Continental 1-in-10 season:** −10 % of baseline today, −6 % in `ssp126` far and +15 % in `ssp370` far.
* **Non-maturing seasons** rise under `ssp370` far from 0.05 % to 0.42 % of cropland area-seasons, and from 0.24 % to 1.19 % in the Mediterranean. These use a fixed sowing date and cultivar.
* **Heat sterility** barely alters the result: adding it raises risk ratios by 0.04–0.08 for Europe. The hotspot with a sharp edge in eastern Ukraine under `ssp370` follows a calibration boundary, so do not read it as climate.
""",
    "05_explainability": """
## Findings

**The rise in simulated yield is mostly one prescribed parameter. Climate alone is a small net loss, and water and atmospheric demand decide which seasons turn out bad.**

* **CO₂ share:** the `COTB` table on its own gives +36.0 % for `ssp370` far, against +35.7 % from SHAP, and −24.0 % for `hist-aer` against −23.5 %. Resetting CO₂ to normal in the emulator gives a somewhat larger share (+41.7 % and −27.8 %). The table and the emulator were derived independently, so this attribution is firm **as a statement about the model**. Whether the table is right for real wheat is a separate question that these runs cannot answer. The table is also steep below its 360 ppm kink: 0.31 %/ppm below against 0.10 %/ppm above.
* **Everything else, with `COTB` divided out:** −1.9 % (`ssp126` far), −4.2 % (`ssp370` far) and −4.0 % (`hist-GHG`). This is a small net climate loss, and it still includes CO₂'s reduction of transpiration.
* **Within climate (`ssp370` far, SHAP):** atmospheric demand −11.8 and heat extremes −4.2, partly offset by mean temperature +8.6. The response curves show warmer winters raising yield and warmer grain fill lowering it. VPD is the strongest negative climate driver in almost every 1° box, and the warm groups are correlated, so read them as a set (net −7.4).
* **Mechanism:** the gain is in biomass (+29 % in log terms), and harvest index falls (−4 %). Maturity comes 21 days earlier, the heat-stress factor rises from 0.007 to 0.025, and the nitrogen index falls by 0.018 on fixed fertiliser. The mean transpiration factor hardly moves.
* **Bad seasons** are water and demand events in every period: −10 to −13 points from water and −8 to −10 from VPD against the same period's mean, and heat −3 to −6, against the same period's mean. The CO₂ part (−2 to −5) needs care: it is largest where CO₂ climbs fastest within the window, which suggests bad seasons lean towards the window's early, lower-CO₂ years, though this was not tested directly.

**How far to trust the emulator.** It closes on the simulated change to within about 1 point in-sample, and transfers well to an unseen model (R² 0.65–0.71). Extrapolation is weak: R² 0.34 with bias −12.5 % when `ssp370` is held out, and R² 0.35 with bias +11.2 % when `hist-aer` is held out. Trees cannot extrapolate, and the `COTB` check above is what supports the CO₂ share. The climate split has no such check. A torchcrop rerun with CO₂ held fixed under SSP weather would settle it (§7).
""",
}
