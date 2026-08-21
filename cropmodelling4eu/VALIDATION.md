# Germany smoke test — SIMPLACE vs torchcrop vs observations

30 cropland cells across Germany (7.2–13.6 °E, 48.4–54.0 °N, 30 distinct
NUTS-3 regions), winter wheat, harvest years 2000–2010. Both models run the
**same cells** from the same export, so they differ only in the model.

```bash
./submit/submit_simplace.sh  --smoke          # cells, config, SIMPLACE run, collect
./submit/submit_torchcrop.sh --smoke          # the same cells through torchcrop
python scripts/validate_germany.py --cells cells.csv --torchcrop tc.parquet \
    --simplace sp.parquet --out-dir validation/
```

Both `--smoke` flags derive one config and share one cell list, so the models
cannot drift onto different cells or seasons. The steps they wrap
(`select_german_cells.py`, `run_cells_torchcrop.py`, `run_cells_simplace.py`)
are still there to be run individually.

Artefacts: `/data01/FDS/muduchuru/Data/SIMPLACE/cropmodelling4eu/de_smoke/`.

> **The tables below predate the 2026-08-13 re-run on the `SIMPLACE/EU` export**
> (the `europe_torchcrop` export they were produced from no longer exists). On
> the new export, which supplies a real per-cell SAGE sowing calendar in place
> of the DOY 270 fallback, the same 30 cells and seasons give:
> torchcrop 3.22 t/ha (bias −3.93, was −5.17) and SIMPLACE 6.63 t/ha
> (bias −0.53, was −0.12); SIMPLACE's heading and harvest, unbiased to within a
> day before, are now **+21.8 d and +19.3 d late** while torchcrop's soft dough
> improved to −2.8 d. The phenology shift is unexplained and is the first thing
> to look at before quoting either table.

## Yield vs CyBench (NUTS-3 statistics, t/ha)

| model | n | sim mean | obs mean | bias | RMSE | MAE | r |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **simplace** | 324 | 7.02 | 7.13 | **−0.12** | 1.37 | 1.11 | 0.33 |
| torchcrop | 330 | 1.99 | 7.15 | −5.17 | 5.74 | 5.23 | −0.14 |

SIMPLACE is essentially unbiased at the national level and reproduces the 2003
drought (5.10 t/ha against a 7.0 t/ha norm). Interannual correlation is modest
— r = 0.33 across 30 regions — which is what a run with no cultivar
calibration, a fixed sowing date and a single CO₂ value should be expected to
manage.

**The yields are compared on their own bases.** The models report grain dry
matter and CyBench reports market moisture (~13.5 %), so roughly 15 % of the
reference is water. Correcting for it would move SIMPLACE from −0.12 to about
+0.8 t/ha — the agreement above is closer than the models deserve, not further.

## Phenology vs PEP725 (BBCH stage dates, day of year)

Stations within 25 km of a simulated cell; 231 of 1688 German wheat stations
qualify. One observed date per (cell, year) as the median over matched
stations.

| stage | model | n | sim | obs | bias | RMSE | r |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Sowing (BBCH 00) | torchcrop | 324 | 270.0 | 279.9 | −9.9 | 13.8 | — |
| Sowing (BBCH 00) | simplace | 318 | 250.0 | 280.0 | −30.0 | 31.5 | — |
| Emergence (BBCH 10) | simplace | 317 | 254.3 | 293.6 | −39.3 | 40.7 | 0.1 |
| Heading (BBCH 51) | simplace | 316 | 151.2 | 150.8 | **+0.5** | 14.1 | **0.70** |
| Soft dough (BBCH 85) | torchcrop | 323 | 183.7 | 199.0 | −15.3 | 20.2 | 0.5 |
| Soft dough (BBCH 85) | simplace | 317 | 216.8 | 198.9 | +17.9 | 22.1 | 0.5 |
| Harvest (PEP725 100) | torchcrop | 322 | 183.5 | 216.8 | −33.3 | 36.0 | 0.4 |
| Harvest (PEP725 100) | simplace | 316 | 216.7 | 216.7 | **−0.0** | 13.3 | 0.5 |

**SIMPLACE's heading and harvest dates are the strongest results here** —
unbiased to within a day, with r = 0.70 on heading. That is a genuine test:
neither date is an input.

Three rows are not model error and should not be read as such:

- **Sowing is a constant input** in both runs (DOY 270 / 250), so its "bias" is
  one continental offset with no degrees of freedom. `r` is undefined because a
  constant has no variance. It still matters: it propagates into every later
  stage.
- **Soft dough** (BBCH 85) precedes full ripeness (BBCH 89), which is what the
  models report as maturity. SIMPLACE's +17.9 d is the expected sign and rough
  size of that interval; torchcrop's −15.3 d is not.
- **Emergence** at −39 d is a real finding, but about the *solution*, not the
  export: the Brandenburg solution sets `vTSUMEM = 50 °Cd`, which in a German
  September is 3–4 days from sowing to emergence against an observed ~14. A
  winter-wheat `TSUMEM` is normally 100–120 °Cd.

## What the comparison says about each model

**SIMPLACE is usable now** — for level, drought response and the two phenology
stages it genuinely predicts. Its known limits in this configuration are the
fixed DOY 250 sowing, `vTSUMEM`, one CO₂ value for the run and no per-cultivar
calibration.

**torchcrop is not**, in its published configuration. Its 1.99 t/ha is 3.6×
below the statistics, and the cause shows in the intermediates: peak LAI
averages **1.05** where a German wheat canopy reaches 4–6, and maturity arrives
on DOY 183 against an observed 217 — a month early, so grain filling is cut
short. Water and N are not the constraint (TRANRF 0.82, NNI 0.95).

**This is pre-existing, not a regression from the port.** Re-running the moved
code over these 30 cells reproduced the published
`winter_wheat_2000_2024.parquet` **bit-for-bit — 330/330 rows, max difference
0.0001 g m⁻²**. So the refactor is faithful and the low yields were already
there. The likely causes are the crop parameter set (LINTUL-5 `wheat` defaults
rather than a winter-wheat calibration, which would explain both the early
maturity and the thin canopy) and the DOY 270 sowing latch.

## The radiation fix this test depended on

An unmodified export gives SIMPLACE **0.04 t/ha**. The solution reads its
weather by column *position* and declares column 6 as
`kilojoule_per_square_metre_day`, while the export writes a daily-mean flux in
W m⁻² — a factor of 86.4.
[`simplace/weather.py`](src/cropmodelling4eu/src/cropmodelling4eu/simplace/weather.py)
converts on the way into the workspace (`simplace.weather_contract:
brandenburg`), which is what turns 0.04 t/ha into the 7.02 t/ha above.

That is a workaround. The fix belongs in
[`weather_export.py`](../src/data4simplace/exporters/weather_export.py), so the
export carries the units the reference file it claims to conform to actually
uses.
