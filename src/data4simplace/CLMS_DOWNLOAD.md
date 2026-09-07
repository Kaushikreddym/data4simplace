# CLMS winter-wheat phenology — acquisition

Getting Copernicus HRL Croplands emergence and harvest dates onto the 0.1° grid, as the
observed phenology [`cropmodelling4eu/CALIBRATION.md`](../../cropmodelling4eu/CALIBRATION.md)'s stage 1 calibrates against.

Three layers, all 10 m, EPSG:3035, EEA reference grid, 100 × 100 km tiles, **2017–2024**:

| Product | Content |
| --- | --- |
| `CTY` Crop Types | 20 classes; wheat is `1110` |
| `CPMCE` Main Crop Emergence | emergence date as `YYDOY` |
| `CPMCH` Main Crop Harvest | harvest date as `YYDOY` |

## Why this data

Stage 1 has a coverage hole. PEP725 is the only per-year, multi-stage ground observation on
disk — 213 365 wheat rows, BBCH 0/10/31/51/85/100, 1990–2024 — but **94 % of it is
German**:

```
provider 101 (DE)   200 834 rows      providers 4/8/401/1401/2601/2701  ~1 400 total
provider 301 (AT)     5 858           (CZ, BE-NL, ES-Cat, HR, BA, ME)
provider 601 (SK)     3 968
provider 2101 (SI)    1 343
```

So stage 1 can genuinely fit CON, ATC, NEM and parts of ALS/PAN. Every other EnS zone — the
whole Mediterranean, Nordic and Atlantic west — falls back on CyBench's **static**
`sos`/`eos`, which has one value per region and no year dimension, and therefore cannot
reward interannual skill at all.

CLMS covers 2017–2024, i.e. 8 of the calibration's 25 seasons. That is ample: phenology
parameters are time-invariant, so eight years of pan-European observation identifies them.
It only means CLMS cannot feed the pre-2017 anomaly terms.

## The hypothesis that gates everything

`CTY` class `1110` is just "wheat" — **no winter/spring split**.

The workaround under test: `CPMCE` encodes the *year* in `YYDOY`, so wheat emerging in
autumn of `Y−1` is winter wheat and wheat emerging inside `Y` is spring. The independent
check is season length — ~250–320 d winter against ~100–160 d spring.

[`notebooks/clms_winter_wheat_phenology.ipynb`](../../notebooks/clms_winter_wheat_phenology.ipynb)
tests exactly that.

### It was run, on season 2022 over Brandenburg, and **the hypothesis holds**

4 592 347 wheat pixels, 3.8 % of the AOI:

| # | Test | Result | |
| --- | --- | --- | --- |
| 1 | Bimodality | 74.4 % winter / 25.6 % spring | **PASS** |
| 2 | Season length | winter 251 d (p10–p90 228–275); spring 101 d (79–134) | **PASS** |
| 3 | Coverage | 96.4 % of wheat pixels carry a date; 123 of 154 cells hold ≥ 1000 winter pixels | **PASS** |
| 4 | SAGE offset | emergence − plant **+3 d**; harvest − harvest **−52 d** | **QUESTIONABLE** |

`CPMCE` really does encode an autumn emergence: the winter mode's median is **19 Oct 2021**
and the spring mode's is **22 Mar 2022**, and the season lengths they imply do not overlap
even at the p90/p10 boundary (275 d vs 134 d). The winter/spring split is an observation,
not an assumption, and the 1 January cut recovers it correctly at this latitude.

**The open question is `CPMCH`.** Harvest comes back **52 days earlier** than SAGE with a
tight spread (IQR −52..−46), so it is systematic, not noise — median CLMS harvest near
DOY 179 against PEP725's observed German harvest near DOY 217. Either `CPMCH` detects
senescence rather than the combine date, or 2022's drought genuinely advanced harvest. One
season cannot separate those, which is the first reason to repeat over 2017–2024.

Emergence is the milder version of the same worry: SAGE plants at DOY 290 and CLMS emerges
at DOY 293, a **+3 d** gap where germination should cost 10–20 d. The notebook flagged in
advance that a near-zero difference would be the *suspicious* result. It does not
invalidate the split, but CLMS emergence should not be treated as a sowing date without
the degree-day back-off in phase 4.

## What is established

Verified live, not assumed:

| Fact | Evidence |
| --- | --- |
| `~/.clms` service key works | JWT grant → bearer token, HTTP 200; `@search` returns all three datasets |
| Anonymous S3 is refused | 403/401 on every `s3.waw3-1.cloudferro.com/eodata/...` variant — a CDSE account is genuinely required for bulk |
| `EPSG:3035` is offered | `@projections` lists it for all three UIDs, so the categorical rasters need no resampling |
| Volume is **80.7 GB** | 664 of 897 tiles intersect the 70 705 exported cells; 15 936 files over 2017–2024 × 3 layers |
| Staging goes to `/data01/FDS/muduchuru/Data/Agri/CLMS/` | 80.7 GB against 2.5 TB free; it belongs with the other reference datasets |
| `sdba` is the env | only `sdba`, `dev_env`, `esmvaltool` carry `jwt` + `cryptography` + `rasterio`; `sdba` is already the torchcrop env |

Per-product, over the 664 tiles that intersect the export:

| Product | Files | Size |
| --- | --- | --- |
| `CPMCE` | 5 312 | 29.3 GB |
| `CPMCH` | 5 312 | 25.9 GB |
| `CTY` | 5 312 | 25.6 GB |

Roughly 9.5 GB per year, except 2024 at 14.4 GB. Manifests are staged at
`/data01/FDS/muduchuru/Data/Agri/CLMS/inventory/` (`clms_manifest_full.parquet`,
`clms_manifest_export.parquet`, `tiles_export.txt`).

### Identifiers, pinned

These remove the discovery round-trip and, more importantly, close a trap: **each dataset
exposes two `DatasetDownloadInformationID`s** — the product *and* a Confidence Layer.
Selecting by title alone can silently fetch the confidence raster, which is a plausible
GeoTIFF of entirely the wrong thing.

| Code | `DatasetID` (UID) | `DatasetDownloadInformationID` (product) |
| --- | --- | --- |
| `CTY` | `0d4d0d517b824d22857e43c4ea6073c9` | `869dd60b-1cd7-414b-8281-4f5e4eaea8f1` |
| `CPMCE` | `325ff9b20fd648cbaee5cd4258f54604` | `7aaf78c5-43d0-46a0-bdc1-fd178d05c0a6` |
| `CPMCH` | `461193fa864d4a6d92de66be9111a4f5` | `89d9c3a4-8fca-4911-ae8c-948f0eddba8b` |

The confidence layers are not waste — they give stage 1 a per-pixel quality weight. Fetch
them in the bulk phase, not the probe.

## Two routes, and why both

| | Route B — CLMS `@datarequest_post` | Route A — CDSE S3 bulk |
| --- | --- | --- |
| Credential | `~/.clms` service key — **works today** | CDSE account + S3 keys — **does not exist yet** |
| Clipping | server-side, by bbox/NUTS | whole tiles, clipped locally |
| Mechanism | asynchronous FME job, **jobs run concurrently** | direct object fetch, fully parallel |
| Unit of work | one clip, ~1.5 deg² before cost runs away | one 100 × 100 km tile |
| Good for | a region, now, with no new credential | 664 tiles × 8 years × 3 layers |

**Measured scaling** (server timestamps, three layers unless noted):

| AOI | Layers | Duration | Delivered |
| --- | --- | --- | --- |
| 0.25 deg² | 1 | 22 s | 0.7 MB |
| 1.5 deg² (the probe) | 3 | 2 m 54 s | 13.0 MB |
| 24 deg² (16×) | 3 | **> 2 h 40 m** | — |

Two results, and neither is the one first assumed:

**Volume is a wash.** Scaling the 1.5 deg² clip over the ~840 deg² the 664 tiles cover
gives ~7 GB a year, ~58 GB for 2017–2024, against the S3 route's 80.7 GB. An earlier
version of this document dismissed the FME route on *uncompressed* raster size — which is
not what crosses the wire — and was wrong to.

**Job cost is super-linear in area.** Sixteen times the AOI cost more than fifty-five
times the time. So a country-sized clip is not a viable unit of work, and staying in the
efficient regime means ~1.5 deg² jobs — roughly **4 500 asynchronous jobs** for eight
years of Europe, each needing submit, poll, download and verify, against a shared public
service with no cancel endpoint, a duplicate guard that blocks re-requesting an identical
clip while a stale task lives, and a demonstrated ability to lose a job for six days.

That is the real case for S3: **granularity and retryability**, not gigabytes. 15 936
immutable objects, each with a length and an MD5, fetched in parallel and re-fetched
individually.

**The queue is fast, healthy and concurrent.** An earlier reading of this said
otherwise: task `66068647426` sat in `Queued` for six days, then turned up `Cancelled` —
an individually dead task, not a busy queue. And jobs genuinely run in parallel: a
0.25 deg² clip submitted *while* the 24 deg² job was still running finished in 22 seconds.
So a job stuck in `Queued` for days should be cleared from the download cart and
resubmitted, not waited on, and nothing is queueing behind anything else.

**So the CDSE account is not a hard blocker.** A working, concurrent acquisition route
exists on the `~/.clms` key alone, and regional work — the PEP725-gap zones stage 1
actually needs — can start today. Route A is the right route for the full continental
build, for the granularity reasons above; it is not a prerequisite for progress.

---

## Phase 1 — Settle the hypothesis on one AOI (route B) — **done, passed**

Run on Brandenburg (12.0–13.5 °E, 52.0–53.0 °N), **season 2022**, task `3852689621` —
the same area as the German smoke test, so the results are checkable against
[`cropmodelling4eu/evaluation/germany_smoke_evaluation.ipynb`](../../cropmodelling4eu/evaluation/germany_smoke_evaluation.ipynb).
Results are in the table above and in the notebook's verdict.

Four things the notebook needed, now fixed in it:

1. **Identifiers pinned** rather than resolved by title each run. (Its `discover()` already
   avoided the confidence-layer trap correctly — that part needed nothing.)
2. **`BoundingBox` stays `[W, N, E, S]`.** The API prose says `[N, E, S, W]`; its own
   worked example contradicts it. The AOI-containment assert confirmed `[W, N, E, S]` is
   right — the returned bounds cover the AOI.
3. **`TemporalFilter` is epoch milliseconds in UTC**, capped at 366 days: one request per
   season year. It was building them in *local* time — see trap 1 at the end.
4. **`wait()` now refreshes the token.** The bearer token lives one hour; a job that
   queues longer 401s mid-poll, which is exactly how the previous attempt died.

Artefacts on disk under `/data01/FDS/muduchuru/Data/Agri/CLMS/aoi_probe/`: the 13 MB
delivered zip and the three 244 MB rasters, **renamed to the product's own convention**:

```
CLMS_HRLVLCC_CPMCE_S2022_R10m_brandenburg_03035.tif
```

The delivery does not name them usefully. It ships
`CLMS_HRLVLCC_CTY_bbox_0.tif` — where `bbox_<i>` is the dataset's *position in the request
payload*, so reordering the request makes the same name mean a different layer — inside a
folder named for the temporal filter's start (`20210101/`) rather than the season it holds,
which for season 2022 is a year out. The rename mirrors the published tiles
(`CLMS_HRLVLCC_CPMCE_S2017_R10m_E40N28_03035_V01_R00`) with the EEA tile id replaced by an
AOI slug, so season and area are in the name and nothing is positional.

Idempotence keys on those renamed rasters, not on the zip or the archive members, because
they are what everything downstream reads: all three present means no download and no
unpack, even if the zip has since been deleted.

**Still to do in this phase:** a second AOI — southern Spain or Denmark — before
generalising. The 1 January winter/spring cut worked at 52 °N but almost certainly needs to
be latitude-dependent. And repeating 2017–2024 over Brandenburg, which is what separates
the `CPMCH` product bias from the 2022 drought.

## Phase 2 — Open the bulk route

Create a CDSE account at `dataspace.copernicus.eu` and issue S3 keys. This is the only step
that cannot be scripted here; everything in phase 3 is written against it and testable on a
single tile the moment keys exist.

## Phase 3 — Bulk fetch (route A, if phase 1 passes)

`scripts/clms_fetch.py`, built on the staged manifests:

- **Reproducible inventory** (`scripts/clms_inventory.py`) — read the three CDSE catalogue
  CSVs, parse the `bbox` WKT, intersect with `site/site.csv` cell centres, emit
  `clms_manifest_export.parquet`. This is the logic that produced the 664-tile figure;
  promoting it out of scratch makes the number reproducible rather than quoted.
- **Fetch once, verify, never refetch.** The manifest carries `checksum_value` (MD5) and
  `content_length` per object. Skip any file already on disk at the right size, so a
  re-run over 15 936 files costs nothing and an interrupted bulk fetch resumes where it
  stopped.

  **Download to a `.part` sibling and rename only on completion.** A bare `path.exists()`
  guard is worse than no guard: an interrupted transfer leaves a truncated file that
  passes the check forever, and the next run sails past it into corrupt data. With the
  rename, the file's *existence* is itself proof it completed, and the size/MD5 check is
  a second, independent guard against anything left by an earlier tool. This is the
  pattern the notebook's download cell now uses — verified both ways, cached and
  truncated.
- **Target** `/data01/FDS/muduchuru/Data/Agri/CLMS/tiles/<product>/<year>/` — beside the other
  reference datasets (`cybench`, `EnSv8`, `PEP725`, `SAGE_crop_calendar`) rather than on
  scratch, so the acquisition is a permanent pipeline input, not run output.
- **SLURM array over tiles**, patterned on `cropmodelling4eu/submit/submit_torchcrop.sh`.

Order the fetch by need: the EnS zones PEP725 cannot reach first, so stage 1 can start on
the Mediterranean and Nordic gap before the full 80.7 GB lands.

## Phase 4 — Reduce tiles to a calibration observation set

`src/cropmodelling4eu/calibration/clms.py`. This is where 80 GB becomes something stage 1
can consume.

- **Decode `YYDOY`** as `YY = v // 1000`, `DOY = v % 1000`, to an **absolute date** and then
  to days since an anchor of 1 August of `SEASON_YEAR − 1`. On that axis winter emergence
  lands near +30…+120 and harvest near +330…+380, both monotone, so an ordinary median
  aggregates them and no circular statistics are needed. The notebook's `decode_yydoy`
  already does this — lift it, do not rewrite it.
- **Mask** to `CTY == 1110` ∧ emergence before the (latitude-dependent) cut.
- **Bin 10 m pixel centres** into the 0.1° cells — never resample. This is the ECIRA
  precedent in the repo `CLAUDE.md`: averaging class `1110` with `1130` gives "barley", and averaging
  `22300` with `23100` gives a date in the wrong year. Only nearest-neighbour is safe on
  either layer.
- **Carry the evidence with the value**: `n_pixels` and `winter_share` per cell, mirroring
  how the soil stage carries `share_percent`. A 0.1° cell holds ~740 000 pixels of 10 m; one
  whose date rests on a few hundred is describing a couple of fields, and the loss weight
  must know that.
- **Emit into `ObservationSet`'s schema** — `(unit, year, variable, value, weight, support)`
  — as `variable ∈ {emergence_doy, harvest_doy}`, so stage 1 consumes CLMS, PEP725 and
  CyBench through one path.

Sowing-from-emergence, if stage 1 wants `site.idpl` from CLMS: back off a degree-day
germination requirement using the MSWX temperature the pipeline already loads, and label it
in `calendar_source` the way the SAGE path labels `nearest` / `fallback`.

## Files

CLMS is a **site-stage calendar source**, so acquisition and decoding belong to
`data4simplace` next to `site/calendar.py`'s `sage` and `ggcmi` readers. Only the bridge
into the calibration's observation schema stays in `cropmodelling4eu`.

| Path | Role |
| --- | --- |
| `notebooks/clms_winter_wheat_phenology.ipynb` | pin identifiers, run phase 1's four tests, fill in the verdict |
| `src/data4simplace/site/clms.py` | fetch, decode `YYDOY`, mask to winter wheat, bin to 0.1° |
| `src/data4simplace/site/calendar.py` | extend `site.calendar_source` with `clms`, beside `sage` / `ggcmi` |
| `scripts/clms_inventory.py` | reproducible manifest + tile selection |
| `scripts/clms_fetch.py` | checksum-verified S3 fetch, resumable |
| `cropmodelling4eu/submit/clms_array.sh` | SLURM array, patterned on `torchcrop_array.sh` |
| `cropmodelling4eu/src/cropmodelling4eu/calibration/clms.py` | read the gridded product, emit `ObservationSet` rows |

## Verification

1. **Grid assertions.** Cell 16's `check_grids` compares the *affine transform*, not the
   shape, at `rtol=0`. That is load-bearing: EPSG:3035 eastings are ~4e6, so `np.allclose`'s
   default `rtol=1e-5` tolerates a 40 m — four-pixel — misalignment and silently passes a
   bad grid. Keep it, and keep cell 18's AOI-containment assert.
2. **Hypothesis gate** — the four tests above, numbers written into the notebook's verdict.
   A negative result is a valid, recorded outcome.
3. **Fetch integrity** — MD5 and byte count per object against the manifest; a second run
   must download nothing.
4. **Two paths, one number** — for Brandenburg, the route-A tile result must reproduce the
   route-B server-clipped result cell for cell.
5. **Against PEP725** — in Germany, where both exist, compare CLMS emergence and harvest
   against PEP725 BBCH 10 and 100 through `cropmodelling4eu/evaluation/germany.py`'s existing station→cell
   matcher. Agreement there is what licenses trusting CLMS where PEP725 is absent, which is
   the entire reason for the acquisition.

## Caveats

- **2017–2024 only.** Eight of the calibration's 25 seasons. Fine for time-invariant
  phenology parameters; useless for pre-2017 anomaly terms.
- **The winter/spring cut is latitude-dependent** and phase 1 tests it at one latitude only.
- **`CTY` is a single-crop-type map per year.** A cell's wheat pixels move between years, so
  `n_pixels` is a per-year quantity, not a fixed mask.
- Two credentials, easily confused: `~/.clms` (a CLMS service key, RSA private key signed
  into a JWT grant) drives `land.copernicus.eu/api`; a **CDSE** account (OIDC or S3 keys)
  drives per-tile downloads. Neither substitutes for the other.


## Three traps this cost, recorded so the next run does not pay them

1. **`epoch_ms` used `time.mktime`**, which reads the date tuple as *local* time. On a
   UTC+1 machine a 2022 filter began at 2021-12-31T23:00Z — one hour inside the previous
   season — and the delivered folder came back labelled `20210101`. The payload was still
   season 2022 (decoding found YY 2021 autumn and 2022 spring, and no 2020 at all), but a
   filter straddling two seasons is one scheduling decision from returning the wrong one.
   Use `calendar.timegm`.
2. **The CRS comes back as `IGNF:ETRS89LAEA`, not `EPSG:3035`.** Same LAEA method, same
   ETRS89 datum and GRS80 ellipsoid, identical parameters; only the axis order differs
   (IGNF east/north, EPSG:3035 formally north/east), and GeoTIFF stores x/y regardless.
   That is still enough to make `to_epsg()` return `None` and `CRS.equals` return `False`,
   either of which fails a raster that is exactly correct. `to_epsg(min_confidence=20)`
   resolves it to 3035 — confidence is docked for the authority label, not the geodesy.
3. **The FME result host 403s `python-requests`' default User-Agent.** It looks precisely
   like an auth failure and is not; a browser UA gets 200 on the same URL. No emailed link
   or extra credential is needed.
4. **Densify a lon/lat AOI before projecting it.** A lon/lat box is not a rectangle in
   LAEA — its east–west edges bow outward. The notebook's containment check projected
   only the SW and NE corners, understating the AOI by ~3 km per side, so it would have
   accepted a raster 3 km short. It also demanded exact containment and so failed on the
   4.4 m of sub-pixel snapping the service applies at the clip edge. Densified boundary
   plus one-pixel tolerance fixes both.
5. **A recorded TaskID is only good while its task lives.** `66068647426` went
   `Cancelled`, and a hard-coded dead id stops the notebook on every later run. Check the
   status before trusting an id, and resubmit a dead one.

With those fixed, `check_grids` passes on the delivered data: all three layers on the
native 10 m grid with identical affine transforms, so the element-wise masking the whole
method depends on is sound.
