# Examples

Hands-on Jupyter notebooks. Notebooks 1–3 are **fully self-contained** — they
run anywhere the package is installed, using synthetic data where a real dataset
would otherwise be needed. Notebook 4 shows the real end-to-end run and points at
the input datasets it requires.

| # | Notebook | Needs external data? |
| --- | --- | --- |
| 1 | [Quickstart](01_quickstart.ipynb) — load & validate a config, inspect the plan | No |
| 2 | [The 10 km target grid](02_target_grid.ipynb) — cells, `SimplaceID`, regridding | No |
| 3 | [Pedotransfer functions](03_pedotransfer_functions.ipynb) — Saxton–Rawls hydraulics | No |
| 4 | [Running the full pipeline](04_full_pipeline.ipynb) — MSWX + SoilGrids + NPK → SIMPLACE | Yes |

!!! tip "Run them yourself"
    ```bash
    pip install -e .[dev]
    pip install jupyter matplotlib
    jupyter lab docs/examples
    ```
