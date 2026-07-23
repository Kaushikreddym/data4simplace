# Command line

Installing the package registers the `data4simplace` console script (with
`simplace-pipeline` kept as a backwards-compatible alias). Both map to
`data4simplace.cli:main`.

## Synopsis

```bash
data4simplace [--config PATH] [--dry-run] [--verbose] [--version]
```

| Option | Description |
| --- | --- |
| `-c`, `--config PATH` | Path to the YAML config (default: `config.yaml`). |
| `--dry-run` | Validate the config and print the enabled stages; run nothing. |
| `-v`, `--verbose` | Debug-level logging (and full tracebacks on failure). |
| `--version` | Print the version and exit. |

## Exit codes

| Code | Meaning |
| --- | --- |
| `0` | Success (including a successful `--dry-run`). |
| `1` | A pipeline stage failed at runtime. |
| `2` | Configuration could not be loaded or failed validation. |

## Examples

```bash
# Check which stages are enabled without touching any data
data4simplace --config config.yaml --dry-run

# Full run with debug logging
data4simplace --config config.yaml --verbose

# Use a config in another location
data4simplace -c experiments/brandenburg_2020.yaml
```

A dry run reports the plan, for example:

```text
12:04:31 INFO     data4simplace.cli: Dry run — configuration valid.
                  Enabled stages: run_climate_processing, run_soil_processing,
                  run_npk_processing, apply_agricultural_mask,
                  export_simplace_weather, export_simplace_soil,
                  export_simplace_management
```
