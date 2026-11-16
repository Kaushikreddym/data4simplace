"""Train the hazard network on CLMS emergence and PEP725 sowing/emergence.

Every row's emergence table ``E[b, s]`` is computed from the **raw** daily
mean temperature with the frozen ``tsumem`` and is never an input: weather
after a candidate sowing day reaches the network only as a loss term.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .cache import STATIC_FEATURES, SeasonCache
from .config import SowingConfig
from .emergence import EmergenceParams, emergence_index, read_teffmx
from .features import SHORT_CHANNELS
from .model import SowingHazardNet, first_event_log_probs, laplace_mixture_nll
from .targets import load_targets

logger = logging.getLogger(__name__)

__all__ = ["Normaliser", "Rows", "emergence_params", "load_model", "train"]

_DAVTMP = SHORT_CHANNELS.index("davtmp")


@dataclass(slots=True)
class Normaliser:
    """Per-channel z-scores, fitted on training rows only; NaN -> 0 after scaling."""

    short: tuple[np.ndarray, np.ndarray]
    medium: tuple[np.ndarray, np.ndarray]
    long: tuple[np.ndarray, np.ndarray]

    @classmethod
    def fit(cls, short: np.ndarray, medium: np.ndarray, long: np.ndarray) -> "Normaliser":
        stat = lambda x, axes: (np.nanmean(x, axis=axes).astype(np.float32),
                                np.maximum(np.nanstd(x, axis=axes), 1e-3).astype(np.float32))
        return cls(stat(short, (0, 1)), stat(medium, (0, 1)), stat(long, 0))

    def apply(self, x: np.ndarray, which: str) -> torch.Tensor:
        mean, std = getattr(self, which)
        return torch.from_numpy(np.nan_to_num((x.astype(np.float32) - mean) / std))

    def to_dict(self) -> dict:
        return {k: [v.tolist() for v in getattr(self, k)] for k in ("short", "medium", "long")}

    @classmethod
    def from_dict(cls, d: dict) -> "Normaliser":
        return cls(**{k: tuple(np.asarray(v, np.float32) for v in d[k]) for k in ("short", "medium", "long")})


@dataclass(slots=True)
class Rows:
    """A block of cache rows in memory: inputs, raw temperature and window start."""

    short: np.ndarray
    medium: np.ndarray
    long: np.ndarray
    window_start: np.ndarray

    @classmethod
    def load(cls, cache: SeasonCache, rows: np.ndarray) -> "Rows":
        order = np.argsort(rows)
        inverse = np.empty_like(order)
        inverse[order] = np.arange(order.size)
        sorted_rows = rows[order]
        cell = sorted_rows // len(cache.years)
        statics = cache.cells[list(STATIC_FEATURES)].to_numpy(np.float32)[cell]
        block = lambda mm: np.asarray(mm[sorted_rows])[inverse]
        long = np.concatenate([block(cache.long), statics[inverse]], axis=1)
        start = cache.cells["sage_start"].to_numpy(np.float32)[cell][inverse]
        return cls(block(cache.short).astype(np.float32), block(cache.medium).astype(np.float32),
                   long, np.nan_to_num(start, nan=np.nanmedian(start)))

    def batch(self, idx: np.ndarray, norm: Normaliser, sowing_days: int,
              params: EmergenceParams, device: torch.device) -> dict[str, torch.Tensor]:
        short = self.short[idx]
        davtmp = torch.from_numpy(short[..., _DAVTMP])
        candidates = torch.arange(sowing_days).expand(len(idx), -1)
        table = emergence_index(davtmp, candidates, params)
        table = torch.nan_to_num(table, nan=float(short.shape[1] - 1))
        return {
            "short": norm.apply(short, "short").to(device),
            "medium": norm.apply(self.medium[idx], "medium").to(device),
            "long": norm.apply(self.long[idx], "long").to(device),
            "window_start": torch.from_numpy(self.window_start[idx]).to(device),
            "emergence": table.float().to(device),
        }


def emergence_params(config: SowingConfig) -> EmergenceParams:
    """The frozen thermal constants: config value, else ``tsumem.json``."""
    tsumem = config.tsumem
    if tsumem is None:
        if not config.tsumem_file.is_file():
            raise FileNotFoundError(
                f"no tsumem: set sowing.tsumem or run `cm4eu sowing fit-tsumem` "
                f"(expected {config.tsumem_file})")
        tsumem = json.loads(config.tsumem_file.read_text())["tsumem"]
    return EmergenceParams(float(tsumem), config.tbasem, read_teffmx(config.crop_path))


class _Objective(torch.nn.Module):
    """The three likelihood terms with their learned Laplace scales."""

    def __init__(self, config: SowingConfig) -> None:
        super().__init__()
        w = config.loss
        self.w = w
        self.log_clms = torch.nn.Parameter(torch.tensor(np.log(w.clms_scale_init_days), dtype=torch.float32))
        self.log_pep = torch.nn.Parameter(torch.tensor(np.log(w.pep_emergence_scale_init_days), dtype=torch.float32))

    def forward(self, log_p: torch.Tensor, emergence: torch.Tensor, target: dict) -> tuple[torch.Tensor, dict]:
        s = torch.arange(log_p.shape[1], device=log_p.device, dtype=log_p.dtype).expand_as(log_p)
        terms, total = {}, log_p.new_zeros(())
        for name, mask, centres, obs, scale, weight in (
            ("clms_emergence", target["is_clms"] & target["has_em"], emergence, target["em"],
             self.log_clms.exp(), self.w.clms_emergence),
            ("pep_emergence", ~target["is_clms"] & target["has_em"], emergence, target["em"],
             self.log_pep.exp(), self.w.pep_emergence),
            ("pep_sowing", ~target["is_clms"] & target["has_sow"], s, target["sow"],
             log_p.new_tensor(self.w.pep_sowing_scale_days), self.w.pep_sowing),
        ):
            if mask.any():
                nll = laplace_mixture_nll(log_p[mask], centres[mask], obs[mask], scale).mean()
                terms[name] = float(nll)
                total = total + weight * nll
        return total, terms


def _targets(frame: pd.DataFrame, idx: np.ndarray, device: torch.device) -> dict:
    sub = frame.iloc[idx]
    t = lambda col: torch.from_numpy(np.nan_to_num(sub[col].to_numpy(np.float32))).to(device)
    flag = lambda x: torch.from_numpy(np.asarray(x)).to(device)
    return {"is_clms": flag(sub["source"].to_numpy() == "clms"),
            "has_em": flag(sub["em_idx"].notna().to_numpy()),
            "has_sow": flag(sub["sow_idx"].notna().to_numpy()),
            "em": t("em_idx"), "sow": t("sow_idx")}


def train(config: SowingConfig) -> Path:
    """Fit, early-stop on the validation loss, write ``model/`` and return it."""
    tc = config.train
    torch.manual_seed(tc.seed)
    torch.set_num_threads(tc.threads)
    device = torch.device(tc.device)
    rng = np.random.default_rng(tc.seed)

    cache = SeasonCache.open(config.cache_dir)
    params = emergence_params(config)
    targets = load_targets(config)
    targets["row"] = cache.rows(targets["SimplaceID"].to_numpy(), targets["year"].to_numpy())
    targets = targets[(targets["row"] >= 0)].reset_index(drop=True)
    targets = targets[cache.valid[targets["row"].to_numpy()]].reset_index(drop=True)
    # The sowing label must be a candidate day; one outside the window is a
    # spring sowing or a typo, not a sample the hazard can explain.
    late = targets["sow_idx"].notna() & ~targets["sow_idx"].between(0, config.sowing_days - 1)
    targets.loc[late, "sow_idx"] = np.nan
    targets = targets.dropna(subset=["sow_idx", "em_idx"], how="all").reset_index(drop=True)
    logger.info("usable targets: %s", targets.groupby(["source", "split"]).size().to_dict())

    used = targets[targets["split"].isin(("train", "val"))].reset_index(drop=True)
    data = Rows.load(cache, used["row"].to_numpy())
    train_idx = np.flatnonzero(used["split"].to_numpy() == "train")
    val_idx = np.flatnonzero(used["split"].to_numpy() == "val")
    sample = rng.choice(train_idx, min(20000, len(train_idx)), replace=False)
    norm = Normaliser.fit(data.short[sample], data.medium[sample], data.long[train_idx])

    model = SowingHazardNet(data.short.shape[2], data.medium.shape[2], data.long.shape[1],
                            config.sowing_days, config.model).to(device)
    objective = _Objective(config).to(device)
    optim = torch.optim.AdamW([*model.parameters(), *objective.parameters()],
                              lr=tc.lr, weight_decay=tc.weight_decay)

    is_pep = used["source"].to_numpy() == "pep725"
    pools = (train_idx[~is_pep[train_idx]], train_idx[is_pep[train_idx]])

    def evaluate(idx: np.ndarray) -> tuple[float, dict]:
        model.eval()
        losses, parts = [], {}
        with torch.no_grad():
            for chunk in np.array_split(idx, max(1, len(idx) // tc.batch_size)):
                b = data.batch(chunk, norm, config.sowing_days, params, device)
                log_p = first_event_log_probs(model(b["short"], b["medium"], b["long"], b["window_start"]))
                loss, terms = objective(log_p, b["emergence"], _targets(used, chunk, device))
                losses.append(float(loss) * len(chunk))
                for k, v in terms.items():
                    parts.setdefault(k, []).append(v)
        return sum(losses) / len(idx), {k: float(np.mean(v)) for k, v in parts.items()}

    out = config.model_dir
    out.mkdir(parents=True, exist_ok=True)
    best, stale, history = np.inf, 0, []
    for epoch in range(tc.max_epochs):
        model.train()
        n_pep = int(tc.samples_per_epoch * tc.pep_share) if len(pools[1]) else 0
        draw = np.concatenate([rng.choice(pools[0], tc.samples_per_epoch - n_pep),
                               rng.choice(pools[1], n_pep) if n_pep else np.empty(0, int)])
        rng.shuffle(draw)
        running = []
        for chunk in np.array_split(draw, len(draw) // tc.batch_size):
            b = data.batch(chunk, norm, config.sowing_days, params, device)
            log_p = first_event_log_probs(model(b["short"], b["medium"], b["long"], b["window_start"]))
            loss, _ = objective(log_p, b["emergence"], _targets(used, chunk, device))
            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            running.append(float(loss))
        val, parts = evaluate(val_idx)
        history.append({"epoch": epoch, "train": float(np.mean(running)), "val": val, **parts,
                        "clms_scale": float(objective.log_clms.exp()),
                        "pep_em_scale": float(objective.log_pep.exp())})
        logger.info("epoch %d: train %.3f val %.3f %s", epoch, history[-1]["train"], val, parts)
        if val < best - 1e-4:
            best, stale = val, 0
            torch.save(model.state_dict(), out / "model.pt")
        else:
            stale += 1
            if stale >= tc.patience:
                break

    pd.DataFrame(history).to_csv(out / "history.csv", index=False)
    (out / "model.json").write_text(json.dumps({
        "n_short": data.short.shape[2], "n_medium": data.medium.shape[2], "n_long": data.long.shape[1],
        "sowing_days": config.sowing_days, "cache_start_doy": config.cache_start_doy,
        "emergence": {"tsumem": params.tsumem, "tbasem": params.tbasem, "teffmx": params.teffmx},
        "normaliser": norm.to_dict(), "config": json.loads(config.model_dump_json()),
        "best_val": best, "epochs": len(history),
        "rows": {"train": int(len(train_idx)), "val": int(len(val_idx))},
    }, indent=1))
    logger.info("model -> %s (best val %.3f)", out, best)
    return out


def load_model(model_dir: Path) -> tuple[SowingHazardNet, Normaliser, dict]:
    """The trained network (eval mode), its normaliser and its metadata."""
    from .config import ModelConfig

    meta = json.loads((Path(model_dir) / "model.json").read_text())
    model = SowingHazardNet(meta["n_short"], meta["n_medium"], meta["n_long"], meta["sowing_days"],
                            ModelConfig(**meta["config"]["model"]))
    model.load_state_dict(torch.load(Path(model_dir) / "model.pt", map_location="cpu", weights_only=True))
    return model.eval(), Normaliser.from_dict(meta["normaliser"]), meta
