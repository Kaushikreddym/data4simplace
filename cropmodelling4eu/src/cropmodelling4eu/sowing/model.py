"""The three-branch sowing-hazard network and its first-event distribution.

::

    logit_t = slope * (t - mu) / 7 + c        long-term: where in the calendar
            + head([FiLM(TCN(short))_t,       short-term: the trigger, <= 21 d
                    MLP(medium_t)])           medium-term: readiness

``mu`` is the SAGE window start shifted by a learned, normals-driven offset.
The short-term branch sees neither the day of year nor any static input, so it
cannot memorise a calendar; the long-term branch only modulates it.

``P(sow = s) = h_s * prod_{tau < s} (1 - h_tau)`` with ``h_{S-1} = 1``: the
deadline, like ``vSowForceAtWindowEnd``, guarantees a crop every season.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig

__all__ = ["SowingHazardNet", "first_event_log_probs", "laplace_mixture_nll"]


class _CausalBlock(nn.Module):
    """Residual dilated conv that only looks back."""

    def __init__(self, channels: int, dilation: int, dropout: float) -> None:
        super().__init__()
        self.pad = 2 * dilation
        self.conv = nn.Conv1d(channels, channels, 3, dilation=dilation)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.drop(F.gelu(self.conv(F.pad(x, (self.pad, 0)))))


class SowingHazardNet(nn.Module):
    """Per-day sowing logits over the candidate days ``0 .. sowing_days - 1``."""

    def __init__(self, n_short: int, n_medium: int, n_long: int,
                 sowing_days: int, config: ModelConfig) -> None:
        super().__init__()
        c, m, l = config.short_channels, config.medium_channels, config.long_channels
        self.sowing_days = sowing_days
        self.short_in = nn.Conv1d(n_short, c, 1)
        self.tcn = nn.Sequential(*(_CausalBlock(c, d, config.dropout) for d in config.short_dilations))
        self.medium = nn.Sequential(nn.Linear(n_medium, m), nn.GELU(), nn.Linear(m, m), nn.GELU())
        self.long = nn.Sequential(nn.Linear(n_long, l), nn.GELU(), nn.Dropout(config.dropout),
                                  nn.Linear(l, l), nn.GELU())
        self.film = nn.Linear(l, 2 * c)
        self.mu_shift = nn.Linear(l, 1)
        self.head = nn.Sequential(nn.Linear(c + m, c), nn.GELU(), nn.Linear(c, 1))
        self.slope = nn.Parameter(torch.tensor(0.0))
        self.offset = nn.Parameter(torch.tensor(-3.0))
        for layer in (self.film, self.mu_shift):
            nn.init.zeros_(layer.weight), nn.init.zeros_(layer.bias)

    def forward(self, short: torch.Tensor, medium: torch.Tensor, long: torch.Tensor,
                window_start: torch.Tensor) -> torch.Tensor:
        """``[B, S]`` logits.

        Args:
            short: ``[B, T, n_short]`` normalised daily weather.
            medium: ``[B, T, n_medium]`` normalised daily state.
            long: ``[B, n_long]`` normalised normals and statics.
            window_start: ``[B]`` SAGE window start, cache index (raw).
        """
        s = self.sowing_days
        h = self.tcn(self.short_in(short.transpose(1, 2)))[..., :s].transpose(1, 2)
        z = self.long(long)
        gamma, beta = self.film(z).unsqueeze(1).chunk(2, dim=-1)
        h = h * (1.0 + gamma) + beta
        resid = self.head(torch.cat([h, self.medium(medium[:, :s])], dim=-1)).squeeze(-1)
        mu = window_start + 15.0 * self.mu_shift(z).squeeze(-1)
        t = torch.arange(s, device=short.device, dtype=short.dtype)
        base = F.softplus(self.slope) * (t - mu.unsqueeze(1)) / 7.0 + self.offset
        return base + resid


def first_event_log_probs(logits: torch.Tensor) -> torch.Tensor:
    """``log P(sow = s)`` from per-day logits, the last day forced."""
    log_h = F.logsigmoid(logits)
    log_h = torch.cat([log_h[:, :-1], torch.zeros_like(log_h[:, -1:])], dim=1)
    log_stay = F.logsigmoid(-logits[:, :-1])
    survive = F.pad(log_stay.cumsum(dim=1), (1, 0))
    return log_h + survive


def laplace_mixture_nll(log_p: torch.Tensor, centres: torch.Tensor,
                        observed: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """``-log sum_s P(s) Laplace(observed; centres_s, scale)`` per row.

    ``centres`` is ``[B, S]`` (the emergence day each sowing day leads to, or
    the sowing day itself), ``observed`` ``[B]``.
    """
    scale = scale.expand_as(observed).unsqueeze(1)
    log_k = -(observed.unsqueeze(1) - centres).abs() / scale - torch.log(2.0 * scale)
    return -torch.logsumexp(log_p + log_k, dim=1)
