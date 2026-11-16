"""Figure style: one palette, one set of mark specs, two modes.

Every figure in this project is built from the roles defined here rather than
from literal colours, so the whole set stays consistent and a mode switch is
one call. The palette is the validated reference instance from the project's
data-visualisation guidance, with the project's own green/teal set in the
first four categorical slots: eight categorical hues in a fixed order, a
single-hue blue ramp for magnitude, and a blue-to-red diverging ramp with a
neutral grey midpoint for polarity.

Three rules the figures in :mod:`utils.plots` follow from it:

* **Never cycle the categorical hues.** With 23 countries on screen, identity
  cannot be carried by colour — those figures use one hue and separate the
  countries by position (small multiples, a sorted axis) instead.
* **Sequential means magnitude, diverging means polarity.** A bias, which has a
  meaningful zero, gets the diverging ramp centred on zero. A mean yield gets
  the single-hue ramp.
* **Text never wears the data colour.** Labels and annotations use the ink
  tokens; identity comes from the mark beside them.
"""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Literal

import colormaps
import matplotlib as mpl
import numpy as np
from matplotlib import font_manager
from matplotlib.colors import TwoSlopeNorm

__all__ = [
    "DIVERGING_CMAP",
    "MODEL_MARKERS",
    "MODES",
    "SEQUENTIAL_CMAP",
    "SERIES",
    "Mode",
    "diverging_norm",
    "palette",
    "styled",
    "use_poster_style",
    "use_style",
]

Mode = Literal["light", "dark"]

#: The categorical slots, in a fixed order. Slots 1-4 are the project's own
#: green/teal set (2026-11-10), ordered so the first three -- the most a chart
#: here uses -- are the most separated. Slot 3 was #37584f; it is now deepened
#: to #193d34 (same hue, OKLab L 0.43 -> 0.33), because #37584f sat 9.6 OKLab
#: dE x100 from the olive under tritanopia. Worst pair among slots 1-3 is now
#: 13.7 (teal vs olive, normal vision; >= 12.8 under any colour-vision
#: deficiency), and slot 3 is >= 19.4 from both. Slot 4 (#366c62) is 16.0 from
#: slot 3. Charts still label every series, since 13.7 is under the 15 that
#: colour alone needs. Dark mode raises slots 1, 2 and 4 to OKLab L 0.70 and
#: slot 3 to 0.88, so it keeps its distinct lightness (>= 15.9 from slots 1-2).
#: Slots 5-8 are the previous validated hues, for the few charts that need them.
SERIES: dict[Mode, tuple[str, ...]] = {
    "light": ("#009999", "#5e7127", "#193d34", "#366c62", "#e87ba4",
              "#008300", "#4a3aa7", "#e34948"),
    "dark": ("#37b3b2", "#93a95f", "#bae2d6", "#74aba0", "#d55181",
             "#008300", "#9085e9", "#e66767"),
}

#: Fixed colours for the two models and torchcrop's two devices, following the
#: TorchCrop poster (ZALF, 2026-10-12): SIMPLACE is the grey reference,
#: torchcrop CPU olive, torchcrop GPU teal; ``torchcrop`` alone (no device in
#: the figure) is the poster's teal. Exposed by :func:`palette` as the
#: ``simplace``, ``torchcrop``, ``torchcrop_cpu`` and ``torchcrop_gpu`` roles. Teal vs olive is
#: 14.9 OKLab dE x100 at worst (deuteranopia), and both are >= 20 from the grey
#: in lightness and chroma, so the GPU series no longer sits next to the CPU one.
MODEL_COLOURS: dict[Mode, dict[str, str]] = {
    "light": {"simplace": "#a8a7a0", "torchcrop": "#009999",
              "torchcrop_cpu": "#5e7127", "torchcrop_gpu": "#009999"},
    "dark": {"simplace": "#7d7c76", "torchcrop": "#37b3b2",
             "torchcrop_cpu": "#93a95f", "torchcrop_gpu": "#37b3b2"},
}

#: The poster's secondary encoding, so identity never rests on colour alone:
#: torchcrop CPU squares, GPU circles, SIMPLACE a plain line.
MODEL_MARKERS: dict[str, str | None] = {
    "simplace": None, "torchcrop_cpu": "s", "torchcrop_gpu": "o",
}

#: Surfaces and ink, by mode.
MODES: dict[Mode, dict[str, str]] = {
    "light": {
        "surface": "#fcfcfb",
        "page": "#f9f9f7",
        "primary": "#0b0b0b",
        "secondary": "#52514e",
        "muted": "#898781",
        "grid": "#e1e0d9",
        "baseline": "#c3c2b7",
        "land": "#f2f1ed",
        "ocean": "#eef2f5",
    },
    "dark": {
        "surface": "#1a1a19",
        "page": "#0d0d0d",
        "primary": "#ffffff",
        "secondary": "#c3c2b7",
        "muted": "#898781",
        "grid": "#2c2c2a",
        "baseline": "#383835",
        "land": "#232322",
        "ocean": "#1e2224",
    },
}

#: Perceptually uniform, colour-vision-deficiency-safe sequential ramp from
#: Fabio Crameri's scientific colour maps (via the ``colormaps`` package): the
#: encoding for magnitude. Multi-hue rather than single-hue blue, so a wide
#: dynamic range (a continental yield field) stays resolvable where a
#: single-hue ramp would compress into indistinguishable dark blues.
SEQUENTIAL_CMAP = colormaps.batlow

#: Perceptually uniform diverging ramp, blue to red through a near-white
#: midpoint: the encoding for a signed quantity. The midpoint reads as
#: "nothing" the same way the project's former neutral grey did; a hue there
#: would invent a third category.
DIVERGING_CMAP = colormaps.vik


def palette(mode: Mode = "light") -> dict[str, str]:
    """Colour roles for one mode.

    Args:
        mode: ``"light"`` or ``"dark"``.

    Returns:
        The surface/ink roles of :data:`MODES`, ``series_1``..``series_8`` and
        the model roles of :data:`MODEL_COLOURS`.

    Raises:
        KeyError: On an unknown mode.
    """
    if mode not in MODES:
        raise KeyError(f"unknown mode {mode!r}; expected one of {list(MODES)}")
    roles = dict(MODES[mode])
    roles.update({f"series_{i}": c for i, c in enumerate(SERIES[mode], start=1)})
    roles.update(MODEL_COLOURS[mode])
    if _POSTER_INK:
        # Poster figures set every piece of text in black, including the
        # labels the plot helpers draw in the secondary ink.
        roles.update(primary="black", secondary="black")
    return roles


def _rc(mode: Mode, base_size: float) -> dict[str, object]:
    """Matplotlib rcParams implementing the mark and chrome specs."""
    p = palette(mode)
    return {
        "figure.facecolor": p["surface"],
        "figure.dpi": 110,
        "savefig.dpi": 300,
        "savefig.facecolor": p["surface"],
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,

        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica", "sans-serif"],
        "font.size": base_size,
        "axes.titlesize": base_size + 1.5,
        "axes.labelsize": base_size,
        "xtick.labelsize": base_size - 1,
        "ytick.labelsize": base_size - 1,
        "legend.fontsize": base_size - 1,

        "text.color": p["primary"],
        "axes.labelcolor": p["secondary"],
        "axes.titlecolor": p["primary"],
        "xtick.color": p["muted"],
        "ytick.color": p["muted"],
        "xtick.labelcolor": p["secondary"],
        "ytick.labelcolor": p["secondary"],

        "axes.facecolor": p["surface"],
        "axes.edgecolor": p["baseline"],
        "axes.linewidth": 0.8,
        # Only the left and bottom rules survive; a full box is chrome that
        # competes with the data.
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titlelocation": "left",
        "axes.titlepad": 10.0,
        "axes.axisbelow": True,

        # Hairline, solid, one step off the surface. Never dashed: a dashed
        # grid reads as a threshold.
        "grid.color": p["grid"],
        "grid.linewidth": 0.7,
        "grid.linestyle": "-",
        "axes.grid": True,
        "axes.grid.axis": "both",

        "lines.linewidth": 2.0,
        "lines.solid_capstyle": "round",
        "lines.solid_joinstyle": "round",
        "lines.markersize": 5.0,

        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,

        "legend.frameon": False,
        "legend.handlelength": 1.6,
        "legend.borderaxespad": 0.0,

        "axes.prop_cycle": mpl.cycler(color=list(SERIES[mode])),
        "image.cmap": "d4s_blue" if "d4s_blue" in mpl.colormaps else "viridis",
    }


#: Segoe UI is a licensed Microsoft font and is not on the cluster. Selawik is
#: Microsoft's open (OFL) metric-compatible fallback for it: PNG and PDF embed
#: Selawik, and SVG keeps text as text naming Segoe UI first, so it renders in
#: Segoe UI wherever that is installed.
POSTER_FONT_DIR = Path(os.environ.get(
    "CM4EU_FONT_DIR", "/data01/FDS/muduchuru/Data/SIMPLACE/fonts/selawik/zip"))
POSTER_FONTS = ["Segoe UI", "Selawik", "DejaVu Sans"]
_POSTER_INK = False


def use_poster_style(base_size: float = 9.5 * 1.25) -> dict[str, str]:
    """The TorchCrop poster's look on top of :func:`use_style`, globally.

    Segoe UI (Selawik fallback), all text black, black 1.2 pt spines and ticks
    (1.5x the base 0.8 pt), bold titles, and text 1.25x the base size. The
    palette's ``primary``/``secondary`` ink turns black too, so labels the plot
    helpers draw follow. :func:`use_style` restores the default look.

    Returns:
        The light palette with the poster ink.
    """
    global _POSTER_INK
    for ttf in sorted(POSTER_FONT_DIR.glob("*.ttf")):
        font_manager.fontManager.addfont(str(ttf))
    # Segoe UI is looked up first on purpose; its absence is not news.
    logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
    use_style("light", base_size)
    _POSTER_INK = True
    mpl.rcParams.update({
        # A list as font.family, not font.sans-serif: only then does matplotlib
        # fall back per glyph, drawing what Selawik lacks (the minus sign of
        # every negative tick, ≥, →, Δ) from DejaVu Sans instead of dropping it.
        "font.family": POSTER_FONTS,
        "svg.fonttype": "none", "pdf.fonttype": 42,
        "text.color": "black", "axes.labelcolor": "black", "axes.titlecolor": "black",
        "axes.titleweight": "bold",
        "xtick.labelcolor": "black", "ytick.labelcolor": "black",
        "xtick.color": "black", "ytick.color": "black",
        "axes.edgecolor": "black", "axes.linewidth": 1.2,
        "xtick.major.width": 1.2, "ytick.major.width": 1.2,
    })
    return palette("light")


def use_style(mode: Mode = "light", base_size: float = 9.5) -> dict[str, str]:
    """Apply the project style globally and return its palette.

    Call once at the top of a notebook::

        from cropmodelling4eu.evaluation.style import use_style
        P = use_style()

    Args:
        mode: ``"light"`` (the default, and what the saved figures are for) or
            ``"dark"``.
        base_size: Body font size in points. Raise it for slides.

    Returns:
        The palette dict for ``mode``, so the notebook can reach individual
        roles without a second import.
    """
    for name, cmap in (("d4s_blue", SEQUENTIAL_CMAP), ("d4s_blue_red", DIVERGING_CMAP)):
        if name not in mpl.colormaps:
            mpl.colormaps.register(cmap, name=name)
    global _POSTER_INK
    _POSTER_INK = False
    mpl.rcParams.update(_rc(mode, base_size))
    return palette(mode)


@contextmanager
def styled(mode: Mode = "light", base_size: float = 9.5) -> Iterator[dict[str, str]]:
    """Context manager form of :func:`use_style`, restoring the previous rc."""
    saved = mpl.rcParams.copy()
    try:
        yield use_style(mode, base_size)
    finally:
        mpl.rcParams.update(saved)


def diverging_norm(values, vmax: float | None = None) -> TwoSlopeNorm:
    """A zero-centred norm for the diverging ramp.

    Centring on zero is what makes the ramp honest: without it, an all-positive
    bias field would be painted half blue and half red and read as a sign
    change that is not there.

    Args:
        values: The data the ramp will span.
        vmax: Half-range. ``None`` uses the largest absolute finite value, so
            the two arms are symmetric.

    Returns:
        A :class:`~matplotlib.colors.TwoSlopeNorm` centred on 0.
    """
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    limit = float(vmax) if vmax is not None else (
        float(np.max(np.abs(finite))) if finite.size else 1.0
    )
    limit = limit if limit > 0 else 1.0
    return TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)
