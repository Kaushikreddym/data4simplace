"""Figure style: one palette, one set of mark specs, two modes.

Every figure in this project is built from the roles defined here rather than
from literal colours, so the whole set stays consistent and a mode switch is
one call. The palette is the validated reference instance from the project's
data-visualisation guidance: eight categorical hues in a fixed order, a
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

from contextlib import contextmanager
from typing import Iterator, Literal

import matplotlib as mpl
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

__all__ = [
    "DIVERGING_CMAP",
    "MODES",
    "SEQUENTIAL_CMAP",
    "SERIES",
    "Mode",
    "diverging_norm",
    "palette",
    "styled",
    "use_style",
]

Mode = Literal["light", "dark"]

#: The eight categorical slots, in the order that passes the colour-vision
#: separation checks. Charts here use at most the first three, which are the
#: slots validated for all-pairs forms (scatter, maps, small multiples).
SERIES: dict[Mode, tuple[str, ...]] = {
    "light": ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4",
              "#008300", "#4a3aa7", "#e34948"),
    "dark": ("#3987e5", "#d95926", "#199e70", "#c98500", "#d55181",
             "#008300", "#9085e9", "#e66767"),
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

#: Single-hue blue ramp, light to dark: the sequential encoding for magnitude.
_BLUE_STEPS: tuple[str, ...] = (
    "#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
    "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b",
)

SEQUENTIAL_CMAP = LinearSegmentedColormap.from_list("d4s_blue", _BLUE_STEPS, N=256)

#: Blue to red through a neutral grey: the diverging encoding for a signed
#: quantity. Grey at the midpoint is what makes zero read as "nothing"; a hue
#: there would invent a third category.
DIVERGING_CMAP = LinearSegmentedColormap.from_list(
    "d4s_blue_red",
    ["#0d366b", "#256abf", "#6da7ec", "#cde2fb", "#f0efec",
     "#f7c9c9", "#e88b8b", "#d03b3b", "#8f1f1f"],
    N=256,
)


def palette(mode: Mode = "light") -> dict[str, str]:
    """Colour roles for one mode.

    Args:
        mode: ``"light"`` or ``"dark"``.

    Returns:
        The surface/ink roles of :data:`MODES` plus ``series_1``..``series_8``.

    Raises:
        KeyError: On an unknown mode.
    """
    if mode not in MODES:
        raise KeyError(f"unknown mode {mode!r}; expected one of {list(MODES)}")
    roles = dict(MODES[mode])
    roles.update({f"series_{i}": c for i, c in enumerate(SERIES[mode], start=1)})
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
