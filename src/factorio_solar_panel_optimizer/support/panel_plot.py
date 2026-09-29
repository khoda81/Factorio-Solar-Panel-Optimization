from __future__ import annotations

from collections.abc import Iterable

import numpy as np


_COLORS = {
    "solar": ("rgba(119,181,254,0.55)", "rgb(71,109,152)"),
    "accumulator": ("rgba(51,179,77,0.55)", "rgb(31,107,46)"),
    "substation": ("rgba(153,166,179,0.55)", "rgb(92,100,107)"),
    "roboport": ("rgba(153,26,77,0.55)", "rgb(92,15,46)"),
    "medium": ("rgba(255,179,77,0.65)", "rgb(153,107,46)"),
    "coverage": ("rgba(191,191,191,0.16)", "rgba(191,191,191,0.0)"),
}


def _go():
    try:
        import plotly.graph_objects as go
    except ImportError as exc:
        raise RuntimeError(
            "Plotting is optional. Install it with: uv sync --extra plot"
        ) from exc
    return go


def _visible_periodic_rectangles(
    coords: Iterable[tuple[int, int]] | np.ndarray,
    structure_size: int,
    domain_size: int,
    periodic: bool,
):
    """Yield only rectangle copies that actually intersect the visible cell."""

    coords = np.asarray(list(coords), dtype=float).reshape(-1, 2)
    if coords.size == 0:
        return

    shifts = (-domain_size, 0, domain_size) if periodic else (0,)
    for x, y in coords:
        for dx in shifts:
            x0 = x + dx
            x1 = x0 + structure_size
            if x1 <= 0 or x0 >= domain_size:
                continue
            for dy in shifts:
                y0 = y + dy
                y1 = y0 + structure_size
                if y1 <= 0 or y0 >= domain_size:
                    continue
                yield x0, y0, x1, y1


def _add_rectangles(
    fig,
    coords: Iterable[tuple[int, int]] | np.ndarray,
    structure_size: int,
    *,
    domain_size: int,
    fillcolor: str,
    linecolor: str,
    periodic: bool,
    name: str,
    line_width: float = 1.0,
) -> None:
    """Add all same-style rectangles as one Plotly trace.

    Plotly layout shapes are SVG objects and become painfully slow in the
    thousands. One filled Scatter trace can represent every rectangle of an
    entity type with None-separated closed polygons, which keeps rendering
    essentially constant in trace count.
    """

    xs: list[float | None] = []
    ys: list[float | None] = []

    for x0, y0, x1, y1 in _visible_periodic_rectangles(
        coords,
        structure_size,
        domain_size,
        periodic,
    ):
        xs.extend((x0, x1, x1, x0, x0, None))
        ys.extend((y0, y0, y1, y1, y0, None))

    if not xs:
        return

    go = _go()
    fig.add_trace(
        go.Scatter(
            x=xs,
            y=ys,
            mode="lines",
            fill="toself",
            fillcolor=fillcolor,
            line={"color": linecolor, "width": line_width},
            name=name,
            hoverinfo="skip",
        )
    )


def plot_solar_array_periodic(
    size,
    solar_coords,
    acc_coords,
    subs_coords,
    robo_coords,
    pole_coords=None,
    *,
    plot_electric=False,
):
    """Build an interactive Plotly figure for one periodic cell."""

    go = _go()
    fig = go.Figure()

    if plot_electric:
        sub_coverage = np.asarray(subs_coords) - np.array([8, 8])
        _add_rectangles(
            fig,
            sub_coverage,
            18,
            domain_size=size,
            fillcolor=_COLORS["coverage"][0],
            linecolor=_COLORS["coverage"][1],
            periodic=True,
            name="Substation coverage",
            line_width=0,
        )
        if pole_coords is not None:
            pole_coverage = np.asarray(pole_coords) - np.array([3, 3])
            _add_rectangles(
                fig,
                pole_coverage,
                7,
                domain_size=size,
                fillcolor=_COLORS["coverage"][0],
                linecolor=_COLORS["coverage"][1],
                periodic=True,
                name="Medium-pole coverage",
                line_width=0,
            )

    for name, coords, structure_size, key in (
        ("Solar panel", solar_coords, 3, "solar"),
        ("Accumulator", acc_coords, 2, "accumulator"),
        ("Substation", subs_coords, 2, "substation"),
        ("Roboport", robo_coords, 4, "roboport"),
    ):
        _add_rectangles(
            fig,
            coords,
            structure_size,
            domain_size=size,
            fillcolor=_COLORS[key][0],
            linecolor=_COLORS[key][1],
            periodic=True,
            name=name,
        )

    if pole_coords is not None:
        _add_rectangles(
            fig,
            pole_coords,
            1,
            domain_size=size,
            fillcolor=_COLORS["medium"][0],
            linecolor=_COLORS["medium"][1],
            periodic=True,
            name="Medium electric pole",
        )

    fig.add_shape(
        type="rect",
        x0=0,
        y0=0,
        x1=size,
        y1=size,
        fillcolor="rgba(0,0,0,0)",
        line={"color": "black", "width": 2},
    )

    fig.update_xaxes(
        range=[0, size],
        dtick=1,
        showgrid=True,
        showticklabels=False,
        zeroline=False,
        constrain="domain",
    )
    fig.update_yaxes(
        range=[0, size],
        dtick=1,
        showgrid=True,
        showticklabels=False,
        zeroline=False,
        scaleanchor="x",
        scaleratio=1,
    )
    fig.update_layout(
        width=900,
        height=900,
        margin={"l": 20, "r": 20, "t": 70, "b": 20},
        legend={"orientation": "h", "y": 1.04, "x": 0},
        dragmode="pan",
    )
    return fig
