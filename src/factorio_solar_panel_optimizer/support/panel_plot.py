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
    coords = np.asarray(list(coords), dtype=float).reshape(-1, 2)
    if coords.size == 0:
        return

    shifts = (-domain_size, 0, domain_size) if periodic else (0,)
    for dx in shifts:
        for dy in shifts:
            for x, y in coords:
                fig.add_shape(
                    type="rect",
                    x0=x + dx,
                    y0=y + dy,
                    x1=x + dx + structure_size,
                    y1=y + dy + structure_size,
                    fillcolor=fillcolor,
                    line={"color": linecolor, "width": line_width},
                    layer="below",
                )

    # One legend entry per entity type; shapes themselves do not need to carry
    # thousands of legend entries.
    go = _go()
    fig.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="markers",
            marker={
                "symbol": "square",
                "size": 12,
                "color": fillcolor,
                "line": {"color": linecolor, "width": line_width},
            },
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
    """Build an interactive Plotly figure for one periodic 50x50 cell."""

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
