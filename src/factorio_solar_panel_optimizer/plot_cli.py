from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from . import parameters
from .support import panel_plot, utilities


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render a saved solar-array solution as interactive Plotly HTML."
    )
    parser.add_argument("solution", type=Path)
    parser.add_argument(
        "--planet",
        choices=sorted(parameters.PLANETS),
        default="nauvis",
    )
    parser.add_argument(
        "--roboport",
        choices=("temporary", "permanent"),
        default="temporary",
    )
    parser.add_argument("--grid", type=int, default=50)
    parser.add_argument("--electric-coverage", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        help="HTML output path; defaults beside the solution file.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Write HTML without opening the browser.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()

    parameters.configure_planet(args.planet)
    parameters.GRID_SIZE = args.grid

    solution = utilities.load_solution(args.solution)
    solution = np.rint(solution).astype(int)
    (
        solar_panels,
        accumulators,
        substations,
        roboports,
        medium_poles,
    ) = utilities.state_vector_to_coordinates(solution)

    replacement_accumulators = (
        4 * len(roboports) if args.roboport == "temporary" else 0
    )
    effective_accumulators = len(accumulators) + replacement_accumulators

    solar_power = (
        len(solar_panels)
        * parameters.SOLAR_PANEL_POWER
        * parameters.ETA_S
    )
    accumulator_power = (
        effective_accumulators
        * parameters.ACCUMULATOR_CHARGE
        / parameters.DAY_DURATION
        / parameters.C_ON
    )
    power = min(solar_power, accumulator_power)

    fig = panel_plot.plot_solar_array_periodic(
        args.grid,
        solar_panels,
        accumulators,
        substations,
        roboports,
        medium_poles,
        plot_electric=args.electric_coverage,
    )
    fig.update_layout(
        title=(
            f"{args.planet.title()} — sustained power "
            f"{power / 1000:.3f} MW"
        )
    )

    output = args.output or args.solution.with_suffix(".html")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(output, include_plotlyjs=True)

    print(f"Solar panels: {len(solar_panels)}")
    print(f"Accumulators: {len(accumulators)}")
    if replacement_accumulators:
        print(f"Post-roboport accumulators: +{replacement_accumulators}")
    print(f"Substations: {len(substations)}")
    print(f"Medium poles: {len(medium_poles)}")
    print(f"Sustained power: {power / 1000:.6f} MW")
    print(f"Plot: {output.resolve()}")

    if not args.no_show:
        fig.show()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
