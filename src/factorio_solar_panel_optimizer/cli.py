from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import re

import highspy
import numpy as np
import scipy.sparse as sp

from . import objectives, parameters


_SOLUTION_LINE = re.compile(
    r"^\s*x\[(\d+)\]\s+"
    r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)"
)


def _load_indexed_solution(path: Path, size: int) -> np.ndarray:
    values = np.zeros(size, dtype=float)
    seen = False
    with path.open(errors="replace") as handle:
        for line in handle:
            match = _SOLUTION_LINE.match(line)
            if not match:
                continue
            index = int(match.group(1))
            if index >= size:
                raise ValueError(f"{path} contains x[{index}], outside size {size}")
            values[index] = float(match.group(2))
            seen = True
    if not seen:
        raise ValueError(f"{path} contains no indexed x values")
    return values


def _finite_highs_bounds(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64).copy()
    values[np.isposinf(values)] = highspy.kHighsInf
    values[np.isneginf(values)] = -highspy.kHighsInf
    return values


def _build_highs(
    matrix: sp.spmatrix,
    row_lower: np.ndarray,
    row_upper: np.ndarray,
    objective: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    integrality: np.ndarray,
    *,
    threads: int,
    time_limit: float | None,
    mip_rel_gap: float,
    random_seed: int,
) -> highspy.Highs:
    matrix = sp.csr_matrix(matrix, dtype=np.float64)
    matrix.sort_indices()

    highs = highspy.Highs()
    n = matrix.shape[1]
    column_indices = np.arange(n, dtype=np.int32)

    highs.addVars(
        n,
        _finite_highs_bounds(lower),
        _finite_highs_bounds(upper),
    )
    highs.changeColsCost(n, column_indices, np.asarray(objective, dtype=np.float64))

    integer_indices = np.flatnonzero(np.asarray(integrality) != 0).astype(np.int32)
    if integer_indices.size:
        integer_types = np.full(
            integer_indices.size,
            highspy.HighsVarType.kInteger,
            dtype=np.uint8,
        )
        highs.changeColsIntegrality(
            integer_indices.size,
            integer_indices,
            integer_types,
        )

    highs.addRows(
        matrix.shape[0],
        _finite_highs_bounds(row_lower),
        _finite_highs_bounds(row_upper),
        matrix.nnz,
        matrix.indptr.astype(np.int32, copy=False),
        matrix.indices.astype(np.int32, copy=False),
        matrix.data,
    )

    highs.setOptionValue("output_flag", True)
    highs.setOptionValue("threads", threads)
    highs.setOptionValue("parallel", "on")
    highs.setOptionValue("random_seed", random_seed)
    highs.setOptionValue("mip_rel_gap", mip_rel_gap)
    if time_limit is not None:
        highs.setOptionValue("time_limit", time_limit)

    return highs


def _save_solution(
    solution: np.ndarray,
    objective_value: float,
    output_dir: Path,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "best.sol"
    with path.open("w") as handle:
        handle.write(f"# Objective value = {objective_value:.16g}\n")
        for index, value in enumerate(solution):
            handle.write(f"x[{index}] {value:.16g}\n")
    return path


def _summary(solution: np.ndarray, grid: int, temporary_roboport: bool) -> None:
    dd = grid * grid
    solar = int(np.rint(solution[:dd]).sum())
    accumulators = int(np.rint(solution[dd : 2 * dd]).sum())
    substations = int(np.rint(solution[2 * dd : 3 * dd]).sum())
    roboports = int(np.rint(solution[3 * dd : 4 * dd]).sum())
    medium_poles = int(np.rint(solution[4 * dd : 5 * dd]).sum())

    replacement_accumulators = 4 * roboports if temporary_roboport else 0
    effective_accumulators = accumulators + replacement_accumulators

    solar_power = (
        solar
        * parameters.SOLAR_PANEL_POWER
        * parameters.ETA_S
    )
    accumulator_power = (
        effective_accumulators
        * parameters.ACCUMULATOR_CHARGE
        / parameters.DAY_DURATION
        / parameters.C_ON
    )
    sustained_power = min(solar_power, accumulator_power)

    print()
    print(f"Sustained power: {sustained_power / 1000:.6f} MW")
    print(f"Solar panels: {solar}")
    print(
        f"Accumulators: {accumulators}"
        + (
            f" + {replacement_accumulators} after roboport removal"
            if replacement_accumulators
            else ""
        )
    )
    print(f"Substations: {substations}")
    print(f"Medium poles: {medium_poles}")
    print(f"Roboports during construction: {roboports}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Solve periodic Factorio solar-array MILPs with HiGHS."
    )
    parser.add_argument(
        "--planet",
        choices=sorted(parameters.PLANETS),
        default="nauvis",
    )
    parser.add_argument(
        "--roboport",
        choices=("temporary", "permanent"),
        default="temporary",
        help=(
            "temporary counts the 4x4 roboport footprint as four accumulators "
            "after construction"
        ),
    )
    parser.add_argument("--grid", type=int, default=50)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--time-limit", type=float)
    parser.add_argument("--mip-rel-gap", type=float, default=0.0)
    parser.add_argument("--random-seed", type=int, default=0)
    parser.add_argument("--max-substations", type=int, default=7)
    parser.add_argument("--max-medium-poles", type=int, default=5)
    parser.add_argument(
        "--network",
        type=Path,
        help=(
            "Fix the electrical network from a Stage-A/network .sol file. "
            "Omit for the full simultaneous A+B search."
        ),
    )
    parser.add_argument(
        "--min-power",
        type=float,
        default=0.0,
        help="Require at least this much sustained power, in kW.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Result directory. Defaults to results/<run-name>_<timestamp>.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()

    if args.threads < 1:
        raise SystemExit("--threads must be at least 1")
    if args.min_power < 0:
        raise SystemExit("--min-power must be nonnegative")
    if args.mip_rel_gap < 0:
        raise SystemExit("--mip-rel-gap must be nonnegative")

    planet = parameters.configure_planet(args.planet)
    parameters.GRID_SIZE = args.grid
    temporary = args.roboport == "temporary"
    roboport_substitution_factor = 4 if temporary else 0

    print(
        f"Planet: {planet.name}; peak solar={planet.solar_panel_power:g} kW; "
        f"day={planet.day_duration:g}s"
    )
    print(
        f"Power balance: {planet.sustained_solar_panel_power:g} kW/panel, "
        f"{planet.accumulator_supported_power:g} kW/accumulator, "
        f"ratio={planet.accumulator_per_solar:.6f} accumulators/panel"
    )
    print(
        f"Roboport mode: {args.roboport}; "
        f"threads={args.threads}; parallel=on"
    )

    dd = args.grid * args.grid

    if args.network is None:
        model = objectives.construct_restrictive_solver(
            args.grid,
            min_substations=None,
            max_substations=args.max_substations,
            min_medium_poles=None,
            max_medium_poles=args.max_medium_poles,
            roboport_substitution_factor=roboport_substitution_factor,
        )
        run_kind = "full"
    else:
        network = _load_indexed_solution(args.network, 2 * dd)
        model = objectives.construct_matrix_coverage_fixed_network(
            args.grid,
            network[:dd],
            network[dd:],
            min_power=args.min_power,
            roboport_substitution_factor=roboport_substitution_factor,
        )
        run_kind = "fixed"

    (
        constraint_matrix,
        constraint_lower_bounds,
        constraint_upper_bounds,
        variable_count,
        objective,
        variable_lower_bounds,
        variable_upper_bounds,
        integrality,
    ) = model

    if args.network is None and args.min_power:
        variable_lower_bounds = np.asarray(variable_lower_bounds).copy()
        variable_lower_bounds[-1] = args.min_power

    print(
        f"Model: {constraint_matrix.shape[0]} rows, "
        f"{variable_count} columns"
    )

    highs = _build_highs(
        constraint_matrix,
        constraint_lower_bounds,
        constraint_upper_bounds,
        objective,
        variable_lower_bounds,
        variable_upper_bounds,
        integrality,
        threads=args.threads,
        time_limit=args.time_limit,
        mip_rel_gap=args.mip_rel_gap,
        random_seed=args.random_seed,
    )

    highs.run()

    info = highs.getInfo()
    status = highs.getModelStatus()
    status_text = highs.modelStatusToString(status)
    primal_status = highs.solutionStatusToString(info.primal_solution_status)

    print()
    print(f"Model status: {status_text}")
    print(f"Primal solution status: {primal_status}")
    print(f"Runtime: {highs.getRunTime():.3f}s")

    if primal_status.lower() != "feasible":
        return 2

    solution = np.asarray(list(highs.getSolution().col_value), dtype=float)
    integer_mask = np.asarray(integrality) != 0
    solution[integer_mask] = np.rint(solution[integer_mask])

    _summary(solution, args.grid, temporary)

    if args.output is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = (
            Path("results")
            / f"{planet.name}_{args.roboport}_{run_kind}_{timestamp}"
        )

    solution_path = _save_solution(
        solution,
        highs.getObjectiveValue(),
        args.output,
    )
    print(f"Solution: {solution_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
