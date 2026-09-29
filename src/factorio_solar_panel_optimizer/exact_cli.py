from __future__ import annotations

import argparse
from datetime import datetime
import multiprocessing as mp
from pathlib import Path
import queue
import sys
import time

from . import parameters
from .packing import (
    build_fixed_geometry,
    target_from_power,
    validate_target,
    write_solution,
)
from .support import utilities


def _solve_exact_child(geometry, target, timeout: float, result_queue) -> None:
    try:
        from exact import Exact
    except ImportError:
        result_queue.put(("ERROR", "Exact is not installed"))
        return

    solver = Exact([("verbosity", "1")])

    solar_names = {
        anchor: f"s_{anchor}" for anchor in geometry.solar_anchors
    }
    accumulator_names = {
        anchor: f"a_{anchor}" for anchor in geometry.accumulator_anchors
    }

    for name in solar_names.values():
        solver.addVariable(name)
    for name in accumulator_names.values():
        solver.addVariable(name)

    # Every tile may belong to at most one placed entity. The fixed building
    # counts determine the number of uncovered holes automatically.
    for tile in geometry.free_tiles:
        terms = []
        for kind, anchor in geometry.covering_by_tile[tile]:
            if kind == "solar":
                terms.append((1, solar_names[anchor]))
            else:
                terms.append((1, accumulator_names[anchor]))
        if terms:
            solver.addConstraint(terms, False, 0, True, 1)

    solver.addConstraint(
        [(1, name) for name in solar_names.values()],
        True,
        target.solar_count,
        True,
        target.solar_count,
    )
    solver.addConstraint(
        [(1, name) for name in accumulator_names.values()],
        True,
        target.accumulator_count,
        True,
        target.accumulator_count,
    )

    status = solver.runFull(optimize=False, timeout=timeout)

    if status == "SAT":
        solar_values = solver.getLastSolutionFor(list(solar_names.values()))
        accumulator_values = solver.getLastSolutionFor(
            list(accumulator_names.values())
        )
        selected_solar = {
            anchor
            for anchor, value in zip(solar_names, solar_values)
            if value
        }
        selected_accumulators = {
            anchor
            for anchor, value in zip(
                accumulator_names,
                accumulator_values,
            )
            if value
        }
        result_queue.put(
            (
                "SAT",
                selected_solar,
                selected_accumulators,
            )
        )
    else:
        result_queue.put((status,))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Solve fixed-network Factorio packing with Exact's native "
            "pseudo-Boolean engine."
        )
    )
    parser.add_argument("--network", type=Path, required=True)
    parser.add_argument(
        "--planet",
        choices=sorted(parameters.PLANETS),
        default="vulcanus",
    )
    parser.add_argument(
        "--roboport",
        choices=("temporary", "permanent"),
        default="temporary",
    )
    parser.add_argument("--grid", type=int, default=50)
    parser.add_argument("--target-power", type=float)
    parser.add_argument("--solar", type=int)
    parser.add_argument("--accumulators", type=int)
    parser.add_argument(
        "--time-limit",
        type=float,
        default=0,
        help="Seconds; 0 means no solver time limit.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Result directory; defaults under results/.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()

    if not 20 <= args.grid <= 50:
        raise SystemExit("--grid currently supports 20..50")
    if args.time_limit < 0:
        raise SystemExit("--time-limit must be nonnegative")

    planet = parameters.configure_planet(args.planet)
    parameters.GRID_SIZE = args.grid
    temporary = args.roboport == "temporary"

    network = utilities.read_solution_layout(args.network)
    geometry = build_fixed_geometry(args.grid, network)

    try:
        target = target_from_power(
            grid=args.grid,
            planet=planet,
            temporary_roboport=temporary,
            target_power=args.target_power,
            solar_count=args.solar,
            accumulator_count=args.accumulators,
        )
        holes = validate_target(geometry, target)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    print(
        f"Exact packing: grid={args.grid}, "
        f"solar={target.solar_count}, "
        f"accumulators={target.accumulator_count}, "
        f"target={target.target_power / 1000:.6f} MW"
    )
    print(
        f"Geometry: {len(geometry.occupied)} fixed tiles, "
        f"{len(geometry.free_tiles)} free, {holes} implied holes"
    )
    print(
        f"Candidates: {len(geometry.solar_anchors)} solar, "
        f"{len(geometry.accumulator_anchors)} accumulators"
    )

    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue()
    process = ctx.Process(
        target=_solve_exact_child,
        args=(geometry, target, args.time_limit, result_queue),
        daemon=False,
    )

    start = time.monotonic()
    process.start()

    try:
        while process.is_alive():
            process.join(0.25)
    except KeyboardInterrupt:
        print("\nInterrupting Exact...", file=sys.stderr)
        process.terminate()
        process.join(2)
        if process.is_alive():
            process.kill()
            process.join()
        return 130

    runtime = time.monotonic() - start

    try:
        result = result_queue.get_nowait()
    except queue.Empty:
        if process.exitcode:
            raise SystemExit(
                f"Exact worker exited with status {process.exitcode}"
            )
        raise SystemExit("Exact worker exited without returning a result.")

    status = result[0]
    print(f"Exact status: {status}")
    print(f"Runtime: {runtime:.3f}s")

    if status == "ERROR":
        raise SystemExit(result[1])
    if status == "UNSAT":
        return 2
    if status == "TIMEOUT":
        return 3
    if status != "SAT":
        raise SystemExit(f"Unexpected Exact status: {status}")

    selected_solar = result[1]
    selected_accumulators = result[2]

    if args.output is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = (
            Path("results")
            / f"{planet.name}_{args.roboport}_exact_{args.grid}_{timestamp}"
        )

    solution_path = args.output / "best.sol"
    write_solution(
        path=solution_path,
        geometry=geometry,
        target_power=target.target_power,
        selected_solar=selected_solar,
        selected_accumulators=selected_accumulators,
        source="Exact",
    )

    print(
        f"Found: {len(selected_solar)} solar, "
        f"{len(selected_accumulators)} accumulators, {holes} holes"
    )
    print(f"Solution: {solution_path.resolve()}")
    return 0


if __name__ == "__main__":
    mp.freeze_support()
    raise SystemExit(main())
