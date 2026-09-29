from __future__ import annotations

import argparse
from datetime import datetime
import math
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np

from . import objectives, parameters
from .support import utilities


def _sat_imports():
    try:
        from pysat.card import CardEnc, EncType
        from pysat.formula import CNF, IDPool
    except ImportError as exc:
        raise SystemExit(
            "SAT support is optional. Install it with: uv sync --extra sat"
        ) from exc
    return CardEnc, EncType, CNF, IDPool


def _anchor(coords: tuple[int, int], grid: int) -> int:
    row, column = coords
    return row * grid + column


def _fixed_geometry(
    grid: int,
    network: utilities.NetworkLayout,
) -> tuple[set[int], tuple[int, ...], np.ndarray, np.ndarray]:
    dd = grid * grid
    substations = np.zeros(dd, dtype=np.int8)
    medium_poles = np.zeros(dd, dtype=np.int8)

    occupied: set[int] = set()
    for coords in network.substations:
        root = _anchor(coords, grid)
        substations[root] = 1
        occupied.update(utilities.block_indices(root, 0, 1, grid))

    for coords in network.medium_poles:
        root = _anchor(coords, grid)
        medium_poles[root] = 1
        occupied.add(root)

    roboport_roots = objectives.roboport_roots_for_grid(grid)
    for root in roboport_roots:
        footprint = set(utilities.block_indices(root, 0, 3, grid))
        collision = occupied.intersection(footprint)
        if collision:
            raise ValueError(
                "Fixed network overlaps the fixed roboport at tiles "
                f"{sorted(collision)[:8]}"
            )
        occupied.update(footprint)

    return occupied, roboport_roots, substations, medium_poles


def _powered_tiles(
    grid: int,
    network: utilities.NetworkLayout,
) -> set[int]:
    powered: set[int] = set()
    for coords in network.substations:
        powered.update(
            utilities.block_indices(_anchor(coords, grid), 8, 9, grid)
        )
    for coords in network.medium_poles:
        powered.update(
            utilities.block_indices(_anchor(coords, grid), 3, 3, grid)
        )
    return powered


def _candidate_anchors(
    grid: int,
    structure_size: int,
    occupied: set[int],
    powered: set[int],
) -> tuple[list[int], dict[int, tuple[int, ...]]]:
    candidates: list[int] = []
    footprints: dict[int, tuple[int, ...]] = {}

    for anchor in range(grid * grid):
        footprint = tuple(
            utilities.block_indices(anchor, 0, structure_size - 1, grid)
        )
        footprint_set = set(footprint)
        if occupied.intersection(footprint_set):
            continue
        # Same semantics as the existing MILP: an entity is powered when its
        # footprint intersects at least one pole/substation supply tile.
        if not powered.intersection(footprint_set):
            continue
        candidates.append(anchor)
        footprints[anchor] = footprint

    return candidates, footprints


def _append_cardinality(formula, encoded) -> None:
    formula.extend(encoded.clauses)


def _build_cnf(
    *,
    grid: int,
    network: utilities.NetworkLayout,
    solar_count: int,
    accumulator_count: int,
):
    CardEnc, EncType, CNF, IDPool = _sat_imports()

    occupied, roboport_roots, substations, medium_poles = _fixed_geometry(
        grid, network
    )
    powered = _powered_tiles(grid, network)
    free_tiles = sorted(set(range(grid * grid)) - occupied)

    solar_anchors, solar_footprints = _candidate_anchors(
        grid, 3, occupied, powered
    )
    accumulator_anchors, accumulator_footprints = _candidate_anchors(
        grid, 2, occupied, powered
    )

    implied_holes = (
        len(free_tiles) - 9 * solar_count - 4 * accumulator_count
    )
    if implied_holes < 0:
        raise ValueError(
            "Requested counts exceed available tile area: "
            f"{solar_count} solar + {accumulator_count} accumulators need "
            f"{-implied_holes} more tiles."
        )

    if solar_count > len(solar_anchors):
        raise ValueError(
            f"Only {len(solar_anchors)} valid solar anchors exist, "
            f"but {solar_count} were requested."
        )
    if accumulator_count > len(accumulator_anchors):
        raise ValueError(
            f"Only {len(accumulator_anchors)} valid accumulator anchors exist, "
            f"but {accumulator_count} were requested."
        )

    pool = IDPool()
    solar_vars = {
        anchor: pool.id(("solar", anchor)) for anchor in solar_anchors
    }
    accumulator_vars = {
        anchor: pool.id(("accumulator", anchor))
        for anchor in accumulator_anchors
    }
    hole_vars = {tile: pool.id(("hole", tile)) for tile in free_tiles}

    covering: dict[int, list[int]] = {tile: [] for tile in free_tiles}
    for anchor, footprint in solar_footprints.items():
        literal = solar_vars[anchor]
        for tile in footprint:
            covering[tile].append(literal)
    for anchor, footprint in accumulator_footprints.items():
        literal = accumulator_vars[anchor]
        for tile in footprint:
            covering[tile].append(literal)

    formula = CNF()

    # Every non-infrastructure tile is exactly one panel tile, accumulator
    # tile, or an explicit hole. The per-tile arity is small, so pairwise
    # AtMost1 is compact and propagates strongly.
    for tile in free_tiles:
        literals = covering[tile] + [hole_vars[tile]]
        formula.append(literals)
        for i, left in enumerate(literals):
            for right in literals[i + 1 :]:
                formula.append([-left, -right])

    # Cardinality networks are larger than sequential counters for some bounds,
    # but are balanced, highly propagating, and do not depend on variable order.
    _append_cardinality(
        formula,
        CardEnc.equals(
            lits=list(solar_vars.values()),
            bound=solar_count,
            vpool=pool,
            encoding=EncType.cardnetwrk,
        ),
    )
    _append_cardinality(
        formula,
        CardEnc.equals(
            lits=list(accumulator_vars.values()),
            bound=accumulator_count,
            vpool=pool,
            encoding=EncType.cardnetwrk,
        ),
    )

    metadata = {
        "occupied_tiles": len(occupied),
        "free_tiles": len(free_tiles),
        "implied_holes": implied_holes,
        "solar_candidates": len(solar_anchors),
        "accumulator_candidates": len(accumulator_anchors),
        "solar_vars": solar_vars,
        "accumulator_vars": accumulator_vars,
        "hole_vars": hole_vars,
        "roboport_roots": roboport_roots,
        "substations": substations,
        "medium_poles": medium_poles,
        "variables": pool.top,
    }
    return formula, metadata


def _resolve_solver(requested: str) -> str:
    if requested != "auto":
        resolved = shutil.which(requested)
        if resolved is None and Path(requested).is_file():
            resolved = requested
        if resolved is None:
            raise SystemExit(f"SAT solver not found: {requested}")
        return resolved

    for candidate in ("kissat", "cadical"):
        resolved = shutil.which(candidate)
        if resolved is not None:
            return resolved

    raise SystemExit(
        "No external SAT solver found. Install kissat or cadical, or pass "
        "--solver /path/to/solver."
    )


def _run_solver(solver: str, cnf_path: Path) -> tuple[str, set[int]]:
    print(f"SAT solver: {solver}")
    command = [solver, str(cnf_path)]
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    status = "UNKNOWN"
    positive_model: set[int] = set()

    try:
        assert process.stdout is not None
        for line in process.stdout:
            stripped = line.strip()
            if stripped.startswith("v "):
                for token in stripped[2:].split():
                    literal = int(token)
                    if literal > 0:
                        positive_model.add(literal)
                continue
            if stripped.startswith("s "):
                if "UNSATISFIABLE" in stripped:
                    status = "UNSAT"
                elif "SATISFIABLE" in stripped:
                    status = "SAT"
            print(line, end="", flush=True)
        return_code = process.wait()
    except KeyboardInterrupt:
        print("\nInterrupting SAT solver...", file=sys.stderr)
        process.terminate()
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        raise

    if return_code == 10:
        status = "SAT"
    elif return_code == 20:
        status = "UNSAT"
    elif return_code != 0 and status == "UNKNOWN":
        raise RuntimeError(
            f"SAT solver exited with status {return_code} without a result."
        )

    return status, positive_model


def _write_solution(
    *,
    path: Path,
    grid: int,
    target_power: float,
    positive_model: set[int],
    metadata,
) -> np.ndarray:
    dd = grid * grid
    solution = np.zeros(5 * dd + 1, dtype=float)

    for anchor, variable in metadata["solar_vars"].items():
        if variable in positive_model:
            solution[anchor] = 1
    for anchor, variable in metadata["accumulator_vars"].items():
        if variable in positive_model:
            solution[dd + anchor] = 1

    solution[2 * dd : 3 * dd] = metadata["substations"]

    for root in metadata["roboport_roots"]:
        solution[3 * dd + root] = 1

    solution[4 * dd : 5 * dd] = metadata["medium_poles"]
    solution[-1] = target_power

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        handle.write(f"# SAT target power = {target_power:.16g}\n")
        for index, value in enumerate(solution):
            handle.write(f"x[{index}] {value:.16g}\n")

    return solution


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Solve the fixed-network Factorio packing stage as SAT/exact cover."
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
        "--solver",
        default="auto",
        help="auto, kissat, cadical, or an explicit solver path",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Result directory; defaults to results/<run-name>_<timestamp>.",
    )
    parser.add_argument(
        "--keep-cnf",
        action="store_true",
        help="Keep the generated DIMACS file after the solve.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()

    if not 20 <= args.grid <= 50:
        raise SystemExit("--grid currently supports 20..50")
    if (args.solar is None) != (args.accumulators is None):
        raise SystemExit("Specify both --solar and --accumulators, or neither.")
    if args.target_power is None and args.solar is None:
        raise SystemExit(
            "Specify --target-power, or exact --solar and --accumulators counts."
        )

    planet = parameters.configure_planet(args.planet)
    parameters.GRID_SIZE = args.grid

    roboport_count = len(objectives.roboport_roots_for_grid(args.grid))
    temporary = args.roboport == "temporary"
    replacement_accumulators = 4 * roboport_count if temporary else 0

    if args.solar is None:
        assert args.target_power is not None
        solar_count = math.ceil(
            args.target_power / planet.sustained_solar_panel_power - 1e-12
        )
        effective_accumulators = math.ceil(
            args.target_power / planet.accumulator_supported_power - 1e-12
        )
        accumulator_count = max(
            0, effective_accumulators - replacement_accumulators
        )
    else:
        solar_count = args.solar
        accumulator_count = args.accumulators

    if solar_count < 0 or accumulator_count < 0:
        raise SystemExit("Building counts must be nonnegative.")

    achieved_power = min(
        solar_count * planet.sustained_solar_panel_power,
        (accumulator_count + replacement_accumulators)
        * planet.accumulator_supported_power,
    )
    if args.target_power is not None and achieved_power + 1e-7 < args.target_power:
        raise SystemExit(
            "Requested exact building counts do not reach --target-power."
        )
    target_power = args.target_power or achieved_power

    network = utilities.read_solution_layout(args.network)

    print(
        f"SAT packing: grid={args.grid}, solar={solar_count}, "
        f"accumulators={accumulator_count}, target={target_power / 1000:.6f} MW"
    )
    formula, metadata = _build_cnf(
        grid=args.grid,
        network=network,
        solar_count=solar_count,
        accumulator_count=accumulator_count,
    )

    print(
        f"Geometry: {metadata['occupied_tiles']} fixed tiles, "
        f"{metadata['free_tiles']} free, "
        f"{metadata['implied_holes']} implied holes"
    )
    print(
        f"Candidates: {metadata['solar_candidates']} solar, "
        f"{metadata['accumulator_candidates']} accumulators"
    )
    print(
        f"CNF: {metadata['variables']} variables, "
        f"{len(formula.clauses)} clauses"
    )

    if args.output is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = (
            Path("results")
            / f"{planet.name}_{args.roboport}_sat_{args.grid}_{timestamp}"
        )
    args.output.mkdir(parents=True, exist_ok=True)
    cnf_path = args.output / "packing.cnf"
    formula.to_file(cnf_path)

    solver = _resolve_solver(args.solver)
    try:
        status, positive_model = _run_solver(solver, cnf_path)
    finally:
        if not args.keep_cnf:
            cnf_path.unlink(missing_ok=True)

    print(f"SAT status: {status}")
    if status == "UNSAT":
        return 2
    if status != "SAT":
        return 3

    solution_path = args.output / "best.sol"
    solution = _write_solution(
        path=solution_path,
        grid=args.grid,
        target_power=target_power,
        positive_model=positive_model,
        metadata=metadata,
    )

    dd = args.grid * args.grid
    found_solar = int(solution[:dd].sum())
    found_accumulators = int(solution[dd : 2 * dd].sum())
    found_holes = sum(
        1
        for variable in metadata["hole_vars"].values()
        if variable in positive_model
    )

    print(
        f"Found: {found_solar} solar, {found_accumulators} accumulators, "
        f"{found_holes} holes"
    )
    print(f"Solution: {solution_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
