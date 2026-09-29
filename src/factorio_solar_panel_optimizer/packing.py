from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path

import numpy as np

from . import objectives, parameters
from .support import utilities


@dataclass(frozen=True, slots=True)
class PackingGeometry:
    grid: int
    occupied: frozenset[int]
    free_tiles: tuple[int, ...]
    roboport_roots: tuple[int, ...]
    substation_roots: tuple[int, ...]
    medium_pole_roots: tuple[int, ...]
    solar_anchors: tuple[int, ...]
    accumulator_anchors: tuple[int, ...]
    solar_footprints: dict[int, tuple[int, ...]]
    accumulator_footprints: dict[int, tuple[int, ...]]
    covering_by_tile: dict[int, tuple[tuple[str, int], ...]]

    def implied_holes(self, solar_count: int, accumulator_count: int) -> int:
        return (
            len(self.free_tiles)
            - 9 * solar_count
            - 4 * accumulator_count
        )


@dataclass(frozen=True, slots=True)
class PackingTarget:
    solar_count: int
    accumulator_count: int
    target_power: float
    achieved_power: float
    replacement_accumulators: int


def _anchor(coords: tuple[int, int], grid: int) -> int:
    row, column = coords
    return row * grid + column


def build_fixed_geometry(
    grid: int,
    network: utilities.NetworkLayout,
) -> PackingGeometry:
    dd = grid * grid
    substation_roots = tuple(_anchor(coords, grid) for coords in network.substations)
    medium_pole_roots = tuple(
        _anchor(coords, grid) for coords in network.medium_poles
    )

    occupied: set[int] = set()
    for root in substation_roots:
        occupied.update(utilities.block_indices(root, 0, 1, grid))
    occupied.update(medium_pole_roots)

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

    powered: set[int] = set()
    for root in substation_roots:
        powered.update(utilities.block_indices(root, 8, 9, grid))
    for root in medium_pole_roots:
        powered.update(utilities.block_indices(root, 3, 3, grid))

    def candidates(structure_size: int):
        anchors: list[int] = []
        footprints: dict[int, tuple[int, ...]] = {}
        for anchor in range(dd):
            footprint = tuple(
                utilities.block_indices(
                    anchor, 0, structure_size - 1, grid
                )
            )
            footprint_set = set(footprint)
            if occupied.intersection(footprint_set):
                continue
            # Match the existing MILP semantics: at least one entity tile
            # intersects electrical supply coverage.
            if not powered.intersection(footprint_set):
                continue
            anchors.append(anchor)
            footprints[anchor] = footprint
        return tuple(anchors), footprints

    solar_anchors, solar_footprints = candidates(3)
    accumulator_anchors, accumulator_footprints = candidates(2)

    free_tiles = tuple(sorted(set(range(dd)) - occupied))
    covering: dict[int, list[tuple[str, int]]] = {
        tile: [] for tile in free_tiles
    }
    for anchor, footprint in solar_footprints.items():
        for tile in footprint:
            covering[tile].append(("solar", anchor))
    for anchor, footprint in accumulator_footprints.items():
        for tile in footprint:
            covering[tile].append(("accumulator", anchor))

    return PackingGeometry(
        grid=grid,
        occupied=frozenset(occupied),
        free_tiles=free_tiles,
        roboport_roots=roboport_roots,
        substation_roots=substation_roots,
        medium_pole_roots=medium_pole_roots,
        solar_anchors=solar_anchors,
        accumulator_anchors=accumulator_anchors,
        solar_footprints=solar_footprints,
        accumulator_footprints=accumulator_footprints,
        covering_by_tile={
            tile: tuple(entries) for tile, entries in covering.items()
        },
    )


def target_from_power(
    *,
    grid: int,
    planet: parameters.PlanetPower,
    temporary_roboport: bool,
    target_power: float | None,
    solar_count: int | None,
    accumulator_count: int | None,
) -> PackingTarget:
    if (solar_count is None) != (accumulator_count is None):
        raise ValueError(
            "Specify both solar_count and accumulator_count, or neither."
        )
    if target_power is None and solar_count is None:
        raise ValueError(
            "Specify target_power, or exact solar and accumulator counts."
        )

    roboport_count = len(objectives.roboport_roots_for_grid(grid))
    replacement_accumulators = (
        4 * roboport_count if temporary_roboport else 0
    )

    if solar_count is None:
        assert target_power is not None
        solar_count = math.ceil(
            target_power / planet.sustained_solar_panel_power - 1e-12
        )
        effective_accumulators = math.ceil(
            target_power / planet.accumulator_supported_power - 1e-12
        )
        accumulator_count = max(
            0, effective_accumulators - replacement_accumulators
        )

    assert accumulator_count is not None
    if solar_count < 0 or accumulator_count < 0:
        raise ValueError("Building counts must be nonnegative.")

    achieved_power = min(
        solar_count * planet.sustained_solar_panel_power,
        (accumulator_count + replacement_accumulators)
        * planet.accumulator_supported_power,
    )
    if target_power is not None and achieved_power + 1e-7 < target_power:
        raise ValueError(
            "Requested exact building counts do not reach target_power."
        )

    return PackingTarget(
        solar_count=solar_count,
        accumulator_count=accumulator_count,
        target_power=target_power or achieved_power,
        achieved_power=achieved_power,
        replacement_accumulators=replacement_accumulators,
    )


def validate_target(
    geometry: PackingGeometry,
    target: PackingTarget,
) -> int:
    holes = geometry.implied_holes(
        target.solar_count,
        target.accumulator_count,
    )
    if holes < 0:
        raise ValueError(
            "Requested counts exceed available tile area by "
            f"{-holes} tiles."
        )
    if target.solar_count > len(geometry.solar_anchors):
        raise ValueError(
            f"Only {len(geometry.solar_anchors)} valid solar anchors exist, "
            f"but {target.solar_count} were requested."
        )
    if target.accumulator_count > len(geometry.accumulator_anchors):
        raise ValueError(
            "Only "
            f"{len(geometry.accumulator_anchors)} valid accumulator anchors "
            f"exist, but {target.accumulator_count} were requested."
        )
    return holes


def write_solution(
    *,
    path: Path,
    geometry: PackingGeometry,
    target_power: float,
    selected_solar: set[int],
    selected_accumulators: set[int],
    source: str,
) -> np.ndarray:
    grid = geometry.grid
    dd = grid * grid
    solution = np.zeros(5 * dd + 1, dtype=float)

    for anchor in selected_solar:
        solution[anchor] = 1
    for anchor in selected_accumulators:
        solution[dd + anchor] = 1
    for root in geometry.substation_roots:
        solution[2 * dd + root] = 1
    for root in geometry.roboport_roots:
        solution[3 * dd + root] = 1
    for root in geometry.medium_pole_roots:
        solution[4 * dd + root] = 1
    solution[-1] = target_power

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        handle.write(f"# {source} target power = {target_power:.16g}\n")
        for index, value in enumerate(solution):
            handle.write(f"x[{index}] {value:.16g}\n")

    return solution
