from dataclasses import dataclass


# Solar array geometry fundamentals
GRID_SIZE = 50
SOLAR_SIZE = 3
ACCUMULATOR_SIZE = 2
SUBSTATION_SIZE = 2
ROBOPORT_SIZE = 4

# Normal-quality vanilla entities.
BASE_SOLAR_PANEL_POWER = 60.0  # kW at 100% solar intensity
ACCUMULATOR_CHARGE = 5e3  # kJ == 5 MJ
ETA_S = 0.70
C_ON = 0.24


@dataclass(frozen=True, slots=True)
class PlanetPower:
    """Power-generation parameters needed by the MILP.

    solar_multiplier is Factorio's surface solar-power multiplier relative
    to Nauvis. eta_s and c_on describe the vanilla daylight curve; they
    are the same for Nauvis and Vulcanus, while the day duration differs.
    """

    name: str
    day_duration: float
    solar_multiplier: float
    eta_s: float = ETA_S
    c_on: float = C_ON
    accumulator_charge: float = ACCUMULATOR_CHARGE
    base_solar_panel_power: float = BASE_SOLAR_PANEL_POWER

    @property
    def solar_panel_power(self) -> float:
        return self.base_solar_panel_power * self.solar_multiplier

    @property
    def sustained_solar_panel_power(self) -> float:
        return self.solar_panel_power * self.eta_s

    @property
    def accumulator_supported_power(self) -> float:
        return self.accumulator_charge / self.day_duration / self.c_on

    @property
    def accumulator_per_solar(self) -> float:
        return self.sustained_solar_panel_power / self.accumulator_supported_power


PLANETS: dict[str, PlanetPower] = {
    "nauvis": PlanetPower(
        name="nauvis",
        day_duration=7 * 60,
        solar_multiplier=1.0,
    ),
    "vulcanus": PlanetPower(
        name="vulcanus",
        day_duration=90,
        solar_multiplier=4.0,
    ),
}


# Compatibility globals used by the original model constructors. They are
# configured through configure_planet() rather than duplicated throughout
# the solver.
DAY_DURATION: float
SOLAR_PANEL_POWER: float
CURRENT_PLANET: PlanetPower


def configure_planet(name: str) -> PlanetPower:
    """Select a power profile for the existing model constructors."""

    try:
        planet = PLANETS[name.lower()]
    except KeyError as exc:
        choices = ", ".join(sorted(PLANETS))
        raise ValueError(f"Unknown planet {name!r}; choose one of: {choices}") from exc

    global DAY_DURATION, SOLAR_PANEL_POWER, CURRENT_PLANET
    CURRENT_PLANET = planet
    DAY_DURATION = planet.day_duration
    SOLAR_PANEL_POWER = planet.solar_panel_power
    return planet


configure_planet("nauvis")


# Optimization settings / retained for compatibility with the original code.
ROBOPORT_NUMBER = 2
SUBSTATION_MAX = 36
