# Factorio Solar Panel Optimization

Hi!

As I mentioned in the video, this code is not the cleanest you will ever see,
but you can probably figure out how it works. Just in case, I'm leaving this as
a quick guideline on how to run your first optimization.

## Useful files

- [`parameters.py`](parameters.py) holds most of the main problem constants. You
  probably do not want to touch this too much.
- [`plot_solution.py`](plot_solution.py) can take either a path string or a
  blueprint string and plot the panel just as you see in the video.
- [`print_blueprint.py`](print_blueprint.py) just prints out a blueprint from a
  path. I left that just so you can see how the blueprint parser works.
- [`solar_vs_substation.py`](solar_vs_substation.py) plots the optimality-proof
  charts, showing either the required electric-network efficacy or the number
  of empty tiles.

## Main solvers

### [`solve_system_highs_fixed_network.py`](solve_system_highs_fixed_network.py)

This is what you want to start with. It takes in a fixed electric network, and
I left one pretty good one for you to use. The optimization is limited to just
100 seconds, and you should see results right away, which you can plot with
`plot_solution.py`. This is what I call a Stage B solver: it takes in a network
and solves the packing around it.

### [`solve_system_highs_stage_A.py`](solve_system_highs_stage_A.py)

This is a Stage A solver, so you can generate your own networks. It is currently
set to run for only 300 seconds, just so you can see how it works. The score is
the number of tiles used by the network divided by 4, so it can be interpreted
as the number of substations. If you run it as is, you should be able to get an
8 to 8.5 result, which you can then plug into the previous fixed-network solver
to get an easy 8232 kW solution.

### [`solve_system_highs.py`](solve_system_highs.py)

This is the real-deal Stage A+B solver with the full problem, including
connectivity, so it takes forever to run but will give you solutions without an
electric-network input. When I say forever, I mean forever. On a single core,
using HiGHS, you might be waiting for days until you see an incumbent, or weeks
for more progress. You probably want to find a parallel-capable solver to run
this well, but if you have time, you can theoretically do it with HiGHS.

### [`staged_solver_highs.py`](staged_solver_highs.py)

This is the staged solver I used to find the 8316 permanent-roboport setup in
the video. It is currently made for that specific purpose, so it takes advantage
of fixed building counts and also uses seeds I generated with the other two
solvers. You can make modifications to suit your needs and solve other similar
problems, but its not as easy as changing variables, this is not made fully genetic, but specifically intended at solving the recalcitrant 8316 problem.

## Model constructors

- [`objectives.py`](objectives.py) contains most of my constructors for Stage B
  or Stage A+B. Do not touch this unless you are familiar with everything else.
- [`coverage_objectives.py`](coverage_objectives.py) contains constructors for
  Stage A (connectivity and tileability). Again, do not touch this until you are
  familiar with the formulation.

## License

This project is available under the [MIT License](LICENSE).


## Parameterized planet solver

This fork adds a direct `highspy` CLI so the same MILP can be solved with
different planetary power profiles without editing model code.

Install/update the environment:

```bash
uv sync
```

For the Vulcanus temporary-roboport problem (normal solar panels and
accumulators):

```bash
uv run factorio-solar-solve \
  --planet vulcanus \
  --roboport temporary \
  --threads 8
```

That runs the full simultaneous Stage A+B search. For a much faster packing
search around the repository's sample electrical network:

```bash
uv run factorio-solar-solve \
  --planet vulcanus \
  --roboport temporary \
  --threads 8 \
  --network support/sample_network.txt
```

Use `--time-limit SECONDS` to cap a run and `--min-power KW` to turn a
candidate power level into a feasibility target. Results are written under
`results/`.

The built-in normal-quality presets are:

| Planet | Day | Solar multiplier | Sustained panel power | Accumulators / panel |
| --- | ---: | ---: | ---: | ---: |
| Nauvis | 420 s | 100% | 42 kW | 0.84672 |
| Vulcanus | 90 s | 400% | 168 kW | 0.72576 |

In temporary-roboport mode the central 4x4 roboport must still fit and be
powered during construction, but the power balance treats its eventual
footprint as four 2x2 accumulators.


### Interactive solution plots

Plotting is optional and uses Plotly rather than Matplotlib:

```bash
uv sync --extra plot

uv run --extra plot factorio-solar-plot \
  results/<run>/best.sol \
  --planet vulcanus \
  --roboport temporary
```

The viewer writes a self-contained HTML file beside the solution and opens it
in the browser. Add `--electric-coverage` to overlay pole/substation supply
areas, or `--no-show` when running headless.

The solver itself has no plotting dependency.


### Smaller periodic cells and target feasibility

A roboport's logistics area is 50x50, so a single centered roboport per
periodic cell remains connected for repeat widths up to 50 tiles. The new CLI
supports `--grid 20` through `--grid 50` and reports both total power and
power density so different cell sizes can be compared fairly.

For hard endgames, target feasibility is often better than another long
maximize run:

```bash
uv run factorio-solar-solve \
  --planet vulcanus \
  --roboport temporary \
  --threads 8 \
  --network support/sample_network.txt \
  --target-power 34722.222222
```

`--target-power` fixes the sustained-power variable and switches to a zero
objective. HiGHS can then stop at the first feasible packing or prove that
target impossible instead of spending time optimizing beyond it.
