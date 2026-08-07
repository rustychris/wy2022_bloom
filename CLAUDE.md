# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Research code for modeling the 2022 phytoplankton bloom in South San Francisco Bay. Combines
Delft3D Flexible Mesh (DFM) hydrodynamics, Delft3D-WAQ (DWAQ) Lagrangian tracer transport,
remote-sensing chlorophyll/temperature observations, and simplified nutrient-phytoplankton-zooplankton-virus
models fit to those tracers. There is no package structure, build system, or test suite — this is
a working notebook/script collection for one research project, not a library.

## Environment

- No `environment.yml` / `requirements.txt` / `pyproject.toml`. Dependencies are managed via an ambient
  conda environment (assume it's already active; do not try to create one from scratch).
- Target interpreter is Python 3.12 (per `.idea/misc.xml`, SDK `general312`).
- **`stompy`** (Rusty Holleman's SF Bay/Delta modeling toolkit — grids, DFM/DWAQ I/O, spatial fields,
  station data loaders) is imported everywhere (`from stompy import ...`, `stompy.model.delft.dflow_model`,
  `stompy.model.delft.waq_scenario`, `stompy.grid.unstructured_grid`, `stompy.io.local.{noaa_coops,usgs_nwis,ndbc}`,
  `stompy.plot.*`) but is **not** a pip dependency — it's a local checkout expected on `PYTHONPATH`
  (see the commented-out `sys.path.append(".../stompy")` in the opt script). If imports of `stompy` fail,
  the fix is locating/adding that checkout, not adding it to a requirements file.
- `golumn.py` hardcodes a GOTM `LD_LIBRARY_PATH` (conda prefix + an Intel oneAPI compiler path) at import
  time — only relevant if actually running the GOTM 1-D water column model.
- There is no test suite and no lint/build command. "Running" work here means executing a notebook or a
  jupytext-exported `.py` script cell-by-cell (see below), not `pytest`.

## Notebooks vs. `.py` scripts

Most of the actual analysis lives in `.ipynb` notebooks (many versioned/superseded — `v00`, `v01`,
`_newKd`, `-Copy1`, etc.; check modification dates and read the last few cells before assuming a given
version is current). Some notebooks are paired with `.py` exports via jupytext — these retain
`# In[n]:` cell markers and are meant to be read/run as a sequence of cells (often with plotting/exploration
cells that depend on state from earlier cells and aren't meant to run start-to-finish as a plain script).
`bloom_lagrangian_zoop_etal_opt_20260729.py` is one such export. When editing these `.py` files, preserve
the cell-marker structure rather than "cleaning it up" into a linear script.

`bloom_lagrangian_zoop_etal_stan_20260806.py` is a new, currently-empty file — the planned Stan/Bayesian
reimplementation of the `scipy.optimize`-based parameter tuning in the `_opt_` script (no stan/cmdstanpy
usage exists anywhere yet).

Large model outputs (netCDF, movies, frame dumps, DFM run directories, RS scene caches) are git-ignored
(see `.gitignore`: `data_*`, `cache`, `frames-*`, `bloom_tracers_v*`, `rotodfm_v*`, `rs_chl/`, `*.mp4`,
`dfm_spinup`, etc.) — don't expect them to be present in a fresh checkout, and don't assume a missing
data directory is a bug.

## Core module map

The modeling pipeline is: DFM hydro run -> DWAQ Lagrangian tracer run -> per-tracer extraction/analysis ->
simplified biogeochemical ODE fit to observations. Shared logic lives in root-level `.py` modules imported
across notebooks:

- **`tracer_group.py`** — `TracerGroup`, the central abstraction wrapping one DFM/DWAQ tracer-release run
  (swim speed, release region, temperature on/off, horizontal diffusion, variable-name suffix). Its
  `extract_tracers()` (memoized) pulls per-cell/per-timestep concentration, age, and light-limitation data
  out of a run's map output. The module-level `tracer_groups` list at the bottom is the master catalog of
  every configured run (v12–v17 variants) — other code selects runs from this list by tag/swim/initial
  filters rather than constructing `TracerGroup` directly.
- **`bloom_common.py`** — Grab-bag shared by most other modules: DFM/DWAQ run configuration
  (`configure_dfm_*`), a memoized `load_model()` used by `TracerGroup.model`, remote-sensing chlorophyll
  scene loading/reprojection (`load_chl_scenes`, `chl_from_RS`), region polygons, a DWAQ process-table
  generator for age tracers (`CART`), and the light-limitation math (`calc_kLight`, `ratio`) that
  `solveNP.py` and the opt script build on.
- **`common.py`** — A smaller, separate light-limitation ODE module (`k_indef`/`kLight_m`/`kLight`,
  `sigma_median`, xarray `resample`). Despite similar-sounding functions to `bloom_common.py`, it's a
  distinct, standalone file (earlier prototype) with no cross-imports — don't conflate the two.
- **`solveNP.py`** / **`solveNP_graze.py`** — Integrate simple nutrient-phytoplankton (N-P) or
  N-P-zooplankton (N-P-Z, with grazing) ODEs per tracer cell via `scipy.integrate.solve_ivp`, driven by a
  `TracerGroup`'s extracted tracer data. These are the standalone/simpler precursors to the
  `LagrangianModel`/`NPZIVModel` classes used in the main opt script.
- **`bloom_lagrangian_zoop_etal_opt_20260729.py`** — The main parameter-tuning driver. Defines
  `LagrangianModel` and `NPZIVModel` (N-P-Z-Infected-Virus ODE, subclassable `npziv_diff`), fits them
  against mooring chlorophyll/salinity/temperature and RS chlorophyll data via `loss`/`cost` and
  `scipy.optimize.minimize(method='powell')` over ~10 biological parameters (grazing rate, virus infection
  rate, nutrient supplement, light saturation, etc.). Consumes pre-run DFM/DWAQ tracer output through
  `tracer_group.tracer_groups` / `bloom_common.load_model` — it does not itself launch DFM/DWAQ runs.
- **`golumn.py`** — Wraps running a 1-D GOTM water-column model (`Golumn` class) as a simplified
  vertical-mixing alternative/complement to the full 3-D DFM runs; writes GOTM yaml config, shells out to
  the `gotm` executable, reads back results. Config files live in `gotm/rouse_up/`.
- **Observation/forcing loaders** — `insitu_temperature.py` (unified in-situ temperature/water-level
  DataFrame from mooring CSVs, NOAA CO-OPS, USGS NWIS), `met_updater.py` (`MetUpdater`: replaces DFM
  meteorological air-temperature forcing with interpolated NOAA station data), `asos.py` (NOAA ASOS
  5-minute cloudiness data for solar forcing), `RS_temperature/temp_utils.py` (AVHRR SST fetch/cache from
  a CenCOOS THREDDS endpoint plus grid watermasking).

## Data directories

- `waq_tables_rusty_laptop/` — checked-in reference copies of the Delft3D-WAQ process-library tables
  (`proces.csv`, `items.csv`, `stochi.csv`, etc.), needed when defining custom DWAQ biogeochemical
  processes (paired with `bloom_common.CART`/`configure_dfm_*`).
  Note this directory is *not* gitignored despite the "local"/laptop-specific name — treat edits here as
  affecting the checked-in reference tables, not a throwaway local cache.
- `usgs_peterson/` — USGS Peterson cruise water-quality CSV for 2022, used as an independent observation
  source.
- `sfbofs/`, `gotm/` — validation notebook for SF Bay Operational Forecast System extraction, and GOTM
  case config, respectively.
