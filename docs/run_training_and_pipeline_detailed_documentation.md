# RoBox Training Launcher and Pipeline: Detailed Structural Documentation

## 1) Scope and Goal of This Document

This document explains, in detail, the structure and purpose of:

- `run_training.py` (the launcher script you asked about)
- The downstream orchestration code called by that launcher, mainly:
  - `src/robox_octagon/pipeline.py`
  - `src/robox_octagon/experiment_runner.py`
  - The analysis/model modules invoked by the pipeline

The objective is to make the full execution chain explicit: what each part does, why it exists, what it consumes, and what it produces.


## 2) High-Level Execution Chain

When you run:

```bash
python run_training.py
```

the flow is:

1. Parse CLI arguments in `run_training.py`
2. Resolve and load `config.yaml`
3. Optionally force `pipeline.training.run_training = true`
4. Import and call `robox_octagon.pipeline.main(config_path=...)`
5. In pipeline:
   - Optionally run/continue training via `experiment_runner.run_experiment(...)`
   - Load trial data from generated logs or explicit CSV
   - Run solo analysis
   - Run social analysis
   - Run social-vs-blind RL comparison analysis
   - Fit solo/social attractor models and run attractor analysis
   - Build summary dashboard PDF
   - Run reproducibility checks and save report


## 3) `run_training.py` Detailed Structure

### 3.1 Imports and Why They Exist

- `argparse`: command-line interface (`--config`, `--respect-config`)
- `importlib`: dynamic import of `robox_octagon.pipeline`
- `sys`: inject `src` path so local package imports work from repo root
- `tempfile`: create temporary config file when forcing training on
- `Path`: robust file path handling
- `Dict`, `Any`: type annotations
- `yaml` (PyYAML): read/write YAML configs

Why dynamic import is used:

- The launcher does not directly import every module at import time.
- It resolves package path first (`src` insertion), then imports pipeline safely.


### 3.2 Helper Function: `_load_yaml(path)`

Purpose:

- Reads a YAML file and returns a Python dictionary.

Behavior:

- Opens with UTF-8
- Uses `yaml.safe_load` (safe parser)
- Returns `{}` if YAML is empty

Why it exists:

- Keeps config load logic isolated and reusable.


### 3.3 Helper Function: `_write_yaml(path, data)`

Purpose:

- Writes a dictionary to YAML on disk.

Behavior:

- Uses UTF-8
- Uses `yaml.safe_dump(..., sort_keys=False)` to preserve human-friendly key order

Why it exists:

- Needed for the temporary forced-training config mode.


### 3.4 `main()` in `run_training.py`

This is the launcher's complete control center.

#### Step A: Parse arguments

- `--config` (default `config.yaml`): selects base pipeline config.
- `--respect-config`: if set, do not override training flag.

Meaning of these modes:

- Default mode: launcher forces training to run.
- Respect mode: launcher obeys whatever config says (`true` or `false`).

#### Step B: Resolve repository and `src` import path

- Detect repo root as parent of `run_training.py`.
- If `src` exists and is not in `sys.path`, insert it at index 0.

Why this matters:

- Ensures `import robox_octagon.pipeline` works without separate installation step.

#### Step C: Import and retrieve pipeline entry point

- `pipeline_module = importlib.import_module("robox_octagon.pipeline")`
- `pipeline_main = getattr(pipeline_module, "main")`

Why this is robust:

- It defers loading pipeline until path setup is complete.

#### Step D: Resolve config path and validate existence

- Joins repo root + provided `--config`
- Raises `FileNotFoundError` early if missing

#### Step E: Branch behavior

Branch 1: `--respect-config` enabled

- Prints selected config path
- Calls pipeline directly with that config

Branch 2: default mode (force training)

- Loads config YAML
- Ensures nested keys exist:
  - `pipeline`
  - `pipeline.training`
- Sets `pipeline.training.run_training = True`
- Writes modified config to temporary YAML file
- Calls pipeline with temp config

Why this branch exists:

- Gives a one-command way to guarantee training executes, even if base config disabled it.


## 4) Configuration Files and Their Roles

### 4.1 `config.yaml` (pipeline-level settings)

This controls global orchestration behavior:

- Which experiment config to use (`pipeline.experiment_config`)
- Where final results are saved (`pipeline.results_dir`)
- Whether to run training (`pipeline.training.run_training`)
- Optional data sources (`pipeline.data.agents_trials_csv`, `pipeline.data.blind_trials_csv`)
- Attractor fit controls (`pipeline.attractor.*`)


### 4.2 `configs/experiment.yaml` (training protocol settings)

This controls experiment/training internals:

- Pair count and seeds
- Log/checkpoint paths and frequency
- Environment physics/time constants
- MAPPO hyperparameters
- Multi-phase protocol lengths and thresholds


## 5) Downstream Orchestration: `pipeline.py`

`pipeline.py` is the true end-to-end workflow coordinator.

### 5.1 Structural Responsibilities

1. Prepare output directories
2. Optionally run training
3. Load datasets
4. Execute analysis modules
5. Fit computational models
6. Produce dashboard and reproducibility report


### 5.2 Key Utility Functions and Purpose

- `_ensure_dir(path)`: create directory tree and return `Path`
- `_load_yaml(path)`: read YAML config safely
- Geometry helpers (`_octagon_vertices`, `_point_in_octagon`): octagon shape math for plotting and masking
- Plot helpers (`_plot_learning`, `_plot_distance_45`, ...): build dashboard panels
- Data reshape helpers (`_aligned_xy`, `_plow_heatmap_data`, ...): align positions and compute map statistics
- Repro check helpers (`_fit_logit_opp`, `_forced_rt_medians`, ...): compute paper-style validation metrics

These helpers make `main()` readable by separating concerns.


### 5.3 Data Loading Strategy in Pipeline

- `_load_agent_trials(...)`:
  - If `pipeline.data.agents_trials_csv` is set, use it directly.
  - Otherwise, collect `logs/pair_*/trials.csv` and concatenate.
- `_load_blind_trials(...)`:
  - If explicit blind CSV path exists, load it.
  - Else try filtering `agent_type` in agent data.
  - Else fallback to full agent data to keep pipeline runnable.

Why this design is useful:

- Allows fully automatic usage after training.
- Also supports external/offline datasets.


### 5.4 Training Trigger in Pipeline

`_run_training_or_resume(...)` behavior:

- Reads `pipeline.training.run_training`.
- If false, does nothing (analysis-only mode).
- If true, builds seed pairs and calls `run_experiment(...)` for each pair.

Note:

- This is exactly what `run_training.py` manipulates in default mode.


### 5.5 Pipeline `main()` Step-by-Step

The `main(config_path="config.yaml")` function executes this exact sequence:

1. Load pipeline config YAML
2. Load referenced experiment config YAML
3. Create output dirs under `results`:
   - `results/solo`
   - `results/social`
   - `results/comparison`
   - `results/attractor`
4. Training stage (optional)
5. Load trials data (`df_agents`) and blind data (`df_blind`)
6. Solo analysis: `run_solo_analysis(...)`
7. Social analysis: `run_social_analysis(...)`
8. RL comparison: `run_rl_comparison(...)`
9. Attractor modeling:
   - Instantiate solo/social `AttractorModel`
   - Optionally downsample fit rows
   - Fit solo and social parameters
   - Run attractor visualization analysis
   - Save `attractor_best_params.csv`
10. Build consolidated dashboard PDF
11. Run reproducibility tests and write text report
12. Print saved output locations


## 6) Experiment Runtime Core: `experiment_runner.py`

This module is where MAPPO training episodes are executed and trial logs are generated.

### 6.1 Important Pieces

- `ExperimentLogger`:
  - Caches trial and block records in memory
  - Saves `trials.csv`, `blocks.csv`, `summary.h5`
  - Can load existing logs for resume scenarios

- `RunState` dataclass:
  - Serializable progress state for two-stage protocol
  - Tracks phase, block, trial counters, and progression markers

- Checkpoint helpers:
  - `_latest_checkpoint(...)`
  - `_save_run_checkpoint(...)`
  - `_load_run_checkpoint(...)`

These allow interruption-safe continuation.


### 6.2 Trial Row Construction

`_build_trial_rows(...)` converts raw environment/trainer trial outputs into normalized tabular rows with fields such as:

- trial identifiers
- phase (`solo`/`social`)
- agent identity
- onset coordinates
- high/low patch ids and separation angle
- choice encoding (0=high, 1=low)
- RT, travel distance
- self and opponent distance-to-patch metrics
- winner id (social)
- raw and shaped rewards

Why this is critical:

- Every downstream analysis assumes this schema.


### 6.3 `run_experiment(...)` Protocol Flow

Inputs:

- pair id
- seed A and seed B
- full experiment config

Internal runtime sequence:

1. Build pair log/checkpoint directories
2. Build `OctagonEnv`
3. Initialize `MAPPOTrainer`
4. Resume from latest checkpoint if present
5. Run Phase 1 (solo pre-training per actor)
   - With early stop rule based on rolling P(high)
6. Run Phase 2 (interleaved blocks)
   - Stage order per block: solo_A -> solo_B -> social
   - After social stage, log block summary metrics
7. Periodically save checkpoints/logs
8. Save final checkpoint/logs
9. Return run summary dictionary


### 6.4 Batch Execution and CLI in `experiment_runner.py`

- `_pair_seeds(...)` computes paired seeds from experiment config.
- `main()` supports serial or multiprocessing execution over pairs.
- Writes `experiment_index.csv` under log dir as index of pair runs.


## 7) Downstream Analysis Modules Invoked by Pipeline

### 7.1 `run_solo_analysis(...)` in `solo_analysis.py`

Purpose:

- Generate solo-behavior figures and save to `results/solo`.

Outputs:

- `plot1_learning_curves.png`
- `plot2_location_heatmap.png`
- `plot3_spatial_phigh.png`
- `plot4_distance_phigh.png`
- `plot5_rt_analysis.png`


### 7.2 `run_social_analysis(...)` in `social_analysis.py`

Purpose:

- Quantify social modulation effects and social spatial signatures.

Outputs:

- `plot6_preference_shift.png`
- `plot7_8_social_spatial.png`
- `plot9_delta_plow.png`
- `plot10_opponent_speed_corr.png`
- `plot11_opponent_position.png`


### 7.3 `run_rl_comparison(...)` in `rl_comparison.py`

Purpose:

- Compare social-aware vs socially-blind behavior patterns.

Main operations:

- Adds `agent_type` labels
- Produces sensitivity/regression and context-shift panels
- Creates position-specific comparison outputs

Named outputs include:

- `plot12_social_agent_sensitivity.png`
- `plot13_blind_agent_sensitivity.png`
- `plot14_context_shift_comparison.png`
- Additional comparison figures written by the function


### 7.4 `run_attractor_analysis(...)` in `attractor_analysis.py`

Purpose:

- Build Figure-4-style model diagnostics/visualizations for fitted attractor models.

Outputs include:

- `plot16_phase_space_solo.png`
- `plot17_solo_phigh_heatmap.png`
- `plot18_forced_rt.png`
- `plot19_social_trajectories.png`
- Additional attractor plots from the same routine


## 8) Attractor Computational Model: `attractor_model.py`

### 8.1 What It Represents

`AttractorModel` encodes a two-node bistable decision dynamical system:

- Node 1: drive toward high-value patch
- Node 2: drive toward low-value patch
- Decision = first node crossing threshold (or higher terminal activity if no crossing)

### 8.2 Core Simulation API

- `simulate_trial(...)`: Monte-Carlo estimate of P(high) and mean RT for one trial condition
- `simulate_trajectories(...)`: full trajectory ensembles for plotting
- `simulate_dataset(...)`: per-row model predictions for a full DataFrame

### 8.3 Fitting API

- `log_likelihood(...)`: Bernoulli log-likelihood of observed choices
- `_fit_context(...)`: shotgun random search + Nelder-Mead local optimization
- `fit_solo(...)`: fit 6-parameter solo model
- `fit_social(...)`: fit 9-parameter social model

Why this matters in pipeline:

- Fitted parameters are used for attractor analyses and dashboard overlays.


## 9) Reproducibility Checks in `pipeline.py`

`check_reproduction(...)` aggregates multiple tests and produces a textual PASS/FAIL report.

Examples of evaluated properties:

- Post-learning solo preference threshold
- Solo-to-social preference drop magnitude and significance
- Opponent sensitivity coefficient sign/significance
- Socially-blind expected lack of opponent sensitivity
- Forced-trial RT ordering
- Center-seeking onset-position behavior

Output:

- `results/reproducibility_report.txt`


## 10) Expected Artifacts After a Full Run

Depending on config and existing logs, you should see:

- Training logs/checkpoints under `logs/pair_*`
  - `trials.csv`
  - `blocks.csv`
  - `summary.h5`
  - `checkpoints/experiment_state_*.pt`
- Pipeline outputs under `results`
  - `solo/*.png`
  - `social/*.png`
  - `comparison/*.png`
  - `attractor/*.png`
  - `attractor/attractor_best_params.csv`
  - `summary_dashboard.pdf`
  - `reproducibility_report.txt`


## 11) Two Important Operational Modes

### Mode A: Force training

Command:

```bash
python run_training.py
```

Effect:

- Launcher creates temp config with `pipeline.training.run_training=true` and executes full pipeline.


### Mode B: Respect existing config

Command:

```bash
python run_training.py --respect-config
```

Effect:

- Pipeline obeys `config.yaml` exactly.
- If training is disabled there, it runs analysis from available CSVs/logs only.


## 12) Why This Architecture Is Useful

- Clear separation of concerns:
  - launcher control
  - experiment runtime/training
  - analytics and reporting
- Resume robustness via checkpoints
- Supports both full training and analysis-only workflows
- Keeps outputs organized and reproducibility-oriented


## 13) Practical Reading Order for Future Maintenance

For quickest understanding/debugging, read in this order:

1. `run_training.py`
2. `config.yaml`
3. `src/robox_octagon/pipeline.py` (`main`, load/run stages)
4. `configs/experiment.yaml`
5. `src/robox_octagon/experiment_runner.py` (protocol loops + logging)
6. Analysis modules (`solo_analysis.py`, `social_analysis.py`, `rl_comparison.py`)
7. `attractor_model.py` and `attractor_analysis.py`
