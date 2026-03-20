# RoBox

**Author:** Samuel Lozano

Package-ready implementation of `OctagonEnv` with a src-layout Python package.

## Structure

```text
RoBox/
	pyproject.toml
	octagon_env.py                  # compatibility shim
	src/
		robox_octagon/
			__init__.py
			env.py                      # OctagonEnv
			rewards.py                  # RewardShaper
			observations.py             # build_observation, build_global_state
			mappo.py                    # MAPPOActor, MAPPOCritic, MAPPOBuffer, MAPPOTrainer
						navigation.py               # NavigationController and octagon clipping
						experiment_runner.py        # full experimental protocol orchestration
						solo_analysis.py            # Figure 1 / S2-S3 solo behavior analysis plots
						social_analysis.py          # Figure 2 / S5-S6 competitive behavior analysis plots
						rl_comparison.py            # Figure 3 / S8-S9 RL agent-type comparison analysis
						attractor_model.py          # Figure 4 bistable attractor decision model
						attractor_analysis.py       # Figure 4 attractor-model visualization plots
			trial_types.py              # TrialResult dataclass
			spaces.py                   # gym/gymnasium Box helper
```

## Install As Package

This repository is prepared to be installed in a virtual environment (not created automatically):

```bash
pip install -e .
```

Optional extras:

```bash
pip install -e .[gym]
pip install -e .[gymnasium]
pip install -e .[train]
pip install -e .[experiment]
pip install -e .[analysis]
```

## Import

```python
from robox_octagon import OctagonEnv
from robox_octagon.mappo import MAPPOTrainer

env = OctagonEnv()
obs = env.reset(context="social")

trainer = MAPPOTrainer(
	obs_dim=env.obs_dim,
	global_state_dim=2 * env.obs_dim + 3,
)

# Full experiment protocol (10 pairs, checkpoint resume, CSV + HDF5 logs)
# python -m robox_octagon.experiment_runner --config configs/experiment.yaml

# Solo behavior analysis plots
# from robox_octagon.solo_analysis import run_solo_analysis
# run_solo_analysis(df_trials, output_dir="logs/solo_analysis")

# Social behavior analysis plots
# from robox_octagon.social_analysis import run_social_analysis
# run_social_analysis(df_trials_post_learning, output_dir="logs/social_analysis")

# RL agent comparison (social vs socially-blind)
# from robox_octagon.rl_comparison import run_rl_comparison
# run_rl_comparison(df_social, df_blind, output_dir="logs/rl_comparison")

# Figure 4 attractor-model fitting
# from robox_octagon.attractor_model import AttractorModel
# model = AttractorModel()
# best_solo = model.fit_solo(df_trials)
# best_social = model.fit_social(df_trials)

# Figure 4 attractor-model visualization suite
# from robox_octagon.attractor_analysis import run_attractor_analysis
# run_attractor_analysis(model_solo, model_social, df_trials, output_dir="logs/attractor")

# End-to-end summary dashboard + reproducibility checks
# python -m robox_octagon.pipeline --config config.yaml

# One-command full integration launcher (forces training on)
# python run_training.py --config config.yaml
```

Backward-compatible import remains available:

```python
from octagon_env import OctagonEnv
```