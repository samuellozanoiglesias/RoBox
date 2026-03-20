# Author: Samuel Lozano
from .env import OctagonEnv
from .navigation import NavigationController, clip_to_octagon, estimate_travel_time
from .observations import build_global_state, build_observation
from .rewards import RewardShaper
from .trial_types import TrialResult

try:
    from .mappo import MAPPOActor, MAPPOBuffer, MAPPOCritic, MAPPOTrainer
except Exception:  # pragma: no cover - optional dependency path
    MAPPOActor = None
    MAPPOCritic = None
    MAPPOBuffer = None
    MAPPOTrainer = None

try:
    from .experiment_runner import ExperimentLogger, run_experiment
except Exception:  # pragma: no cover - optional dependency path
    ExperimentLogger = None
    run_experiment = None

try:
    from .solo_analysis import (
        plot_distance_phigh,
        plot_learning_curves,
        plot_location_heatmap,
        plot_rt_analysis,
        plot_spatial_phigh,
        run_solo_analysis,
    )
except Exception:  # pragma: no cover - optional dependency path
    plot_learning_curves = None
    plot_location_heatmap = None
    plot_spatial_phigh = None
    plot_distance_phigh = None
    plot_rt_analysis = None
    run_solo_analysis = None

try:
    from .social_analysis import (
        plot_delta_plow,
        plot_opponent_position,
        plot_opponent_speed_corr,
        plot_preference_shift,
        plot_social_spatial,
        run_social_analysis,
    )
except Exception:  # pragma: no cover - optional dependency path
    plot_preference_shift = None
    plot_social_spatial = None
    plot_delta_plow = None
    plot_opponent_speed_corr = None
    plot_opponent_position = None
    run_social_analysis = None

try:
    from .rl_comparison import fit_logistic_regression, plot_agent_sensitivity, run_rl_comparison
except Exception:  # pragma: no cover - optional dependency path
    plot_agent_sensitivity = None
    fit_logistic_regression = None
    run_rl_comparison = None

try:
    from .attractor_model import AttractorModel, FitResult
except Exception:  # pragma: no cover - optional dependency path
    AttractorModel = None
    FitResult = None

try:
    from .attractor_analysis import run_attractor_analysis
except Exception:  # pragma: no cover - optional dependency path
    run_attractor_analysis = None

try:
    from .pipeline import check_reproduction, generate_summary_dashboard
except Exception:  # pragma: no cover - optional dependency path
    check_reproduction = None
    generate_summary_dashboard = None

__all__ = [
    "OctagonEnv",
    "MAPPOActor",
    "MAPPOCritic",
    "MAPPOBuffer",
    "MAPPOTrainer",
    "NavigationController",
    "clip_to_octagon",
    "estimate_travel_time",
    "ExperimentLogger",
    "run_experiment",
    "plot_learning_curves",
    "plot_location_heatmap",
    "plot_spatial_phigh",
    "plot_distance_phigh",
    "plot_rt_analysis",
    "run_solo_analysis",
    "plot_preference_shift",
    "plot_social_spatial",
    "plot_delta_plow",
    "plot_opponent_speed_corr",
    "plot_opponent_position",
    "run_social_analysis",
    "plot_agent_sensitivity",
    "fit_logistic_regression",
    "run_rl_comparison",
    "AttractorModel",
    "FitResult",
    "run_attractor_analysis",
    "generate_summary_dashboard",
    "check_reproduction",
    "RewardShaper",
    "TrialResult",
    "build_observation",
    "build_global_state",
]
