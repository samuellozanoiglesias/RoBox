from __future__ import annotations

from typing import Optional


class RewardShaper:
    """Discounted travel-cost reward with optional dense shaping helpers."""

    def __init__(
        self,
        gamma: float = 0.98,
        action_cost_normalized: float = 0.05,
        action_cost_paper: float = 5.0,
        high_value_normalized: float = 1.0,
        low_value_normalized: float = 2.0 / 3.0,
        high_value_paper: float = 750.0,
        low_value_paper: float = 500.0,
        use_normalized: bool = True,
    ) -> None:
        self.gamma = float(gamma)
        self.action_cost_normalized = float(action_cost_normalized)
        self.action_cost_paper = float(action_cost_paper)
        self.high_value_normalized = float(high_value_normalized)
        self.low_value_normalized = float(low_value_normalized)
        self.high_value_paper = float(high_value_paper)
        self.low_value_paper = float(low_value_paper)
        self.use_normalized = bool(use_normalized)

    def _choice_value(self, choice: Optional[str]) -> float:
        if choice == "high":
            return self.high_value_normalized if self.use_normalized else self.high_value_paper
        if choice == "low":
            return self.low_value_normalized if self.use_normalized else self.low_value_paper
        return 0.0

    def _action_cost(self) -> float:
        return self.action_cost_normalized if self.use_normalized else self.action_cost_paper

    def _discounted_reward(self, choice: Optional[str], travel_dist: Optional[float]) -> float:
        if choice is None or travel_dist is None:
            return 0.0

        d = max(0.0, float(travel_dist))
        gamma_pow = float(self.gamma**d)
        r_choice = self._choice_value(choice)
        action_cost = self._action_cost()
        if abs(1.0 - self.gamma) < 1e-12:
            return gamma_pow * r_choice - action_cost * d
        cost_term = action_cost * (1.0 - gamma_pow) / (1.0 - self.gamma)
        return gamma_pow * r_choice - cost_term

    def compute_solo_reward(self, choice: Optional[str], travel_dist: Optional[float]) -> float:
        return float(self._discounted_reward(choice=choice, travel_dist=travel_dist))

    def compute_social_reward(
        self,
        agent_id: int,
        winner_id: Optional[int],
        choice_winner: Optional[str],
        travel_dist_winner: Optional[float],
    ) -> float:
        if winner_id is None or agent_id != winner_id:
            return 0.0
        return float(self._discounted_reward(choice=choice_winner, travel_dist=travel_dist_winner))

    def compute_reward(
        self,
        agent_id: int,
        winner_id: Optional[int],
        choice: Optional[str],
        travel_dist: Optional[float],
        context: str,
    ) -> float:
        if context == "solo":
            return self.compute_solo_reward(choice=choice, travel_dist=travel_dist)
        return self.compute_social_reward(
            agent_id=agent_id,
            winner_id=winner_id,
            choice_winner=choice,
            travel_dist_winner=travel_dist,
        )
