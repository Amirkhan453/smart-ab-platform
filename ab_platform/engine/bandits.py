import math
from typing import Dict, List
import numpy as np


class MultiArmedBanditEngine:
    """
    Движок адаптивной аллокации трафика (Multi-Armed & Contextual Bandits):
    - Epsilon-Greedy с затуханием (decaying epsilon)
    - Upper Confidence Bound (UCB1)
    - Thompson Sampling (Байесовский Beta-Bernoulli сэмплинг)
    - Contextual LinUCB / Ridge Bandit для персонализации по фичам пользователя
    """

    @staticmethod
    def select_arm_epsilon_greedy(
        successes: np.ndarray,
        trials: np.ndarray,
        epsilon: float = 0.10,
        decay_rate: float = 0.001,
        random_seed: int = 42,
    ) -> Dict[str, object]:
        rng = np.random.default_rng(random_seed)
        total_steps = int(np.sum(trials))
        effective_eps = epsilon / (1.0 + decay_rate * total_steps)

        means = np.where(trials > 0, successes / np.maximum(trials, 1), 0.0)
        if rng.random() < effective_eps:
            chosen_arm = int(rng.integers(0, len(means)))
            mode = "explore"
        else:
            chosen_arm = int(np.argmax(means))
            mode = "exploit"

        return {
            "chosen_arm": chosen_arm,
            "mode": mode,
            "effective_epsilon": round(float(effective_eps), 5),
        }

    @staticmethod
    def select_arm_ucb1(
        successes: np.ndarray, trials: np.ndarray, exploration_c: float = 2.0
    ) -> Dict[str, object]:
        """Алгоритм Upper Confidence Bound (UCB): выбирает руку с максимальной верхней границей."""
        for idx, n_i in enumerate(trials):
            if n_i == 0:
                return {"chosen_arm": int(idx), "ucb_scores": []}

        total_n = float(np.sum(trials))
        means = successes / trials
        uncertainty = np.sqrt((exploration_c * np.log(total_n)) / trials)
        ucb_scores = means + uncertainty
        chosen_arm = int(np.argmax(ucb_scores))

        return {
            "chosen_arm": chosen_arm,
            "ucb_scores": [round(float(s), 5) for s in ucb_scores],
        }

    @staticmethod
    def thompson_sampling_allocation(
        successes: np.ndarray,
        trials: np.ndarray,
        n_mc_draws: int = 5000,
        random_seed: int = 42,
    ) -> Dict[str, object]:
        """
        Thompson Sampling: вычисляет апостериорную вероятность того, что каждый вариант лучший,
        и возвращает оптимальные доли аллокации трафика.
        """
        rng = np.random.default_rng(random_seed)
        failures = np.maximum(0, trials - successes)
        alphas = successes + 1.0
        betas = failures + 1.0

        draws = rng.beta(alphas, betas, size=(n_mc_draws, len(successes)))
        best_arms = np.argmax(draws, axis=1)
        win_probs = np.bincount(best_arms, minlength=len(successes)) / float(n_mc_draws)

        return {
            "recommended_arm": int(np.argmax(win_probs)),
            "optimal_traffic_shares": [round(float(p), 4) for p in win_probs],
        }

    @staticmethod
    def contextual_linucb_select(
        user_context: np.ndarray,
        arm_gram_matrices: List[np.ndarray],
        arm_reward_vectors: List[np.ndarray],
        alpha_explore: float = 1.0,
    ) -> Dict[str, object]:
        """
        Контекстный бандит (LinUCB): предсказывает ожидаемую награду каждой руки
        с учетом вектора признаков конкретного пользователя (устройство, география, история).
        """
        x = np.asarray(user_context, dtype=float).reshape(-1, 1)
        scores = []

        for A_mat, b_vec in zip(arm_gram_matrices, arm_reward_vectors):
            A_inv = np.linalg.inv(A_mat)
            theta_hat = A_inv @ b_vec.reshape(-1, 1)
            mean_pred = float((theta_hat.T @ x)[0, 0])
            conf_width = alpha_explore * math.sqrt(max(float((x.T @ A_inv @ x)[0, 0]), 0.0))
            scores.append(mean_pred + conf_width)

        return {
            "chosen_arm": int(np.argmax(scores)),
            "contextual_scores": [round(s, 5) for s in scores],
        }
