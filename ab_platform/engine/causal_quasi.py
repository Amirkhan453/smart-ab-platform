import math
from typing import Dict, List
import numpy as np
import pandas as pd
from scipy import optimize, stats


class CausalAndMLDebiasingEngine:
    """
    Модуль многомерного дебайасинга (LinkedIn methodology) и Квази-экспериментов (когда классический A/B невозможен):
    - Многомерная ковариатная корректировка с раздельными моделями в тесте и контроле (Count & Ratio metrics)
    - Propensity Score Matching (PSM) и Doubly Robust (AIPW) оценка на исторических данных
    - Difference-in-Differences (DiD) для оценки региональных и нерандомизированных запусков
    """

    @staticmethod
    def multivariate_debiasing_ate(
        df: pd.DataFrame,
        outcome_col: str,
        feature_cols: List[str],
        variant_col: str = "variant",
        alpha: float = 0.05,
    ) -> dict:
        """
        Многомерный дебайасинг по схеме LinkedIn (раздельные регрессии mu_T(X) и mu_C(X) на наборе ковариат):
        Позволяет сжать дисперсию сильнее одномерного CUPED без внесения смещения.
        """
        data = df.copy()
        X_all = data[feature_cols].to_numpy(dtype=float)
        X_mean = np.mean(X_all, axis=0)
        X_centered = X_all - X_mean

        c_mask = (data[variant_col] == "control").to_numpy()
        t_mask = (data[variant_col] == "treatment").to_numpy()

        y_c = data.loc[c_mask, outcome_col].to_numpy(dtype=float)
        y_t = data.loc[t_mask, outcome_col].to_numpy(dtype=float)
        X_c = X_centered[c_mask]
        X_t = X_centered[t_mask]

        d = X_all.shape[1]
        beta_c = np.linalg.lstsq(X_c.T @ X_c + 1e-4 * np.eye(d), X_c.T @ (y_c - np.mean(y_c)), rcond=None)[0]
        beta_t = np.linalg.lstsq(X_t.T @ X_t + 1e-4 * np.eye(d), X_t.T @ (y_t - np.mean(y_t)), rcond=None)[0]

        n_c, n_t = len(y_c), len(y_t)
        n_total = n_c + n_t
        beta_pooled = (n_t / n_total) * beta_c + (n_c / n_total) * beta_t

        y_c_adj = y_c - X_c @ beta_pooled
        y_t_adj = y_t - X_t @ beta_pooled

        var_raw = float(np.var(y_t, ddof=1) / n_t + np.var(y_c, ddof=1) / n_c)
        var_adj = float(np.var(y_t_adj, ddof=1) / n_t + np.var(y_c_adj, ddof=1) / n_c)
        var_red_pct = (1.0 - var_adj / max(var_raw, 1e-15)) * 100.0

        ate = float(np.mean(y_t_adj) - np.mean(y_c_adj))
        se_adj = math.sqrt(max(var_adj, 1e-15))
        z_stat = ate / se_adj
        p_val = float(2.0 * (1.0 - stats.norm.cdf(abs(z_stat))))

        return {
            "ate_debiased": ate,
            "se_debiased": se_adj,
            "p_value": p_val,
            "variance_reduction_pct": round(var_red_pct, 2),
            "is_significant": bool(p_val < alpha),
        }

    @staticmethod
    def propensity_score_matching_and_aipw(
        df: pd.DataFrame,
        treatment_col: str,
        outcome_col: str,
        confounder_cols: List[str],
    ) -> dict:
        """
        Оценка эффекта без рандомизации (Quasi-Experiment):
        1) Оценивает Propensity Score e(X) = P(T=1 | X) через логистическую регрессию с L2-регуляризацией
        2) Выполняет 1-to-1 Nearest Neighbor Matching по логиту propensity score (ATT)
        3) Вычисляет Doubly Robust (AIPW) оценку среднего эффекта (ATE)
        """
        X_raw = df[confounder_cols].to_numpy(dtype=float)
        X_std = (X_raw - np.mean(X_raw, axis=0)) / np.maximum(np.std(X_raw, axis=0), 1e-8)
        X_design = np.column_stack([np.ones(len(df)), X_std])

        t = df[treatment_col].to_numpy(dtype=float)
        y = df[outcome_col].to_numpy(dtype=float)

        def neg_log_likelihood(w: np.ndarray) -> float:
            logits = np.clip(X_design @ w, -20.0, 20.0)
            probs = 1.0 / (1.0 + np.exp(-logits))
            nll = -np.sum(t * np.log(probs + 1e-12) + (1.0 - t) * np.log(1.0 - probs + 1e-12))
            return float(nll + 0.5 * np.sum(w[1:] ** 2))

        opt = optimize.minimize(neg_log_likelihood, np.zeros(X_design.shape[1]), method="BFGS")
        logits = np.clip(X_design @ opt.x, -15.0, 15.0)
        propensity = np.clip(1.0 / (1.0 + np.exp(-logits)), 0.02, 0.98)

        # 1-to-1 Nearest Neighbor Matching на логитах Propensity Score
        treated_idx = np.where(t == 1)[0]
        control_idx = np.where(t == 0)[0]

        treated_scores = logits[treated_idx]
        control_scores = logits[control_idx]

        matched_control_outcomes = np.empty(len(treated_idx), dtype=float)
        for i, score_t in enumerate(treated_scores):
            nearest_c_pos = int(np.argmin(np.abs(control_scores - score_t)))
            matched_control_outcomes[i] = y[control_idx[nearest_c_pos]]

        att_psm = float(np.mean(y[treated_idx] - matched_control_outcomes))

        # Doubly Robust (AIPW) оценка
        X_c = X_design[control_idx]
        X_t = X_design[treated_idx]
        beta_c = np.linalg.lstsq(X_c, y[control_idx], rcond=None)[0]
        beta_t = np.linalg.lstsq(X_t, y[treated_idx], rcond=None)[0]

        mu_0 = X_design @ beta_c
        mu_1 = X_design @ beta_t

        aipw_1 = mu_1 + (t / propensity) * (y - mu_1)
        aipw_0 = mu_0 + ((1.0 - t) / (1.0 - propensity)) * (y - mu_0)
        diff_aipw = aipw_1 - aipw_0

        ate_dr = float(np.mean(diff_aipw))
        se_dr = float(np.std(diff_aipw, ddof=1) / math.sqrt(len(df)))
        p_val_dr = float(2.0 * (1.0 - stats.norm.cdf(abs(ate_dr / max(se_dr, 1e-12)))))

        return {
            "naive_biased_diff": float(np.mean(y[treated_idx]) - np.mean(y[control_idx])),
            "att_psm_matched": round(att_psm, 4),
            "ate_doubly_robust": round(ate_dr, 4),
            "se_doubly_robust": round(se_dr, 4),
            "p_value_doubly_robust": p_val_dr,
        }

    @staticmethod
    def difference_in_differences(
        df: pd.DataFrame,
        outcome_col: str,
        is_treated_group_col: str,
        is_post_period_col: str,
    ) -> dict:
        """
        Метод Двойных Разностей (Difference-in-Differences, DiD):
        Оценивает интеракцию (Treated_Group x Post_Period) с проверкой стат. значимости.
        """
        g = df[is_treated_group_col].to_numpy(dtype=float)
        post = df[is_post_period_col].to_numpy(dtype=float)
        inter = g * post
        y = df[outcome_col].to_numpy(dtype=float)

        X = np.column_stack([np.ones(len(df)), g, post, inter])
        beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        residuals = y - X @ beta

        n, k = X.shape
        sigma2 = float(np.sum(residuals ** 2) / max(n - k, 1))
        cov_beta = sigma2 * np.linalg.inv(X.T @ X)
        se_did = float(math.sqrt(max(cov_beta[3, 3], 1e-15)))

        did_effect = float(beta[3])
        t_stat = did_effect / se_did
        p_val = float(2.0 * (1.0 - stats.t.cdf(abs(t_stat), df=max(n - k, 1))))

        return {
            "did_treatment_effect": round(did_effect, 4),
            "standard_error": round(se_did, 4),
            "t_statistic": round(t_stat, 4),
            "p_value": p_val,
            "is_significant": bool(p_val < 0.05),
        }
