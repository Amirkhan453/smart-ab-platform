import math
from typing import List, Optional
import numpy as np
import pandas as pd
from scipy import optimize, stats


class CausalAndMLDebiasingEngine:
    """
    Модуль многомерного дебайасинга (LinkedIn methodology) и Квази-экспериментов:
    - Многомерная ковариатная корректировка с раздельными моделями в тесте и контроле
    - Propensity Score Matching (PSM) и Doubly Robust (AIPW) с проверкой Common Support (Overlap)
    - Difference-in-Differences (DiD) с гетероскедастично-робастными ошибками HC3 и проверкой пре-трендов
    """

    @staticmethod
    def multivariate_debiasing_ate(
        df: pd.DataFrame,
        outcome_col: str,
        feature_cols: List[str],
        variant_col: str = "variant",
        alpha: float = 0.05,
    ) -> dict:
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
        trim_bounds: tuple[float, float] = (0.05, 0.95),
    ) -> dict:
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
        raw_propensity = 1.0 / (1.0 + np.exp(-logits))

        in_support = (raw_propensity >= trim_bounds[0]) & (raw_propensity <= trim_bounds[1])
        overlap_share = float(np.mean(in_support))
        propensity = np.clip(raw_propensity, trim_bounds[0], trim_bounds[1])

        treated_idx = np.where(t == 1)[0]
        control_idx = np.where(t == 0)[0]
        if len(treated_idx) < 2 or len(control_idx) < 2:
            raise ValueError("Для PSM/AIPW необходимо минимум по 2 наблюдения в тестовой и контрольной группах.")

        treated_scores = logits[treated_idx]
        control_scores = logits[control_idx]

        matched_control_outcomes = np.empty(len(treated_idx), dtype=float)
        for i, score_t in enumerate(treated_scores):
            nearest_c_pos = int(np.argmin(np.abs(control_scores - score_t)))
            matched_control_outcomes[i] = y[control_idx[nearest_c_pos]]

        att_psm = float(np.mean(y[treated_idx] - matched_control_outcomes))

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
            "common_support_overlap_share": round(overlap_share, 4),
            "overlap_warning": overlap_share < 0.70,
        }

    @staticmethod
    def difference_in_differences(
        df: pd.DataFrame,
        outcome_col: str,
        is_treated_group_col: str,
        is_post_period_col: str,
        time_step_col: Optional[str] = None,
    ) -> dict:
        g = df[is_treated_group_col].to_numpy(dtype=float)
        post = df[is_post_period_col].to_numpy(dtype=float)
        inter = g * post
        y = df[outcome_col].to_numpy(dtype=float)

        X = np.column_stack([np.ones(len(df)), g, post, inter])
        n, k = X.shape
        if n <= k:
            raise ValueError("Недостаточно наблюдений для оценки модели DiD.")

        xtx_inv = np.linalg.inv(X.T @ X)
        beta = xtx_inv @ (X.T @ y)
        residuals = y - X @ beta

        hat_diag = np.einsum("ij,jk,ik->i", X, xtx_inv, X)
        hc3_weights = (residuals / np.maximum(1.0 - hat_diag, 1e-6)) ** 2
        meat = (X * hc3_weights[:, None]).T @ X
        cov_hc3 = xtx_inv @ meat @ xtx_inv

        se_did = float(math.sqrt(max(cov_hc3[3, 3], 1e-15)))
        did_effect = float(beta[3])
        t_stat = did_effect / se_did
        p_val = float(2.0 * (1.0 - stats.t.cdf(abs(t_stat), df=n - k)))

        parallel_trends_pval = None
        parallel_trends_passed = True
        if time_step_col and time_step_col in df.columns:
            pre_df = df[df[is_post_period_col] == 0]
            if len(pre_df) > 4:
                g_pre = pre_df[is_treated_group_col].to_numpy(dtype=float)
                t_pre = pre_df[time_step_col].to_numpy(dtype=float)
                y_pre = pre_df[outcome_col].to_numpy(dtype=float)
                X_pre = np.column_stack([np.ones(len(pre_df)), g_pre, t_pre, g_pre * t_pre])
                b_pre, _, _, _ = np.linalg.lstsq(X_pre, y_pre, rcond=None)
                res_pre = y_pre - X_pre @ b_pre
                s2_pre = float(np.sum(res_pre ** 2) / max(len(pre_df) - 4, 1))
                cov_pre = s2_pre * np.linalg.inv(X_pre.T @ X_pre + 1e-8 * np.eye(4))
                se_slope_diff = math.sqrt(max(cov_pre[3, 3], 1e-15))
                t_pre_stat = float(b_pre[3]) / se_slope_diff
                parallel_trends_pval = float(2.0 * (1.0 - stats.t.cdf(abs(t_pre_stat), df=max(len(pre_df) - 4, 1))))
                parallel_trends_passed = bool(parallel_trends_pval > 0.05)

        return {
            "did_treatment_effect": round(did_effect, 4),
            "standard_error": round(se_did, 4),
            "se_type": "HC3_robust",
            "t_statistic": round(t_stat, 4),
            "p_value": p_val,
            "is_significant": bool(p_val < 0.05),
            "parallel_trends_pvalue": parallel_trends_pval,
            "parallel_trends_passed": parallel_trends_passed,
        }
