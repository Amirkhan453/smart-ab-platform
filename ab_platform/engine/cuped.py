from typing import Optional
import pandas as pd
import numpy as np


class VarianceReductionEngine:
    """
    Модуль снижения дисперсии:
    - Классический CUPED с порогом r^2 >= min_r2_threshold (по умолчанию 0.05) и импутацией NaN
    - Метод Лина (раздельные регрессии для контроля и теста)
    - Расчет Power-Optimal веса lambda для кластерных и Switchback экспериментов
    """

    @staticmethod
    def apply_cuped(
        df: pd.DataFrame,
        metric_col: str,
        covariate_col: str,
        min_r2_threshold: float = 0.05,
        min_corr_threshold: Optional[float] = None,
        winsorize_quantile: Optional[float] = 0.995,
    ) -> tuple[pd.DataFrame, dict]:
        result_df = df.copy()
        y_raw = result_df[metric_col].to_numpy(dtype=float)
        x_raw = result_df[covariate_col].to_numpy(dtype=float)

        # Импутация пропусков в ковариате средним по валидным наблюдениям
        nan_mask = np.isnan(x_raw)
        if np.all(nan_mask):
            cuped_col = f"{metric_col}_cuped"
            result_df[cuped_col] = y_raw
            return result_df, {
                "applied": False,
                "theta": 0.0,
                "correlation": 0.0,
                "r_squared": 0.0,
                "variance_reduction_pct": 0.0,
                "sensitivity_gain_factor": 1.0,
            }
        if np.any(nan_mask):
            valid_mean = float(np.nanmean(x_raw))
            x_raw = np.where(nan_mask, valid_mean, x_raw)

        if winsorize_quantile is not None and 0.5 < winsorize_quantile < 1.0:
            y_cap = float(np.quantile(y_raw, winsorize_quantile))
            x_cap = float(np.quantile(x_raw, winsorize_quantile))
            y = np.clip(y_raw, np.min(y_raw), y_cap)
            x = np.clip(x_raw, np.min(x_raw), x_cap)
        else:
            y, x = y_raw, x_raw

        var_x = float(np.var(x, ddof=1))
        var_y = float(np.var(y, ddof=1))
        cov_yx = float(np.cov(y, x, ddof=1)[0, 1])
        corr = float(np.corrcoef(y, x)[0, 1]) if (var_x > 1e-12 and var_y > 1e-12) else 0.0
        r_sq = corr ** 2

        effective_r2_thresh = (min_corr_threshold ** 2) if min_corr_threshold is not None else min_r2_threshold
        cuped_col = f"{metric_col}_cuped"

        if r_sq < effective_r2_thresh or var_x <= 1e-12:
            result_df[cuped_col] = y
            return result_df, {
                "applied": False,
                "theta": 0.0,
                "correlation": corr,
                "r_squared": r_sq,
                "variance_reduction_pct": 0.0,
                "sensitivity_gain_factor": 1.0,
            }

        theta = cov_yx / var_x
        mean_x = float(np.mean(x))
        y_cuped = y - theta * (x - mean_x)
        result_df[cuped_col] = y_cuped

        var_cuped = float(np.var(y_cuped, ddof=1))
        variance_reduction = (1.0 - var_cuped / var_y) * 100.0 if var_y > 0 else 0.0
        sensitivity_gain = float(np.sqrt(var_y / max(var_cuped, 1e-12)))

        diagnostics = {
            "applied": True,
            "theta": float(theta),
            "correlation": corr,
            "r_squared": r_sq,
            "variance_reduction_pct": float(variance_reduction),
            "sensitivity_gain_factor": sensitivity_gain,
        }
        return result_df, diagnostics

    @staticmethod
    def apply_lins_method(
        df: pd.DataFrame,
        metric_col: str,
        covariate_col: str,
        variant_col: str = "variant",
        winsorize_quantile: Optional[float] = 0.995,
    ) -> pd.DataFrame:
        result_df = df.copy()
        if result_df[covariate_col].isna().any():
            result_df[covariate_col] = result_df[covariate_col].fillna(result_df[covariate_col].mean())

        if winsorize_quantile is not None and 0.5 < winsorize_quantile < 1.0:
            y_cap = float(result_df[metric_col].quantile(winsorize_quantile))
            x_cap = float(result_df[covariate_col].quantile(winsorize_quantile))
            result_df[metric_col] = np.clip(result_df[metric_col], None, y_cap)
            result_df[covariate_col] = np.clip(result_df[covariate_col], None, x_cap)

        cuped_col = f"{metric_col}_lin_cuped"
        result_df[cuped_col] = result_df[metric_col].astype(float)
        mean_x = float(np.mean(result_df[covariate_col]))

        for variant in result_df[variant_col].unique():
            mask = result_df[variant_col] == variant
            y_v = result_df.loc[mask, metric_col].to_numpy(dtype=float)
            x_v = result_df.loc[mask, covariate_col].to_numpy(dtype=float)

            var_x = float(np.var(x_v, ddof=1))
            if var_x > 1e-12 and len(y_v) > 2:
                cov_yx = float(np.cov(y_v, x_v, ddof=1)[0, 1])
                theta_v = cov_yx / var_x
            else:
                theta_v = 0.0

            result_df.loc[mask, cuped_col] = y_v - theta_v * (x_v - mean_x)

        return result_df

    @staticmethod
    def compute_power_optimal_cell_weight(cell_sizes: np.ndarray) -> dict:
        sizes = np.asarray(cell_sizes, dtype=float)
        if len(sizes) == 0 or np.mean(sizes) <= 0:
            raise ValueError("Массив размеров ячеек должен содержать положительные значения.")

        n_bar = float(np.mean(sizes))
        std_n = float(np.std(sizes, ddof=1)) if len(sizes) > 1 else 0.0
        cv = std_n / n_bar
        lambda_weight = n_bar * (1.0 + cv ** 2)

        return {
            "mean_cell_size": n_bar,
            "cell_size_cv": cv,
            "lambda_macro_weight": float(lambda_weight),
        }
