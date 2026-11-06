from typing import Optional
import pandas as pd
import numpy as np


class VarianceReductionEngine:
    """
    Модуль снижения дисперсии:
    - Классический CUPED (с опциональной защитной винзоризацией тяжелых хвостов)
    - Метод Лина (раздельные регрессии для контроля и теста)
    - Расчет Power-Optimal веса lambda для кластерных и Switchback экспериментов
    """

    @staticmethod
    def apply_cuped(
        df: pd.DataFrame,
        metric_col: str,
        covariate_col: str,
        min_r2_threshold: float = 0.05,
        winsorize_quantile: Optional[float] = None,
    ) -> tuple[pd.DataFrame, dict]:
        """
        Классический глобальный CUPED с защитой от выбросов, корректным порогом
        применимости по R^2 и обработкой пропусков в ковариате пре-периода.

        Порог применимости: r_squared = corr(y, x)**2 >= min_r2_threshold (по
        умолчанию 0.05, т.е. минимум 5% объясняемой дисперсии), а не
        |corr| >= min_r2_threshold — иначе CUPED ложно применяется при слабой
        связи метрики с ковариатой.

        Пропуски (NaN) в ковариате (например, у пользователей без истории в
        пре-периоде) импутируются средним по валидным наблюдениям пула;
        бинарный индикатор пропуска (is_missing_pre) включается в проекцию
        вместе с ковариатой (двумерная OLS-проекция [x_imputed,
        is_missing_pre]), чтобы не терять пользователей и не смещать theta.

        Винзоризация выключена по умолчанию (winsorize_quantile=None): в
        целевом пайплайне она выполняется централизованно в hygiene.py до
        вызова CUPED. Параметр оставлен для точечного использования вне
        пайплайна.
        """
        result_df = df.copy()
        y_raw = result_df[metric_col].to_numpy(dtype=float)
        x_raw = result_df[covariate_col].to_numpy(dtype=float)

        is_missing_pre = np.isnan(x_raw).astype(float)
        n_missing = int(is_missing_pre.sum())
        valid_mask = ~np.isnan(x_raw)
        if valid_mask.sum() < 2:
            raise ValueError(
                f"Недостаточно наблюдений с валидной ковариатой '{covariate_col}' для расчёта CUPED "
                f"(валидных: {int(valid_mask.sum())}, требуется минимум 2)."
            )
        x_mean_valid = float(np.mean(x_raw[valid_mask]))
        x_imputed = np.where(valid_mask, x_raw, x_mean_valid)

        if winsorize_quantile is not None and 0.5 < winsorize_quantile < 1.0:
            y_cap = float(np.quantile(y_raw, winsorize_quantile))
            x_cap = float(np.quantile(x_imputed, winsorize_quantile))
            y = np.clip(y_raw, np.min(y_raw), y_cap)
            x_imputed = np.clip(x_imputed, np.min(x_imputed), x_cap)
        else:
            y = y_raw

        # Применимость (R^2) считаем по исходной (не импутированной) связи y~x
        # на валидных наблюдениях, чтобы импутация не искажала решение.
        var_x_valid = float(np.var(x_raw[valid_mask], ddof=1))
        var_y_valid = float(np.var(y[valid_mask], ddof=1))
        if var_x_valid > 1e-12 and var_y_valid > 1e-12:
            corr = float(np.corrcoef(y[valid_mask], x_raw[valid_mask])[0, 1])
        else:
            corr = 0.0
        r_squared = corr ** 2

        cuped_col = f"{metric_col}_cuped"

        if r_squared < min_r2_threshold or var_x_valid <= 1e-12:
            result_df[cuped_col] = y
            return result_df, {
                "applied": False,
                "theta": 0.0,
                "theta_missing_indicator": 0.0,
                "correlation": corr,
                "r_squared": r_squared,
                "variance_reduction_pct": 0.0,
                "sensitivity_gain_factor": 1.0,
                "n_missing_covariate": n_missing,
            }

        x_centered = x_imputed - float(np.mean(x_imputed))
        has_missing = n_missing > 0
        if has_missing:
            m_centered = is_missing_pre - float(np.mean(is_missing_pre))
            design = np.column_stack([x_centered, m_centered])
            coef, *_ = np.linalg.lstsq(design, y, rcond=None)
            theta_x, theta_missing = float(coef[0]), float(coef[1])
            y_cuped = y - design @ coef
        else:
            theta_x = float(np.cov(y, x_centered, ddof=1)[0, 1] / np.var(x_centered, ddof=1))
            theta_missing = 0.0
            y_cuped = y - theta_x * x_centered

        result_df[cuped_col] = y_cuped
        result_df["is_missing_pre"] = is_missing_pre

        var_cuped = float(np.var(y_cuped, ddof=1))
        var_y = float(np.var(y, ddof=1))
        variance_reduction = (1.0 - var_cuped / var_y) * 100.0 if var_y > 0 else 0.0
        sensitivity_gain = float(np.sqrt(var_y / max(var_cuped, 1e-12)))

        diagnostics = {
            "applied": True,
            "theta": theta_x,
            "theta_missing_indicator": theta_missing,
            "correlation": corr,
            "r_squared": r_squared,
            "variance_reduction_pct": float(variance_reduction),
            "sensitivity_gain_factor": sensitivity_gain,
            "n_missing_covariate": n_missing,
        }
        return result_df, diagnostics

    @staticmethod
    def apply_lins_method(
        df: pd.DataFrame,
        metric_col: str,
        covariate_col: str,
        variant_col: str = "variant",
        winsorize_quantile: Optional[float] = None,
    ) -> pd.DataFrame:
        """
        Метод Лина: оценка коэффициента наклона раздельно для каждого варианта.

        Пропуски (NaN) в ковариате импутируются средним по валидным
        наблюдениям всего пула; бинарный индикатор пропуска is_missing_pre
        включается в проекцию наравне с ковариатой (двумерная OLS-проекция
        внутри каждого варианта, где есть пропуски).

        Винзоризация выключена по умолчанию: централизованно выполняется в
        hygiene.py перед CUPED/Lin.
        """
        result_df = df.copy()

        x_raw_all = result_df[covariate_col].to_numpy(dtype=float)
        valid_mask_all = ~np.isnan(x_raw_all)
        if valid_mask_all.sum() < 2:
            raise ValueError(
                f"Недостаточно наблюдений с валидной ковариатой '{covariate_col}' для метода Лина."
            )
        x_mean_valid = float(np.mean(x_raw_all[valid_mask_all]))
        x_imputed_all = np.where(valid_mask_all, x_raw_all, x_mean_valid)
        is_missing_all = (~valid_mask_all).astype(float)

        result_df[covariate_col] = x_imputed_all
        result_df["is_missing_pre"] = is_missing_all

        if winsorize_quantile is not None and 0.5 < winsorize_quantile < 1.0:
            y_cap = float(result_df[metric_col].quantile(winsorize_quantile))
            x_cap = float(result_df[covariate_col].quantile(winsorize_quantile))
            result_df[metric_col] = np.clip(result_df[metric_col], None, y_cap)
            result_df[covariate_col] = np.clip(result_df[covariate_col], None, x_cap)

        cuped_col = f"{metric_col}_lin_cuped"
        result_df[cuped_col] = result_df[metric_col].astype(float)
        mean_x = float(np.mean(result_df[covariate_col]))
        mean_missing = float(np.mean(result_df["is_missing_pre"]))

        for variant in result_df[variant_col].unique():
            mask = result_df[variant_col] == variant
            y_v = result_df.loc[mask, metric_col].to_numpy(dtype=float)
            x_v = result_df.loc[mask, covariate_col].to_numpy(dtype=float)
            m_v = result_df.loc[mask, "is_missing_pre"].to_numpy(dtype=float)

            x_v_centered = x_v - mean_x
            has_missing_in_variant = bool(m_v.sum() > 0)

            if len(y_v) > 2 and float(np.var(x_v_centered, ddof=1)) > 1e-12:
                if has_missing_in_variant:
                    m_v_centered = m_v - mean_missing
                    design = np.column_stack([x_v_centered, m_v_centered])
                    coef, *_ = np.linalg.lstsq(design, y_v, rcond=None)
                    adjustment = design @ coef
                else:
                    theta_v = float(np.cov(y_v, x_v_centered, ddof=1)[0, 1] / np.var(x_v_centered, ddof=1))
                    adjustment = theta_v * x_v_centered
            else:
                adjustment = np.zeros_like(y_v)

            result_df.loc[mask, cuped_col] = y_v - adjustment

        return result_df

    @staticmethod
    def compute_power_optimal_cell_weight(cell_sizes: np.ndarray) -> dict:
        """Вычисляет макро-вес lambda = n_bar * (1 + cv^2) для Switchback CUPAC."""
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
