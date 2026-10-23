import math
from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class SwitchbackAnalysisResult:
    ate_unadjusted: float
    ate_power_cupac: float
    se_unadjusted: float
    se_power_cupac: float
    variance_reduction_pct: float
    p_value_power_cupac: float
    ci_lower: float
    ci_upper: float
    theta_design_weighted: float
    lambda_macro_weight: float
    is_significant: bool


class SwitchbackAndClusterEngine:
    """
    Движок дизайна и анализа Switchback и кластерных экспериментов (DoorDash methodology):
    - Декомпозиция дисперсии на внутриячеечную (S_res) и макро-компоненты (S_cl, S_time, S_int)
    - Оценка дисперсии при стратификации и кросс-секционном спаривании (Pairing & Mirroring)
    - Power-Optimal CUPAC: закрытая Ridge-регрессия и взвешенный коэффициент theta* под лосс L_power
    """

    @staticmethod
    def compare_design_variances(
        total_variance: float,
        num_clusters_j: int,
        num_windows_h: int,
        mean_cell_size_n: float,
        cell_size_cv: float,
        s_cluster: float = 0.10,
        s_time: float = 0.05,
        s_interaction: float = 0.05,
    ) -> Dict[str, float]:
        """
        Сравнивает асимптотическую дисперсию оценки эффекта в трех дизайнах Switchback:
        1) Наивная рандомизация ячеек (Unstratified)
        2) Стратифицированный дизайн по кластерам (Stratified: убирает S_cl * (1 + cv^2))
        3) Спаривание и зеркалирование кластеров (Pairing & Mirroring: защищает и S_cl, и S_time)
        """
        s_res = max(0.0, 1.0 - (s_cluster + s_time + s_interaction))
        s_macro = s_cluster + s_time + s_interaction
        jh = float(num_clusters_j * num_windows_h)
        cv_penalty = 1.0 + (cell_size_cv ** 2)
        base_scale = (4.0 * total_variance) / max(jh, 1.0)

        var_unstratified = base_scale * (
            (s_res / mean_cell_size_n) + s_macro * ((1.0 / mean_cell_size_n) + cv_penalty)
        )
        var_stratified = base_scale * (
            (1.0 / mean_cell_size_n) + (s_time + s_interaction) * cv_penalty
        )
        var_paired_mirrored = base_scale * (
            (1.0 / mean_cell_size_n) + s_interaction * cv_penalty
        )

        return {
            "var_unstratified": float(var_unstratified),
            "var_stratified": float(var_stratified),
            "var_paired_mirrored": float(var_paired_mirrored),
            "stratification_variance_reduction_pct": round(
                (1.0 - var_stratified / var_unstratified) * 100.0, 2
            ),
            "pairing_variance_reduction_pct": round(
                (1.0 - var_paired_mirrored / var_unstratified) * 100.0, 2
            ),
        }

    @staticmethod
    def generate_paired_mirror_schedule(
        cluster_ids: List[str], num_windows_h: int, random_seed: int = 42
    ) -> pd.DataFrame:
        """
        Генерирует расписание Switchback с зеркальным спариванием (Pairing & Mirroring):
        В каждый момент времени h половине пар назначается противоположный вариант (A vs B),
        что балансирует размеры выборок внутри каждого временного окна.
        """
        if len(cluster_ids) < 2:
            raise ValueError("Для спаренного Switchback необходимо минимум 2 кластера.")

        rng = np.random.default_rng(random_seed)
        records = []
        pairs = [cluster_ids[i : i + 2] for i in range(0, len(cluster_ids), 2)]

        for h in range(num_windows_h):
            for pair in pairs:
                flip = int(rng.integers(0, 2))
                v0 = "treatment" if flip == 1 else "control"
                v1 = "control" if flip == 1 else "treatment"
                records.append({"cluster_id": pair[0], "time_window": h, "variant": v0})
                if len(pair) > 1:
                    records.append({"cluster_id": pair[1], "time_window": h, "variant": v1})

        return pd.DataFrame(records)

    @staticmethod
    def analyze_power_optimal_cupac(
        df: pd.DataFrame,
        cell_col: str,
        variant_col: str,
        outcome_col: str,
        feature_cols: List[str],
        ridge_l2: float = 1.0,
        alpha: float = 0.05,
    ) -> SwitchbackAnalysisResult:
        """
        Реализует Power-Optimal CUPAC для Switchback-экспериментов:
        1) Декомпозирует наблюдения на внутриячеечные отклонения и ячеечные средние
        2) Вычисляет макро-вес lambda = n_bar * (1 + cv^2)
        3) Обучает композитную Ridge-модель под лосс L_power (взвешивая макро-ошибку весом lambda)
        4) Оценивает оптимальный дизайн-взвешенный коэффициент theta* и считает сжатие дисперсии
        """
        data = df.copy()
        cell_groups = data.groupby(cell_col)
        cell_sizes = cell_groups[outcome_col].count().to_numpy(dtype=float)

        n_bar = float(np.mean(cell_sizes))
        std_n = float(np.std(cell_sizes, ddof=1)) if len(cell_sizes) > 1 else 0.0
        cv = std_n / max(n_bar, 1e-12)
        lambda_macro = n_bar * (1.0 + cv ** 2)

        X = data[feature_cols].to_numpy(dtype=float)
        y = data[outcome_col].to_numpy(dtype=float)

        # Вычисляем ячеечные средние для композитной функции потерь L_power
        cell_means_y = data.groupby(cell_col)[outcome_col].transform("mean").to_numpy(dtype=float)
        cell_means_X = data.groupby(cell_col)[feature_cols].transform("mean").to_numpy(dtype=float)

        X_within = X - cell_means_X
        y_within = y - cell_means_y

        # Закрытая форма Ridge под L_power: комбинируем within-ковариацию и взвешенную на lambda macro-ковариацию
        d = X.shape[1]
        gram_within = X_within.T @ X_within
        gram_macro = cell_means_X.T @ cell_means_X
        rhs_within = X_within.T @ y_within
        rhs_macro = cell_means_X.T @ cell_means_y

        A_mat = gram_within + lambda_macro * gram_macro + ridge_l2 * np.eye(d)
        b_vec = rhs_within + lambda_macro * rhs_macro
        weights = np.linalg.solve(A_mat, b_vec)

        data["g_pred"] = X @ weights

        # Агрегируем на уровень ячеек рандомизации (cell level) для корректного расчета ошибки и theta*
        cell_df = (
            data.groupby([cell_col, variant_col], as_index=False)
            .agg(
                cell_y=(outcome_col, "mean"),
                cell_g=("g_pred", "mean"),
                cell_n=(outcome_col, "count"),
            )
        )

        y_cells = cell_df["cell_y"].to_numpy(dtype=float)
        g_cells = cell_df["cell_g"].to_numpy(dtype=float)
        n_cells = cell_df["cell_n"].to_numpy(dtype=float)

        # Взвешиваем оценку theta* с учетом размеров ячеек
        g_centered = g_cells - float(np.average(g_cells, weights=n_cells))
        y_centered = y_cells - float(np.average(y_cells, weights=n_cells))
        denom = float(np.sum((n_cells ** 2) * (g_centered ** 2)))
        theta_star = float(np.sum((n_cells ** 2) * g_centered * y_centered) / denom) if denom > 1e-12 else 0.0

        cell_df["cell_y_adj"] = y_cells - theta_star * g_centered

        c_mask = cell_df[variant_col] == "control"
        t_mask = cell_df[variant_col] == "treatment"

        c_raw = cell_df.loc[c_mask, "cell_y"].to_numpy(dtype=float)
        t_raw = cell_df.loc[t_mask, "cell_y"].to_numpy(dtype=float)
        c_adj = cell_df.loc[c_mask, "cell_y_adj"].to_numpy(dtype=float)
        t_adj = cell_df.loc[t_mask, "cell_y_adj"].to_numpy(dtype=float)

        ate_raw = float(np.mean(t_raw) - np.mean(c_raw))
        ate_adj = float(np.mean(t_adj) - np.mean(c_adj))

        se_raw = math.sqrt(float(np.var(t_raw, ddof=1) / len(t_raw) + np.var(c_raw, ddof=1) / len(c_raw)))
        se_adj = math.sqrt(float(np.var(t_adj, ddof=1) / len(t_adj) + np.var(c_adj, ddof=1) / len(c_adj)))

        var_red_pct = (1.0 - (se_adj ** 2) / max(se_raw ** 2, 1e-15)) * 100.0
        t_stat, p_val = stats.ttest_ind(t_adj, c_adj, equal_var=False)
        z_crit = float(stats.norm.ppf(1.0 - alpha / 2.0))

        return SwitchbackAnalysisResult(
            ate_unadjusted=ate_raw,
            ate_power_cupac=ate_adj,
            se_unadjusted=se_raw,
            se_power_cupac=se_adj,
            variance_reduction_pct=round(var_red_pct, 2),
            p_value_power_cupac=float(p_val),
            ci_lower=ate_adj - z_crit * se_adj,
            ci_upper=ate_adj + z_crit * se_adj,
            theta_design_weighted=round(theta_star, 5),
            lambda_macro_weight=round(lambda_macro, 2),
            is_significant=bool(p_val < alpha),
        )
