from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from scipy import optimize, stats


@dataclass
class StorePairMatch:
    treatment_store: str
    control_store: str
    pre_correlation: float
    scale_diff_pct: float


@dataclass
class SyntheticControlResult:
    ate_absolute: float
    ate_relative_pct: float
    pre_intervention_rmse: float
    donor_weights: Dict[str, float]
    p_value_placebo_or_ttest: float
    is_significant: bool


class OfflineRetailGeoEngine:
    """
    Модуль Оффлайн-экспериментов, Гео-тестов и Синтетического Контроля (ABI TestOps methodology):
    - Жадное спаривание магазинов/регионов (Matched Markets) по корреляции и базовому объему продаж
    - Построение Синтетического Контроля (Synthetic Control) из пула донорских точек
    """

    @staticmethod
    def match_stores_greedy(
        pre_panel_df: pd.DataFrame,
        unit_col: str = "store_id",
        time_col: str = "date",
        metric_col: str = "revenue",
        n_pairs: int = 5,
    ) -> List[StorePairMatch]:
        """
        Подбирает пары наиболее похожих магазинов/регионов на исторических данных до старта теста:
        максимизирует корреляцию временных рядов и минимизирует разницу в среднем объеме выручки.
        """
        pivot = pre_panel_df.pivot(index=time_col, columns=unit_col, values=metric_col).dropna(axis=1)
        stores = list(pivot.columns)
        if len(stores) < 2 * n_pairs:
            raise ValueError(f"Недостаточно магазинов ({len(stores)}) для формирования {n_pairs} пар.")

        corr_matrix = pivot.corr().to_numpy(dtype=float)
        means = pivot.mean(axis=0).to_numpy(dtype=float)

        used = set()
        matches: List[StorePairMatch] = []

        # Считаем композитный скор близости для всех пар
        candidates: List[Tuple[float, int, int, float, float]] = []
        for i in range(len(stores)):
            for j in range(i + 1, len(stores)):
                corr_ij = float(corr_matrix[i, j])
                scale_diff = abs(means[i] - means[j]) / max(means[i], means[j], 1e-9)
                score = corr_ij - 0.5 * scale_diff
                candidates.append((score, i, j, corr_ij, scale_diff))

        candidates.sort(key=lambda x: x[0], reverse=True)

        for _, i, j, corr_ij, scale_diff in candidates:
            if len(matches) >= n_pairs:
                break
            if i not in used and j not in used:
                used.add(i)
                used.add(j)
                matches.append(
                    StorePairMatch(
                        treatment_store=str(stores[i]),
                        control_store=str(stores[j]),
                        pre_correlation=round(corr_ij, 4),
                        scale_diff_pct=round(scale_diff * 100.0, 2),
                    )
                )

        return matches

    @staticmethod
    def estimate_synthetic_control(
        panel_df: pd.DataFrame,
        unit_col: str,
        time_col: str,
        metric_col: str,
        treated_units: List[str],
        intervention_start_time: int,
        cooldown_periods_to_drop: int = 0,
    ) -> SyntheticControlResult:
        """
        Метод Синтетического Контроля (Synthetic Control Method):
        1) На пред-экспериментальном периоде (time < intervention_start_time) находит неотрицательные веса w_j
           контрольных магазинов (sum(w_j) = 1), минимизирующие RMSE с тестовой группой.
        2) Строит контрфактическую траекторию синтетического контроля после интервенции и считает эффект.
        """
        pivot = panel_df.pivot(index=time_col, columns=unit_col, values=metric_col).sort_index()
        donor_cols = [c for c in pivot.columns if c not in treated_units]
        if not donor_cols:
            raise ValueError("Пул контрольных магазинов-доноров пуст.")

        y_treat_series = pivot[treated_units].mean(axis=1)
        X_donors = pivot[donor_cols]

        pre_mask = pivot.index < intervention_start_time
        post_mask = pivot.index >= intervention_start_time

        y_pre = y_treat_series.loc[pre_mask].to_numpy(dtype=float)
        X_pre = X_donors.loc[pre_mask].to_numpy(dtype=float)

        y_post = y_treat_series.loc[post_mask].to_numpy(dtype=float)
        X_post = X_donors.loc[post_mask].to_numpy(dtype=float)

        if cooldown_periods_to_drop > 0 and len(y_post) > cooldown_periods_to_drop:
            y_post = y_post[:-cooldown_periods_to_drop]
            X_post = X_post[:-cooldown_periods_to_drop]

        n_donors = X_pre.shape[1]
        init_w = np.ones(n_donors) / n_donors
        bounds = [(0.0, 1.0) for _ in range(n_donors)]
        constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}

        def loss_func(w: np.ndarray) -> float:
            return float(np.mean((y_pre - X_pre @ w) ** 2))

        res = optimize.minimize(
            loss_func, init_w, method="SLSQP", bounds=bounds, constraints=constraints
        )
        w_opt = res.x / np.sum(res.x)

        synth_pre = X_pre @ w_opt
        synth_post = X_post @ w_opt

        pre_rmse = float(np.sqrt(np.mean((y_pre - synth_pre) ** 2)))
        post_gaps = y_post - synth_post
        ate_abs = float(np.mean(post_gaps))
        baseline_post = float(np.mean(synth_post))
        ate_rel_pct = (ate_abs / baseline_post) * 100.0 if baseline_post != 0 else 0.0

        t_stat, p_val = stats.ttest_1samp(post_gaps, popmean=0.0)
        weights_dict = {str(col): round(float(w), 4) for col, w in zip(donor_cols, w_opt)}

        return SyntheticControlResult(
            ate_absolute=round(ate_abs, 4),
            ate_relative_pct=round(ate_rel_pct, 3),
            pre_intervention_rmse=round(pre_rmse, 4),
            donor_weights=weights_dict,
            p_value_placebo_or_ttest=float(p_val),
            is_significant=bool(p_val < 0.05),
        )
