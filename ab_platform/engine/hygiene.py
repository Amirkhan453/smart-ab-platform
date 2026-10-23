import pandas as pd
import numpy as np


class DataHygieneProcessor:
    """Очистка данных от багов сплитования (Double Exposure), выбросов и скошенности."""

    @staticmethod
    def remove_double_exposure(
        df: pd.DataFrame, user_col: str = "user_id", variant_col: str = "variant"
    ) -> tuple[pd.DataFrame, float]:
        """Удаляет юзеров, попавших в несколько вариантов одновременно, и возвращает их долю."""
        variant_counts = df.groupby(user_col)[variant_col].nunique()
        double_exposed_users = variant_counts[variant_counts > 1].index

        total_users = df[user_col].nunique()
        double_exposure_frac = len(double_exposed_users) / total_users if total_users > 0 else 0.0

        clean_df = df[~df[user_col].isin(double_exposed_users)].copy()
        return clean_df, float(double_exposure_frac)

    @staticmethod
    def apply_winsorization(
        df: pd.DataFrame,
        metric_col: str,
        lower_quantile: float = 0.0,
        upper_quantile: float = 0.999,
    ) -> pd.DataFrame:
        """Винзоризация (capping) тяжелых хвостов распределения по заданным квантилям."""
        if metric_col not in df.columns:
            return df

        clean_df = df.copy()
        lower_bound = float(clean_df[metric_col].quantile(lower_quantile))
        upper_bound = float(clean_df[metric_col].quantile(upper_quantile))

        clean_df[metric_col] = np.clip(clean_df[metric_col], lower_bound, upper_bound)
        return clean_df

    @staticmethod
    def apply_log_transformation(df: pd.DataFrame, metric_col: str) -> pd.DataFrame:
        """Логарифмическая трансформация log1p для сильно скошенных неотрицательных метрик."""
        if metric_col not in df.columns:
            return df
        clean_df = df.copy()
        if (clean_df[metric_col] < 0).any():
            raise ValueError(f"Лог-трансформация невозможна: колонка '{metric_col}' содержит отрицательные значения.")
        clean_df[f"{metric_col}_log"] = np.log1p(clean_df[metric_col])
        return clean_df
