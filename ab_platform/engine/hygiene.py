from typing import Optional
import numpy as np
import pandas as pd


class DataHygieneProcessor:
    """Очистка данных от багов сплитования (Double Exposure), бот-трафика, выбросов и скошенности."""

    @staticmethod
    def remove_double_exposure(
        df: pd.DataFrame,
        user_col: str = "user_id",
        variant_col: str = "variant",
        return_diagnostics: bool = False,
    ):
        """
        Удаляет юзеров, попавших в несколько вариантов одновременно, и
        возвращает их долю (double_exposure_frac).

        При return_diagnostics=True дополнительно возвращает третьим
        элементом словарь с числом удалённых строк в разбивке по каждому из
        вариантов, в которых встречались double-exposed юзеры
        (removed_by_variant); при False сохраняется прежний 2-элементный
        контракт (clean_df, double_exposure_frac) для обратной совместимости.
        """
        for col in (user_col, variant_col):
            if col not in df.columns:
                raise ValueError(f"Колонка '{col}' отсутствует в датасете для проверки double exposure.")

        variant_counts = df.groupby(user_col)[variant_col].nunique()
        double_exposed_users = variant_counts[variant_counts > 1].index

        total_users = df[user_col].nunique()
        double_exposure_frac = len(double_exposed_users) / total_users if total_users > 0 else 0.0

        is_double_exposed = df[user_col].isin(double_exposed_users)
        clean_df = df[~is_double_exposed].copy()

        if not return_diagnostics:
            return clean_df, float(double_exposure_frac)

        removed_by_variant = {
            str(k): int(v)
            for k, v in df.loc[is_double_exposed, variant_col].value_counts().to_dict().items()
        }
        diagnostics = {
            "n_double_exposed_units": int(len(double_exposed_users)),
            "n_rows_removed": int(is_double_exposed.sum()),
            "removed_by_variant": removed_by_variant,
        }
        return clean_df, float(double_exposure_frac), diagnostics

    @staticmethod
    def apply_winsorization(
        df: pd.DataFrame,
        metric_col: str,
        lower_quantile: float = 0.0,
        upper_quantile: float = 0.999,
        variant_col: Optional[str] = "variant",
        return_diagnostics: bool = False,
    ):
        """
        Винзоризация (capping) тяжёлых хвостов распределения по заданным квантилям.

        При return_diagnostics=True дополнительно возвращает словарь с числом
        модифицированных (капнутых) значений в разбивке по variant_col
        (winsorized_by_variant); при False возвращает только clean_df, как
        и раньше.
        """
        if metric_col not in df.columns:
            raise ValueError(f"Колонка '{metric_col}' отсутствует в датасете для винзоризации.")

        clean_df = df.copy()
        lower_bound = float(clean_df[metric_col].quantile(lower_quantile))
        upper_bound = float(clean_df[metric_col].quantile(upper_quantile))

        original_vals = clean_df[metric_col].to_numpy(dtype=float)
        clipped_vals = np.clip(original_vals, lower_bound, upper_bound)
        was_modified = clipped_vals != original_vals
        clean_df[metric_col] = clipped_vals

        if not return_diagnostics:
            return clean_df

        winsorized_by_variant = {}
        if variant_col and variant_col in clean_df.columns:
            winsorized_by_variant = {
                str(k): int(v)
                for k, v in clean_df.loc[was_modified, variant_col].value_counts().to_dict().items()
            }
        diagnostics = {
            "lower_bound": lower_bound,
            "upper_bound": upper_bound,
            "n_rows_modified": int(was_modified.sum()),
            "winsorized_by_variant": winsorized_by_variant,
        }
        return clean_df, diagnostics

    @staticmethod
    def filter_bots_variant_blind(
        df: pd.DataFrame,
        user_col: str = "user_id",
        activity_col: str = "event_count",
        variant_col: str = "variant",
        max_threshold: Optional[float] = None,
        quantile_threshold: Optional[float] = None,
    ):
        """
        Variant-blind фильтрация бот-трафика: порог вычисляется по активности
        единицы рандомизации (сумма activity_col), агрегированной по ВСЕМУ
        пулу вне зависимости от variant_col, чтобы не вносить асимметричное
        смещение между вариантами. Передайте ровно один из параметров:

        - max_threshold: абсолютный порог суммарного числа событий на
          единицу рандомизации;
        - quantile_threshold: квантиль в (0, 1), выше которого юзер
          считается ботом (порог вычисляется по агрегированной активности
          всего пула, без учёта variant_col).

        Возвращает (clean_df, diagnostics), где diagnostics содержит
        n_bot_units, n_rows_removed и removed_by_variant (разбивка по
        вариантам только для отчётности, не для расчёта порога).
        """
        for col in (user_col, activity_col):
            if col not in df.columns:
                raise ValueError(f"Колонка '{col}' отсутствует в датасете для фильтрации ботов.")
        if (max_threshold is None) == (quantile_threshold is None):
            raise ValueError("Нужно передать ровно один из параметров: max_threshold или quantile_threshold.")

        per_unit_activity = df.groupby(user_col)[activity_col].sum()

        if quantile_threshold is not None:
            if not (0.0 < quantile_threshold < 1.0):
                raise ValueError("quantile_threshold должен быть в интервале (0, 1).")
            threshold_value = float(per_unit_activity.quantile(quantile_threshold))
            threshold_source = "quantile"
        else:
            if max_threshold <= 0:
                raise ValueError("max_threshold должен быть строго положительным.")
            threshold_value = float(max_threshold)
            threshold_source = "absolute"

        bot_units = per_unit_activity[per_unit_activity > threshold_value].index
        is_bot_row = df[user_col].isin(bot_units)

        removed_by_variant = {}
        if variant_col in df.columns:
            removed_by_variant = {
                str(k): int(v)
                for k, v in df.loc[is_bot_row, variant_col].value_counts().to_dict().items()
            }

        clean_df = df.loc[~is_bot_row].copy()

        diagnostics = {
            "threshold_value": threshold_value,
            "threshold_source": threshold_source,
            "n_bot_units": int(len(bot_units)),
            "n_rows_removed": int(is_bot_row.sum()),
            "removed_by_variant": removed_by_variant,
        }
        return clean_df, diagnostics

    @staticmethod
    def apply_log_transformation(df: pd.DataFrame, metric_col: str):
        """
        Логарифмическая трансформация log1p для сильно скошенных неотрицательных метрик.

        ВНИМАНИЕ (возвращается в diagnostics["warning"]): трансформация меняет
        оцениваемую величину со среднего арифметического на (приближённо)
        геометрическое/лог-среднее. Разница в трансформированной шкале не
        эквивалентна относительному эффекту на исходной метрике и не может
        напрямую использоваться как бизнес-эффект. Предназначена только для
        исследовательской диагностики распределения, а не для primary-вывода.
        """
        if metric_col not in df.columns:
            raise ValueError(f"Колонка '{metric_col}' отсутствует в датасете для лог-трансформации.")
        clean_df = df.copy()
        if (clean_df[metric_col] < 0).any():
            raise ValueError(f"Лог-трансформация невозможна: колонка '{metric_col}' содержит отрицательные значения.")
        transformed_col = f"{metric_col}_log"
        clean_df[transformed_col] = np.log1p(clean_df[metric_col])

        diagnostics = {
            "warning": (
                "Лог-трансформация меняет оцениваемую величину со среднего арифметического на "
                "геометрическое/лог-среднее; использовать только для исследовательской диагностики, "
                "не как основной (primary) критерий принятия решения."
            ),
            "transformed_col": transformed_col,
        }
        return clean_df, diagnostics
