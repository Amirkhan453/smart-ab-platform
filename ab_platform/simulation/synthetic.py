import numpy as np
import pandas as pd


class SyntheticDataGenerator:
    """
    Генератор синтетических данных для тестирования честности математического ядра:
    - Генерирует пред-экспериментальную ковариату X_pre с заданной корреляцией для проверки CUPED
    - Добавляет тяжелые хвосты (лог-нормальные выбросы / 'киты') для проверки винзоризации
    - Инжектирует заданный процент Double Exposure (баг сплитования) для проверки гигиены данных
    - Генерирует числитель и знаменатель для Ratio-метрик с внутрипользовательской корреляцией
    """

    @staticmethod
    def generate_ab_dataset(
        n_users: int = 20000,
        treatment_ratio: float = 0.5,
        true_lift_rel: float = 0.0,
        pre_post_corr: float = 0.65,
        outlier_fraction: float = 0.002,
        double_exposure_fraction: float = 0.0,
        random_seed: int = 42,
    ) -> pd.DataFrame:
        rng = np.random.default_rng(random_seed)

        user_ids = [f"user_{i:06d}" for i in range(n_users)]
        is_treatment = rng.random(n_users) < treatment_ratio
        variants = np.where(is_treatment, "treatment", "control")

        # Генерируем скоррелированные нормальные латентные факторы (Pre и Post)
        cov_matrix = [[1.0, pre_post_corr], [pre_post_corr, 1.0]]
        z_pre, z_post = rng.multivariate_normal([0.0, 0.0], cov_matrix, size=n_users).T

        # Переводим в реалистичную метрику выручки (базовое среднее ~ 1000 руб, std ~ 300)
        pre_revenue = np.maximum(0.0, 1000.0 + 300.0 * z_pre)
        base_post_revenue = np.maximum(0.0, 1000.0 + 300.0 * z_post)

        # Накладываем истинный продуктовый эффект (ATE) на тестовую группу
        treatment_multiplier = np.where(is_treatment, 1.0 + true_lift_rel, 1.0)
        post_revenue = base_post_revenue * treatment_multiplier

        # Генерируем сессии (знаменатель для Ratio-метрики) и заказы (числитель)
        sessions = rng.poisson(lam=5.0, size=n_users) + 1
        conversion_prob = np.clip(0.15 * treatment_multiplier + 0.02 * z_pre, 0.01, 0.95)
        orders = rng.binomial(n=sessions, p=conversion_prob)

        # Инжектируем экстремальные выбросы ('китов' / ботов)
        if outlier_fraction > 0:
            n_outliers = int(n_users * outlier_fraction)
            if n_outliers > 0:
                outlier_idx = rng.choice(n_users, size=n_outliers, replace=False)
                post_revenue[outlier_idx] *= rng.uniform(15.0, 40.0, size=n_outliers)

        df = pd.DataFrame({
            "user_id": user_ids,
            "variant": variants,
            "pre_revenue": pre_revenue,
            "post_revenue": post_revenue,
            "sessions": sessions,
            "orders": orders,
            "converted": (orders > 0).astype(int),
        })

        # Инжектируем баг Double Exposure (дублирование юзеров в противоположной группе)
        if double_exposure_fraction > 0:
            n_dups = int(n_users * double_exposure_fraction)
            if n_dups > 0:
                dup_rows = df.sample(n=n_dups, random_state=random_seed).copy()
                dup_rows["variant"] = np.where(
                    dup_rows["variant"] == "control", "treatment", "control"
                )
                df = pd.concat([df, dup_rows], ignore_index=True)

        return df