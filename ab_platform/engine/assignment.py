import hashlib
from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np
from scipy import stats


@dataclass
class AssignmentResult:
    user_id: str
    experiment_id: str
    is_in_global_holdout: bool
    is_exposed: bool
    variant_id: str
    bucket_z: int


class DeterministicAssignmentEngine:
    """
    Высокоскоростной детерминированный движок сплитования (по методологии Wish / Google / eBay):
    - Использует 10 000 бакетов Z in [0, 9999] из 64-битного хэша MD5(experiment_id + user_id)
    - Обеспечивает монотонный рампап (R_e = Z // 100) и независимый сплит вариантов (R_b = Z % 100)
    - Поддерживает глобальный Holdout-слой (1-5% трафика) и слои взаимного исключения (Layers)
    """

    MAX_UINT64 = float(0xFFFFFFFFFFFFFFFF)

    @classmethod
    def _hash_to_10000_buckets(cls, salt_key: str, user_id: str) -> int:
        concat_str = f"{salt_key}:{user_id}".encode("utf-8")
        digest = hashlib.md5(concat_str).digest()
        # Берем первые 8 байт (64 бита) для равномерного отображения в [0, 9999]
        hash_64 = int.from_bytes(digest[:8], byteorder="big", signed=False)
        z = int((hash_64 * 10000) // (0xFFFFFFFFFFFFFFFF + 1))
        return z

    @classmethod
    def assign_user(
        cls,
        user_id: str,
        experiment_id: str,
        exposure_rate_pct: float = 100.0,
        control_split_pct: float = 50.0,
        global_holdout_pct: float = 0.0,
        holdout_salt: str = "global_holdout_v1",
        shared_hash_id: Optional[str] = None,
    ) -> AssignmentResult:
        """
        Назначает пользователя в вариант эксперимента:
        1) Проверяет попадание в глобальный Holdout
        2) Проверяет попадание в рампап эксперимента (R_e = Z // 100 < exposure_rate_pct)
        3) Независимо распределяет в control или treatment (R_b = Z % 100 < control_split_pct)
        """
        if global_holdout_pct > 0.0:
            z_holdout = cls._hash_to_10000_buckets(holdout_salt, user_id)
            if (z_holdout / 100.0) < global_holdout_pct:
                return AssignmentResult(
                    user_id=user_id,
                    experiment_id=experiment_id,
                    is_in_global_holdout=True,
                    is_exposed=False,
                    variant_id="holdout",
                    bucket_z=z_holdout,
                )

        effective_key = shared_hash_id if shared_hash_id else experiment_id
        z = cls._hash_to_10000_buckets(effective_key, user_id)

        r_exposure = z // 100  # Диапазон [0..99] для монотонного рампапа
        r_bucket = z % 100     # Независимый диапазон [0..99] для выбора варианта

        if r_exposure >= exposure_rate_pct:
            return AssignmentResult(
                user_id=user_id,
                experiment_id=experiment_id,
                is_in_global_holdout=False,
                is_exposed=False,
                variant_id="ignore",
                bucket_z=z,
            )

        variant = "control" if r_bucket < control_split_pct else "treatment"
        return AssignmentResult(
            user_id=user_id,
            experiment_id=experiment_id,
            is_in_global_holdout=False,
            is_exposed=True,
            variant_id=variant,
            bucket_z=z,
        )

    @classmethod
    def validate_orthogonality_and_uniformity(
        cls, exp_id_1: str, exp_id_2: str, n_sim_users: int = 20000
    ) -> dict:
        """
        Предпусковая проверка сплитования на синтетических юзерах:
        1) Равномерность бакетов внутри эксперимента
        2) Статистическая независимость (ортогональность) между двумя экспериментами (Chi2 test)
        """
        contingency = np.zeros((2, 2), dtype=int)
        for i in range(n_sim_users):
            uid = f"sim_user_{i}"
            res1 = cls.assign_user(uid, exp_id_1, exposure_rate_pct=100.0, control_split_pct=50.0)
            res2 = cls.assign_user(uid, exp_id_2, exposure_rate_pct=100.0, control_split_pct=50.0)
            row = 0 if res1.variant_id == "control" else 1
            col = 0 if res2.variant_id == "control" else 1
            contingency[row, col] += 1

        chi2_stat, p_val_indep, _, _ = stats.chi2_contingency(contingency)
        control_share_1 = float(np.sum(contingency[0, :]) / n_sim_users)

        return {
            "n_simulated": n_sim_users,
            "exp1_control_share": control_share_1,
            "independence_chi2_pvalue": float(p_val_indep),
            "is_orthogonal_and_valid": bool(p_val_indep > 0.01 and 0.48 <= control_share_1 <= 0.52),
        }
