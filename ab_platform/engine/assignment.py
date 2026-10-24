import hashlib
from dataclasses import dataclass
from typing import Optional, Sequence
import numpy as np
from scipy import stats
from ab_platform.schemas.passport import VariantAllocation


@dataclass
class AssignmentResult:
    user_id: str
    experiment_id: str
    layer_id: str
    is_in_global_holdout: bool
    is_exposed: bool
    variant_id: str
    bucket_z: int


class DeterministicAssignmentEngine:
    """
    Детерминированный движок сплитования (10 000 бакетов):
    - Строгая валидация процентов раскатки, сплита и слотов слоя
    - Изоляция экспериментов внутри слоя (Layer / Plane Bucket Ranges)
    - Поддержка как бинарного (A/B), так и многорукого (Multi-Arm A/B/C/n) распределения
    """

    @classmethod
    def _hash_to_10000_buckets(cls, salt_key: str, user_id: str) -> int:
        if not user_id or not isinstance(user_id, str):
            raise ValueError("user_id должен быть непустой строкой.")
        if not salt_key or not isinstance(salt_key, str):
            raise ValueError("salt_key должен быть непустой строкой.")
        concat_str = f"{salt_key}:{user_id}".encode("utf-8")
        digest = hashlib.md5(concat_str).digest()
        hash_64 = int.from_bytes(digest[:8], byteorder="big", signed=False)
        return int((hash_64 * 10000) // (0xFFFFFFFFFFFFFFFF + 1))

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
        layer_id: str = "default_layer",
        layer_slot_start: int = 0,
        layer_slot_end: int = 10000,
        variants: Optional[Sequence[VariantAllocation]] = None,
    ) -> AssignmentResult:
        if not (0.0 <= exposure_rate_pct <= 100.0):
            raise ValueError(f"exposure_rate_pct должен быть в [0, 100], получено: {exposure_rate_pct}")
        if not (0.0 < control_split_pct < 100.0) and variants is None:
            raise ValueError(f"control_split_pct должен быть в (0, 100), получено: {control_split_pct}")
        if not (0.0 <= global_holdout_pct < 100.0):
            raise ValueError(f"global_holdout_pct должен быть в [0, 100), получено: {global_holdout_pct}")
        if not (0 <= layer_slot_start < layer_slot_end <= 10000):
            raise ValueError(f"Некорректный диапазон слотов слоя: [{layer_slot_start}, {layer_slot_end}).")

        if global_holdout_pct > 0.0:
            z_holdout = cls._hash_to_10000_buckets(holdout_salt, user_id)
            if (z_holdout / 100.0) < global_holdout_pct:
                return AssignmentResult(
                    user_id=user_id,
                    experiment_id=experiment_id,
                    layer_id=layer_id,
                    is_in_global_holdout=True,
                    is_exposed=False,
                    variant_id="holdout",
                    bucket_z=z_holdout,
                )

        if layer_slot_start > 0 or layer_slot_end < 10000:
            z_layer = cls._hash_to_10000_buckets(f"layer:{layer_id}", user_id)
            if not (layer_slot_start <= z_layer < layer_slot_end):
                return AssignmentResult(
                    user_id=user_id,
                    experiment_id=experiment_id,
                    layer_id=layer_id,
                    is_in_global_holdout=False,
                    is_exposed=False,
                    variant_id="out_of_layer_slot",
                    bucket_z=z_layer,
                )

        effective_key = shared_hash_id if shared_hash_id else experiment_id
        z = cls._hash_to_10000_buckets(effective_key, user_id)

        r_exposure = z // 100
        r_bucket = z % 100

        if r_exposure >= exposure_rate_pct:
            return AssignmentResult(
                user_id=user_id,
                experiment_id=experiment_id,
                layer_id=layer_id,
                is_in_global_holdout=False,
                is_exposed=False,
                variant_id="ignore",
                bucket_z=z,
            )

        if variants is not None:
            total_pct = sum(v.traffic_percentage for v in variants)
            if abs(total_pct - 100.0) > 1e-4:
                raise ValueError(f"Сумма долей вариантов должна быть 100%, получено: {total_pct}")
            cumulative = 0.0
            chosen_variant = variants[-1].variant_id
            for v in variants:
                cumulative += v.traffic_percentage
                if r_bucket < cumulative:
                    chosen_variant = v.variant_id
                    break
        else:
            chosen_variant = "control" if r_bucket < control_split_pct else "treatment"

        return AssignmentResult(
            user_id=user_id,
            experiment_id=experiment_id,
            layer_id=layer_id,
            is_in_global_holdout=False,
            is_exposed=True,
            variant_id=chosen_variant,
            bucket_z=z,
        )

    @classmethod
    def validate_orthogonality_and_uniformity(
        cls, exp_id_1: str, exp_id_2: str, n_sim_users: int = 20000
    ) -> dict:
        if n_sim_users < 100:
            raise ValueError("n_sim_users должен быть >= 100.")
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
