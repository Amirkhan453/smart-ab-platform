"""
Назначение в гостевом домене (ТЗ v4, раздел 4.1; решение 9; K8).

Хэш: MurmurHash3 x86_32, seed 0, беззнаковый; ключ "{salt}:{unit_id}" в UTF-8;
бакет = hash mod 10 000.

Двухшаговая схема:
  0) holdout:  bucket(holdout_salt, id) < доля holdout → гость вне всех тестов;
  1) слой:     b_L = bucket(layer_salt, id); дорожка эксперимента — диапазон [s, e) бакетов слоя;
               таргетинг (ТЗ v4.1): гость дорожки, не прошедший условия, получает контроль с причиной;
               ramp-up — активный поддиапазон [s, s + ramp·(e − s)), он только растёт (sticky);
  2) вариант:  b_V = bucket(experiment_salt, id), варианты нарезаются по 10 000 бакетам.

Вне теста (holdout, не в дорожке, вне ramp) отдаётся контрольный вариант и причина,
служебные метки наружу не выходят.
"""
import struct
from dataclasses import dataclass
from typing import List, Optional, Sequence, Union

import numpy as np
from scipy import stats

from ab_platform.schemas.passport import VariantAllocation

N_BUCKETS = 10000
MASK32 = 0xFFFFFFFF
SMHASHER_VERIFICATION_X86_32 = 0xB0F57EE3

REASON_ASSIGNED = "assigned"
REASON_HOLDOUT = "holdout"
REASON_NOT_IN_LANE = "not_in_lane"
REASON_RAMP = "ramp"


def _rotl32(x: int, r: int) -> int:
    return ((x << r) | (x >> (32 - r))) & MASK32


def murmurhash3_x86_32(data: Union[str, bytes], seed: int = 0) -> int:
    """Канонический MurmurHash3 x86_32 (Appleby, public domain); результат в [0, 2**32). Строка кодируется в UTF-8."""
    if isinstance(data, str):
        data = data.encode("utf-8")
    length = len(data)
    nblocks = length // 4
    h1 = seed & MASK32
    c1, c2 = 0xCC9E2D51, 0x1B873593

    for k1 in struct.unpack_from(f"<{nblocks}I", data):
        k1 = (k1 * c1) & MASK32
        k1 = _rotl32(k1, 15)
        k1 = (k1 * c2) & MASK32
        h1 ^= k1
        h1 = _rotl32(h1, 13)
        h1 = (h1 * 5 + 0xE6546B64) & MASK32

    tail = nblocks * 4
    rest = length & 3
    k1 = 0
    if rest == 3:
        k1 ^= data[tail + 2] << 16
    if rest >= 2:
        k1 ^= data[tail + 1] << 8
    if rest >= 1:
        k1 ^= data[tail]
        k1 = (k1 * c1) & MASK32
        k1 = _rotl32(k1, 15)
        k1 = (k1 * c2) & MASK32
        h1 ^= k1

    h1 ^= length
    h1 ^= h1 >> 16
    h1 = (h1 * 0x85EBCA6B) & MASK32
    h1 ^= h1 >> 13
    h1 = (h1 * 0xC2B2AE35) & MASK32
    h1 ^= h1 >> 16
    return h1


def smhasher_self_test() -> int:
    """Проверочная сумма smhasher для MurmurHash3_x86_32; эталон 0xB0F57EE3 — общий для всех языков."""
    key = bytes(range(256))
    hashes = b"".join(
        murmurhash3_x86_32(key[:i], seed=256 - i).to_bytes(4, "little") for i in range(256)
    )
    return murmurhash3_x86_32(hashes, seed=0)


def bucket(salt: str, unit_id: str) -> int:
    """Бакет [0, 10000) по ТЗ 4.1: murmur3("{salt}:{unit_id}", seed 0) mod 10000."""
    if not unit_id or not isinstance(unit_id, str):
        raise ValueError("unit_id должен быть непустой строкой.")
    if not salt or not isinstance(salt, str):
        raise ValueError("salt должен быть непустой строкой.")
    return murmurhash3_x86_32(f"{salt}:{unit_id}") % N_BUCKETS


def pct_to_buckets(pct: float) -> int:
    """Доля в процентах → число бакетов из 10 000 (разрешение 0,01%)."""
    return int(round(pct * 100))


def allocate_variant_buckets(percentages: Sequence[float]) -> List[int]:
    """Число бакетов на вариант, сумма ровно 10 000 (метод наибольшего остатка)."""
    raw = [p * 100 for p in percentages]
    counts = [int(r) for r in raw]
    order = sorted(range(len(raw)), key=lambda i: raw[i] - counts[i], reverse=True)
    for i in order[: N_BUCKETS - sum(counts)]:
        counts[i] += 1
    return counts


def ramp_range(slot_start: int, slot_end: int, ramp_pct: float) -> tuple:
    """Активный поддиапазон дорожки при раскатке ramp_pct: [s, s + round(ramp·(e − s)))."""
    return slot_start, slot_start + int(round(ramp_pct / 100.0 * (slot_end - slot_start)))


@dataclass
class AssignmentResult:
    user_id: str
    experiment_id: str
    layer_id: str
    is_in_global_holdout: bool
    is_exposed: bool
    variant_id: str
    bucket_z: int
    reason: str = REASON_ASSIGNED


class DeterministicAssignmentEngine:
    """Детерминированное назначение гостей: MurmurHash3, 10 000 бакетов, слой → дорожка → эксперимент → вариант."""

    @classmethod
    def _hash_to_10000_buckets(cls, salt_key: str, user_id: str) -> int:
        return bucket(salt_key, user_id)

    @classmethod
    def buckets_for(cls, salt_key: str, unit_ids) -> np.ndarray:
        """Бакеты варианта (шаг 2) для набора единиц тем же хэшем, что и assign_user — для A/A на боевом пути."""
        return np.fromiter((bucket(salt_key, str(u)) for u in unit_ids), dtype=np.int64)

    @staticmethod
    def control_from_bucket(z, control_split_pct: float):
        """Правило сплита A/B, общее для assign_user и A/A: control, если бакет варианта < доли контроля."""
        return np.asarray(z) < pct_to_buckets(control_split_pct)

    @staticmethod
    def control_variant_id(variants: Optional[Sequence[VariantAllocation]]) -> str:
        if not variants:
            return "control"
        for v in variants:
            if v.is_control:
                return v.variant_id
        return variants[0].variant_id

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
        layer_slot_end: int = N_BUCKETS,
        variants: Optional[Sequence[VariantAllocation]] = None,
        layer_salt: Optional[str] = None,
        ineligible_reason: Optional[str] = None,
        holdout_bucket: Optional[int] = None,
        layer_bucket: Optional[int] = None,
    ) -> AssignmentResult:
        """
        exposure_rate_pct — ramp-up внутри дорожки [layer_slot_start, layer_slot_end);
        соль эксперимента — shared_hash_id или experiment_id; соль слоя — layer_salt или "layer:{layer_id}".
        ineligible_reason — причина, по которой гость не подходит под таргетинг (проверяется после дорожки);
        holdout_bucket и layer_bucket — уже посчитанные бакеты гостя, чтобы не хэшировать повторно.
        """
        if not (0.0 <= exposure_rate_pct <= 100.0):
            raise ValueError(f"exposure_rate_pct должен быть в [0, 100], получено: {exposure_rate_pct}")
        if variants is None and not (0.0 < control_split_pct < 100.0):
            raise ValueError(f"control_split_pct должен быть в (0, 100), получено: {control_split_pct}")
        if not (0.0 <= global_holdout_pct < 100.0):
            raise ValueError(f"global_holdout_pct должен быть в [0, 100), получено: {global_holdout_pct}")
        if not (0 <= layer_slot_start < layer_slot_end <= N_BUCKETS):
            raise ValueError(f"Некорректный диапазон слотов слоя: [{layer_slot_start}, {layer_slot_end}).")
        if variants is not None:
            if len(variants) < 2:
                raise ValueError("Нужно не меньше двух вариантов.")
            total_pct = sum(v.traffic_percentage for v in variants)
            if abs(total_pct - 100.0) > 1e-4:
                raise ValueError(f"Сумма долей вариантов должна быть 100%, получено: {total_pct}")
            if sum(v.is_control for v in variants) > 1:
                raise ValueError("Контрольный вариант должен быть один.")

        control_id = cls.control_variant_id(variants)

        def outside(reason: str, z: int, in_holdout: bool = False) -> AssignmentResult:
            return AssignmentResult(user_id, experiment_id, layer_id, in_holdout, False, control_id, z, reason)

        # Шаг 0: глобальный holdout
        if global_holdout_pct > 0.0:
            z_holdout = bucket(holdout_salt, user_id) if holdout_bucket is None else holdout_bucket
            if z_holdout < pct_to_buckets(global_holdout_pct):
                return outside(REASON_HOLDOUT, z_holdout, in_holdout=True)

        # Шаг 1: слой → дорожка → ramp
        z_layer = bucket(layer_salt or f"layer:{layer_id}", user_id) if layer_bucket is None else layer_bucket
        if not (layer_slot_start <= z_layer < layer_slot_end):
            return outside(REASON_NOT_IN_LANE, z_layer)
        if ineligible_reason is not None:
            return outside(ineligible_reason, z_layer)
        _, ramp_end = ramp_range(layer_slot_start, layer_slot_end, exposure_rate_pct)
        if z_layer >= ramp_end:
            return outside(REASON_RAMP, z_layer)

        # Шаг 2: эксперимент → вариант
        z = bucket(shared_hash_id or experiment_id, user_id)
        if variants is not None:
            cumulative = 0
            chosen = variants[-1].variant_id
            for v, n_b in zip(variants, allocate_variant_buckets([v.traffic_percentage for v in variants])):
                cumulative += n_b
                if z < cumulative:
                    chosen = v.variant_id
                    break
        else:
            chosen = "control" if bool(cls.control_from_bucket(z, control_split_pct)) else "treatment"

        return AssignmentResult(user_id, experiment_id, layer_id, False, True, chosen, z, REASON_ASSIGNED)

    @classmethod
    def validate_orthogonality_and_uniformity(
        cls,
        exp_id_1: str,
        exp_id_2: str,
        n_sim_users: int = 20000,
        n_uniformity_bins: int = 50,
        n_layer_bins: int = 10,
    ) -> dict:
        """
        Проверки ASG-1 на n_sim_users синтетических ID:
        независимость вариантов двух экспериментов (χ² 2×2), равномерность бакетов варианта,
        независимость бакетов слоёв exp_id_1 и exp_id_2 (χ² на таблице сопряжённости).
        """
        if n_sim_users < 100:
            raise ValueError("n_sim_users должен быть >= 100.")
        uids = [f"sim_user_{i}" for i in range(n_sim_users)]
        z1 = cls.buckets_for(exp_id_1, uids)
        z2 = cls.buckets_for(exp_id_2, uids)
        c1 = cls.control_from_bucket(z1, 50.0)
        c2 = cls.control_from_bucket(z2, 50.0)
        contingency = np.array([[np.sum(c1 & c2), np.sum(c1 & ~c2)], [np.sum(~c1 & c2), np.sum(~c1 & ~c2)]])
        _, p_indep, _, _ = stats.chi2_contingency(contingency)
        control_share_1 = float(c1.mean())

        observed = np.bincount(z1 * n_uniformity_bins // N_BUCKETS, minlength=n_uniformity_bins)
        _, p_uniform = stats.chisquare(observed)

        l1 = cls.buckets_for(f"layer:{exp_id_1}", uids) * n_layer_bins // N_BUCKETS
        l2 = cls.buckets_for(f"layer:{exp_id_2}", uids) * n_layer_bins // N_BUCKETS
        layer_table = np.zeros((n_layer_bins, n_layer_bins), dtype=np.int64)
        np.add.at(layer_table, (l1, l2), 1)
        _, p_layers, _, _ = stats.chi2_contingency(layer_table)

        return {
            "n_simulated": n_sim_users,
            "exp1_control_share": control_share_1,
            "independence_chi2_pvalue": float(p_indep),
            "bucket_uniformity_chi2_pvalue": float(p_uniform),
            "layer_independence_chi2_pvalue": float(p_layers),
            "is_orthogonal_and_valid": bool(
                p_indep > 0.01 and 0.48 <= control_share_1 <= 0.52 and p_uniform > 0.001 and p_layers > 0.001
            ),
        }
