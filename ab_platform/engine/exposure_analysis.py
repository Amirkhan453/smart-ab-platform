"""
Состав анализа по показам и SRM по сегментам таргетинга (ТЗ v4.1: TGT-3, TGT-5).

Гость входит в анализ в момент первого показа — с тем вариантом и теми признаками, что были тогда, — и остаётся
в нём, даже если признак позже перестал подходить под таргетинг. Отбор по текущему признаку запрещён: если вариант
влияет на признак (ускоряет обновление приложения), такой отбор даёт перекос групп и смещение эффекта.
"""
from typing import Dict, Mapping, Optional, Sequence, Tuple

import pandas as pd

from ab_platform.engine.safeguards import ExperimentSafeguardEngine as SG
from ab_platform.schemas.targeting import ATTRIBUTES

NO_VALUE = "(нет)"


def analysis_population(first_exposures: Mapping[str, Mapping]) -> pd.DataFrame:
    """
    Таблица состава анализа из первых показов (ExposureStore.first_exposures): unit_id, variant, first_exposure_time
    и по колонке attr_<признак> на каждый признак таргетинга (снимок на момент первого показа).
    """
    rows = []
    for unit_id, event in first_exposures.items():
        attrs = event.get("attributes") or {}
        rows.append({"unit_id": unit_id, "variant": event["variant_id"], "first_exposure_time": event["event_time"],
                     **{f"attr_{a}": attrs.get(a, NO_VALUE) for a in ATTRIBUTES}})
    columns = ["unit_id", "variant", "first_exposure_time", *(f"attr_{a}" for a in ATTRIBUTES)]
    return pd.DataFrame(rows, columns=columns)


def srm_by_segment(
    population: pd.DataFrame,
    variants: Sequence[Tuple[str, float]],
    segment_col: Optional[str] = None,
    alpha: float = 0.001,
) -> Dict[str, object]:
    """
    SRM по всему составу и по сегментам segment_col (например attr_platform). variants — (variant_id, доля в %).
    Порог для сегментов — alpha / число сегментов (Бонферрони): иначе при десятке сегментов ложная тревога
    была бы в разы чаще заданной. Перекос только в одном сегменте — признак сбоя показа на этой платформе
    или версии, который общий SRM может не заметить.
    """
    ids = [v for v, _ in variants]
    shares = [p / 100.0 for _, p in variants]
    unknown = sorted(set(population["variant"]) - set(ids))
    if unknown:
        raise ValueError(f"В показах есть варианты вне Паспорта: {unknown}.")

    def check(frame: pd.DataFrame, level: float) -> dict:
        counts = [int((frame["variant"] == v).sum()) for v in ids]
        res = SG.detect_multinomial_srm(counts, shares, alpha_srm=level)
        return {"n": int(len(frame)), "counts": dict(zip(ids, counts)), "srm_detected": res["srm_detected"],
                "chi2_pvalue": res["chi2_pvalue"]}

    out: Dict[str, object] = {"overall": check(population, alpha), "segments": {}, "segment_alpha": None}
    if segment_col is not None:
        if segment_col not in population.columns:
            raise ValueError(f"Нет колонки сегмента '{segment_col}'.")
        groups = dict(tuple(population.groupby(segment_col, sort=True)))
        level = alpha / max(len(groups), 1)
        out["segment_alpha"] = level
        out["segments"] = {str(name): check(frame, level) for name, frame in groups.items()}
    out["srm_detected"] = bool(out["overall"]["srm_detected"] or any(s["srm_detected"] for s in out["segments"].values()))
    return out
