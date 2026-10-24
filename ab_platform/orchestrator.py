from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import pandas as pd

from ab_platform.schemas.passport import (
    Direction,
    ExperimentPassport,
    MetricDefinition,
    MetricType,
)
from ab_platform.engine.hygiene import DataHygieneProcessor
from ab_platform.engine.cuped import VarianceReductionEngine
from ab_platform.engine.criterion_router import StatCriterionRouter
from ab_platform.engine.safeguards import ExperimentSafeguardEngine
from ab_platform.copilot.arbor_engine import ArborDecisionCopilot


@dataclass
class GuardrailEvalSummary:
    variant_id: str
    metric_name: str
    control_mean: float
    treatment_mean: float
    relative_change_pct: float
    degradation_pct: float
    max_allowed_degradation_pct: float
    p_value: float
    breached: bool
    action: str
    rationale: str


@dataclass
class SecondaryMetricEvalSummary:
    variant_id: str
    metric_name: str
    metric_type: str
    tier_order: int
    tier_alpha: float
    control_mean: float
    treatment_mean: float
    relative_diff_pct: float
    p_value: float
    is_significant_fdr: bool
    is_significant_tiered: bool


@dataclass
class ExperimentEvaluationReport:
    experiment_id: str
    decision_status: str
    can_trust_data: bool
    winning_variant: Optional[str]
    srm_diagnosis: Dict[str, Any]
    double_exposure_removed_pct: float
    clean_sample_size: int
    cuped_diagnostics: Dict[str, Any]
    primary_metric_result: Dict[str, Any]
    variant_comparisons: Dict[str, Dict[str, Any]]
    guardrail_results: List[GuardrailEvalSummary]
    secondary_results: List[SecondaryMetricEvalSummary]
    site_wide_impact: Dict[str, float]
    executive_summary_ru: str


class SmartABPlatformOrchestrator:
    """
    Главный оркестратор умной A/B-платформы:
    - Жестко проверяет наличие ВСЕХ колонок (Primary, Guardrail, Secondary, Denominators) — никаких silent skips!
    - Принудительно агрегирует сырые события до единицы рандомизации (1 строка == 1 юзер/кластер) перед SRM и тестами
    - Поддерживает Multi-Arm эксперименты (A/B/C/n) с мультиномиальным SRM и поправкой Холма-Бонферрони
    """

    @staticmethod
    def _validate_required_columns(
        df: pd.DataFrame,
        passport: ExperimentPassport,
        user_col: str,
        variant_col: str,
        covariate_col: Optional[str],
    ) -> None:
        required = {user_col, variant_col, passport.primary_metric.numerator_col}
        if passport.primary_metric.denominator_col:
            required.add(passport.primary_metric.denominator_col)

        for g in passport.guardrail_metrics:
            required.add(g.metric.numerator_col)
            if g.metric.denominator_col:
                required.add(g.metric.denominator_col)

        for s in passport.secondary_metrics:
            required.add(s.numerator_col)
            if s.denominator_col:
                required.add(s.denominator_col)

        missing = sorted(col for col in required if col not in df.columns)
        if missing:
            raise ValueError(
                f"Schema Validation Error: В датасете отсутствуют обязательные колонки из Паспорта Эксперимента: {missing}"
            )

        if covariate_col and covariate_col not in df.columns:
            raise ValueError(
                f"Schema Validation Error: Запрошенная колонка ковариаты CUPED '{covariate_col}' отсутствует в датасете."
            )

    @staticmethod
    def _aggregate_to_randomization_unit(
        df: pd.DataFrame,
        passport: ExperimentPassport,
        user_col: str,
        variant_col: str,
        covariate_col: Optional[str],
    ) -> pd.DataFrame:
        if df.duplicated(subset=[user_col]).sum() == 0:
            return df.copy()

        agg_rules: Dict[str, str] = {}
        all_metrics = (
            [passport.primary_metric]
            + [g.metric for g in passport.guardrail_metrics]
            + list(passport.secondary_metrics)
        )
        for m in all_metrics:
            if m.metric_type == MetricType.BINARY:
                agg_rules[m.numerator_col] = "max"
            else:
                agg_rules[m.numerator_col] = "sum"
            if m.denominator_col:
                agg_rules[m.denominator_col] = "sum"

        if covariate_col and covariate_col in df.columns:
            agg_rules[covariate_col] = "first"

        return df.groupby([user_col, variant_col], as_index=False).agg(agg_rules)

    @staticmethod
    def _evaluate_pair_metric(
        df_pair: pd.DataFrame,
        metric_def: MetricDefinition,
        control_id: str,
        treatment_id: str,
        variant_col: str = "variant",
        covariate_col: Optional[str] = None,
        use_lins_method: bool = False,
        alpha: float = 0.05,
        quantile_target: float = 0.90,
    ) -> tuple[Dict[str, Any], Dict[str, Any]]:
        work_df = df_pair.copy()
        num_col = metric_def.numerator_col

        if metric_def.winsorize_percentile is not None and metric_def.metric_type in (
            MetricType.CONTINUOUS,
            MetricType.RATIO,
        ):
            work_df = DataHygieneProcessor.apply_winsorization(
                work_df,
                metric_col=num_col,
                lower_quantile=0.0,
                upper_quantile=metric_def.winsorize_percentile,
            )

        cuped_diag: Dict[str, Any] = {
            "applied": False,
            "theta": 0.0,
            "correlation": 0.0,
            "variance_reduction_pct": 0.0,
            "sensitivity_gain_factor": 1.0,
        }
        target_col = num_col

        if (
            covariate_col
            and covariate_col in work_df.columns
            and metric_def.metric_type == MetricType.CONTINUOUS
        ):
            work_df, cuped_diag = VarianceReductionEngine.apply_cuped(
                work_df,
                metric_col=num_col,
                covariate_col=covariate_col,
                winsorize_quantile=metric_def.winsorize_percentile,
            )
            if cuped_diag["applied"]:
                if use_lins_method:
                    work_df = VarianceReductionEngine.apply_lins_method(
                        work_df,
                        metric_col=num_col,
                        covariate_col=covariate_col,
                        variant_col=variant_col,
                        winsorize_quantile=metric_def.winsorize_percentile,
                    )
                    target_col = f"{num_col}_lin_cuped"
                else:
                    target_col = f"{num_col}_cuped"

        c_df = work_df[work_df[variant_col] == control_id]
        t_df = work_df[work_df[variant_col] == treatment_id]

        if metric_def.metric_type == MetricType.BINARY:
            stat_res = StatCriterionRouter.proportions_z_test(
                c_df[target_col].to_numpy(dtype=float),
                t_df[target_col].to_numpy(dtype=float),
                alpha=alpha,
            )
        elif metric_def.metric_type == MetricType.CONTINUOUS:
            stat_res = StatCriterionRouter.welchs_t_test(
                c_df[target_col].to_numpy(dtype=float),
                t_df[target_col].to_numpy(dtype=float),
                alpha=alpha,
            )
        elif metric_def.metric_type == MetricType.RATIO:
            den_col = metric_def.denominator_col
            if not den_col or den_col not in work_df.columns:
                raise ValueError(f"Колонка знаменателя '{den_col}' не найдена в датасете.")
            stat_res = StatCriterionRouter.ratio_delta_test(
                c_df[num_col].to_numpy(dtype=float),
                c_df[den_col].to_numpy(dtype=float),
                t_df[num_col].to_numpy(dtype=float),
                t_df[den_col].to_numpy(dtype=float),
                alpha=alpha,
            )
        elif metric_def.metric_type == MetricType.QUANTILE:
            stat_res = StatCriterionRouter.bootstrap_quantile_test(
                c_df[num_col].to_numpy(dtype=float),
                t_df[num_col].to_numpy(dtype=float),
                quantile=quantile_target,
                alpha=alpha,
            )
        else:
            raise ValueError(f"Неизвестный тип метрики: {metric_def.metric_type}")

        return stat_res, cuped_diag

    @classmethod
    def evaluate_experiment(
        cls,
        passport: ExperimentPassport,
        df: pd.DataFrame,
        user_col: str = "user_id",
        variant_col: str = "variant",
        covariate_col: Optional[str] = None,
        use_lins_method: Optional[bool] = None,
        assigned_counts: Optional[tuple[int, int]] = None,
        triggered_share_alpha: float = 1.0,
        secondary_tiers: Optional[Dict[str, int]] = None,
    ) -> ExperimentEvaluationReport:
        cls._validate_required_columns(df, passport, user_col, variant_col, covariate_col)

        effective_use_lin = passport.use_lins_method if use_lins_method is None else use_lins_method

        dedup_df, double_exp_frac = DataHygieneProcessor.remove_double_exposure(
            df, user_col=user_col, variant_col=variant_col
        )

        clean_df = cls._aggregate_to_randomization_unit(
            dedup_df, passport, user_col=user_col, variant_col=variant_col, covariate_col=covariate_col
        )

        control_alloc = next(v for v in passport.variants if v.is_control)
        control_id = control_alloc.variant_id
        treatment_allocs = [v for v in passport.variants if not v.is_control]

        observed_variants = set(clean_df[variant_col].unique())
        expected_variants = {v.variant_id for v in passport.variants}
        missing_variants = expected_variants - observed_variants
        if missing_variants:
            raise ValueError(
                f"В датасете отсутствуют наблюдения для вариантов, заявленных в Паспорте: {sorted(missing_variants)}"
            )

        if len(passport.variants) == 2:
            t_id = treatment_allocs[0].variant_id
            n_c_trig = int((clean_df[variant_col] == control_id).sum())
            n_t_trig = int((clean_df[variant_col] == t_id).sum())
            n_c_ass, n_t_ass = assigned_counts if assigned_counts is not None else (n_c_trig, n_t_trig)
            expected_treat_ratio = treatment_allocs[0].traffic_percentage / 100.0
            srm_diag = ExperimentSafeguardEngine.diagnose_srm_stage(
                assigned_control=n_c_ass,
                assigned_treatment=n_t_ass,
                triggered_control=n_c_trig,
                triggered_treatment=n_t_trig,
                expected_ratio=expected_treat_ratio,
            )
            has_srm = bool(srm_diag["assigned_srm"] or srm_diag["triggered_srm"])
        else:
            obs_counts = [int((clean_df[variant_col] == v.variant_id).sum()) for v in passport.variants]
            exp_shares = [v.traffic_percentage / 100.0 for v in passport.variants]
            multi_srm = ExperimentSafeguardEngine.detect_multinomial_srm(obs_counts, exp_shares)
            has_srm = bool(multi_srm["srm_detected"])
            srm_diag = {
                "assigned_srm": False,
                "triggered_srm": has_srm,
                "diagnosis": (
                    "CRITICAL [Multi-Arm SRM]: Обнаружен перекос долей трафика между вариантами."
                    if has_srm
                    else "OK: Доли всех вариантов соответствуют паспорту."
                ),
                "multinomial_details": multi_srm,
            }

        variant_comparisons: Dict[str, Dict[str, Any]] = {}
        cuped_diag_map: Dict[str, Dict[str, Any]] = {}
        arm_pvals: List[float] = []
        arm_ids: List[str] = []

        for t_alloc in treatment_allocs:
            t_id = t_alloc.variant_id
            pair_df = clean_df[clean_df[variant_col].isin([control_id, t_id])]
            p_res, c_diag = cls._evaluate_pair_metric(
                df_pair=pair_df,
                metric_def=passport.primary_metric,
                control_id=control_id,
                treatment_id=t_id,
                variant_col=variant_col,
                covariate_col=covariate_col,
                use_lins_method=effective_use_lin,
                alpha=passport.alpha,
            )
            variant_comparisons[t_id] = p_res
            cuped_diag_map[t_id] = c_diag
            arm_pvals.append(p_res["p_value"])
            arm_ids.append(t_id)

        if len(arm_ids) > 1:
            holm_rejects, holm_pvals = StatCriterionRouter.apply_holm_bonferroni(arm_pvals, alpha=passport.alpha)
            for t_id, is_sig_h, p_adj in zip(arm_ids, holm_rejects, holm_pvals):
                variant_comparisons[t_id]["p_value_holm_adjusted"] = p_adj
                variant_comparisons[t_id]["is_significant"] = bool(is_sig_h)

        guardrail_summaries: List[GuardrailEvalSummary] = []
        secondary_summaries: List[SecondaryMetricEvalSummary] = []
        breached_arms = set()
        tier_alpha_map = {1: 0.05, 2: 0.01, 3: 0.001}

        for t_alloc in treatment_allocs:
            t_id = t_alloc.variant_id
            pair_df = clean_df[clean_df[variant_col].isin([control_id, t_id])]

            for g_def in passport.guardrail_metrics:
                g_stat, _ = cls._evaluate_pair_metric(
                    df_pair=pair_df,
                    metric_def=g_def.metric,
                    control_id=control_id,
                    treatment_id=t_id,
                    variant_col=variant_col,
                    covariate_col=None,
                    alpha=0.05,
                )
                higher_is_better = g_def.metric.expected_direction == Direction.INCREASE
                breach_eval = ExperimentSafeguardEngine.evaluate_guardrail_breach(
                    baseline_control=g_stat["mean_control"],
                    treatment_value=g_stat["mean_treatment"],
                    p_value=g_stat["p_value"],
                    max_allowed_degradation_rel=g_def.max_allowed_degradation_rel,
                    higher_is_better=higher_is_better,
                )
                if breach_eval["guardrail_breached"]:
                    breached_arms.add(t_id)

                guardrail_summaries.append(
                    GuardrailEvalSummary(
                        variant_id=t_id,
                        metric_name=g_def.metric.name,
                        control_mean=g_stat["mean_control"],
                        treatment_mean=g_stat["mean_treatment"],
                        relative_change_pct=breach_eval["relative_change_pct"],
                        degradation_pct=breach_eval["degradation_pct"],
                        max_allowed_degradation_pct=breach_eval["threshold_pct"],
                        p_value=g_stat["p_value"],
                        breached=breach_eval["guardrail_breached"],
                        action=breach_eval["action"],
                        rationale=g_def.tradeoff_rationale,
                    )
                )

            raw_sec = []
            sec_pvals = []
            for sec_def in passport.secondary_metrics:
                tier = (secondary_tiers or {}).get(sec_def.name, 2)
                t_alpha = tier_alpha_map.get(tier, 0.01)
                s_res, _ = cls._evaluate_pair_metric(
                    df_pair=pair_df,
                    metric_def=sec_def,
                    control_id=control_id,
                    treatment_id=t_id,
                    variant_col=variant_col,
                    covariate_col=None,
                    alpha=t_alpha,
                )
                raw_sec.append((sec_def, tier, t_alpha, s_res))
                sec_pvals.append(s_res["p_value"])

            fdr_decisions = StatCriterionRouter.apply_fdr_correction(sec_pvals, alpha=passport.alpha)
            for (sec_def, tier, t_alpha, s_res), is_sig_fdr in zip(raw_sec, fdr_decisions):
                secondary_summaries.append(
                    SecondaryMetricEvalSummary(
                        variant_id=t_id,
                        metric_name=sec_def.name,
                        metric_type=sec_def.metric_type.value,
                        tier_order=tier,
                        tier_alpha=t_alpha,
                        control_mean=s_res["mean_control"],
                        treatment_mean=s_res["mean_treatment"],
                        relative_diff_pct=round(s_res["relative_diff"] * 100.0, 3),
                        p_value=s_res["p_value"],
                        is_significant_fdr=bool(is_sig_fdr),
                        is_significant_tiered=bool(s_res["p_value"] < t_alpha),
                    )
                )

        expected_dir = passport.primary_metric.expected_direction
        safe_arms = [t_id for t_id in arm_ids if t_id not in breached_arms]
        candidate_pool = safe_arms if safe_arms else arm_ids

        def _arm_score(tid: str) -> float:
            r = variant_comparisons[tid]["relative_diff"]
            return r if expected_dir == Direction.INCREASE else -r

        best_arm_id = max(candidate_pool, key=_arm_score)
        primary_res = variant_comparisons[best_arm_id]
        cuped_diag = cuped_diag_map[best_arm_id]

        rel_diff = primary_res["relative_diff"]
        is_positive_move = (rel_diff > 0) if expected_dir == Direction.INCREASE else (rel_diff < 0)
        primary_won = bool(primary_res["is_significant"] and is_positive_move and (best_arm_id not in breached_arms))
        primary_degraded = bool(primary_res["is_significant"] and not is_positive_move)
        any_guardrail_breached = len(breached_arms) == len(arm_ids)

        swi = ArborDecisionCopilot.compute_site_wide_impact(
            local_lift_rel=rel_diff, triggered_share_alpha=triggered_share_alpha
        )

        winning_variant: Optional[str] = None
        if has_srm or double_exp_frac > 0.05:
            decision_status = "INVALID_EXPERIMENT_SRM_OR_CONTAMINATION"
            can_trust = False
            verdict_ru = (
                f"⛔ ЭКСПЕРИМЕНТ СКОМПРОМЕТИРОВАН (ДАННЫМ ДОВЕРЯТЬ НЕЛЬЗЯ): "
                f"{srm_diag['diagnosis']} Доля пересечения групп (Double Exposure): {double_exp_frac*100:.2f}%."
            )
        elif any_guardrail_breached:
            decision_status = "DO_NOT_SHIP_GUARDRAIL_BREACH"
            can_trust = True
            breached_names = ", ".join(sorted({g.metric_name for g in guardrail_summaries if g.breached}))
            verdict_ru = (
                f"🛑 ЗАПРЕТ РАСКАТКИ (ПРОБОЙ ЗАЩИТНЫХ МЕТРИК): Пробиты критические Guardrail-пороги "
                f"безопасности по метрикам: [{breached_names}]. Рекомендуется откат (Revert)."
            )
        elif primary_won:
            decision_status = "SHIP_TO_PRODUCTION"
            can_trust = True
            winning_variant = best_arm_id
            ci_low_pct = (primary_res["ci_lower"] / max(abs(primary_res["mean_control"]), 1e-12)) * 100.0
            ci_high_pct = (primary_res["ci_upper"] / max(abs(primary_res["mean_control"]), 1e-12)) * 100.0
            verdict_ru = (
                f"✅ РЕКОМЕНДУЕТСЯ К РАСКАТКЕ НА 100% (Вариант '{best_arm_id}'): Целевая метрика "
                f"'{passport.primary_metric.name}' статистически значимо улучшилась на {rel_diff*100:+.2f}% "
                f"(95% ДИ: [{ci_low_pct:+.2f}%, {ci_high_pct:+.2f}%], p-value = {primary_res['p_value']:.4f}, "
                f"критерий: {primary_res['criterion']}, сжатие дисперсии CUPED: {cuped_diag['variance_reduction_pct']:.1f}%). "
                f"Влияние на весь бизнес (Site Wide Impact): {swi['site_wide_impact_pct']:+.2f}%."
            )
        elif primary_degraded:
            decision_status = "DO_NOT_SHIP_PRIMARY_DEGRADED"
            can_trust = True
            verdict_ru = (
                f"❌ ОТКЛОНИТЬ ГИПОТЕЗУ (УХУДШЕНИЕ ЦЕЛЕВОЙ МЕТРИКИ): В варианте '{best_arm_id}' метрика "
                f"'{passport.primary_metric.name}' ухудшилась на {rel_diff*100:+.2f}% (p-value = {primary_res['p_value']:.4f})."
            )
        else:
            decision_status = "NEUTRAL_NO_SIGNIFICANT_EFFECT"
            can_trust = True
            verdict_ru = (
                f"⚪ СТАТИСТИЧЕСКИ ЗНАЧИМОГО ЭФФЕКТА НЕ ОБНАРУЖЕНО: Лучший вариант '{best_arm_id}' показал "
                f"{rel_diff*100:+.2f}% (p-value = {primary_res['p_value']:.4f} >= {passport.alpha})."
            )

        return ExperimentEvaluationReport(
            experiment_id=passport.experiment_id,
            decision_status=decision_status,
            can_trust_data=can_trust,
            winning_variant=winning_variant,
            srm_diagnosis=srm_diag,
            double_exposure_removed_pct=round(double_exp_frac * 100.0, 3),
            clean_sample_size=len(clean_df),
            cuped_diagnostics=cuped_diag,
            primary_metric_result=primary_res,
            variant_comparisons=variant_comparisons,
            guardrail_results=guardrail_summaries,
            secondary_results=secondary_summaries,
            site_wide_impact=swi,
            executive_summary_ru=verdict_ru,
        )
