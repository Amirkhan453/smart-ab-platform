from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import numpy as np
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
    srm_diagnosis: Dict[str, Any]
    double_exposure_removed_pct: float
    clean_sample_size: int
    cuped_diagnostics: Dict[str, Any]
    primary_metric_result: Dict[str, Any]
    guardrail_results: List[GuardrailEvalSummary]
    secondary_results: List[SecondaryMetricEvalSummary]
    site_wide_impact: Dict[str, float]
    executive_summary_ru: str


class SmartABPlatformOrchestrator:
    """
    Главный оркестратор умной A/B-платформы:
    Связывает Паспорт Эксперимента, SRM-защиту, очистку данных, CUPED/Lin,
    маршрутизатор стат-критериев, Guardrail-контроль и генератор бизнес-вердиктов.
    """

    @staticmethod
    def _evaluate_single_metric(
        df: pd.DataFrame,
        metric_def: MetricDefinition,
        variant_col: str = "variant",
        covariate_col: Optional[str] = None,
        use_lins_method: bool = False,
        alpha: float = 0.05,
        quantile_target: float = 0.90,
    ) -> tuple[Dict[str, Any], Dict[str, Any]]:
        work_df = df.copy()
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

        c_df = work_df[work_df[variant_col] == "control"]
        t_df = work_df[work_df[variant_col] == "treatment"]

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
        use_lins_method: bool = False,
        assigned_counts: Optional[ tuple[int, int] ] = None,
        triggered_share_alpha: float = 1.0,
        secondary_tiers: Optional[Dict[str, int]] = None,
    ) -> ExperimentEvaluationReport:
        """
        Полный цикл автоматической проверки и подведения итогов эксперимента:
        1) Очистка от Double Exposure
        2) Проверка SRM (Assigned & Triggered) через последовательный тест SPRT
        3) Оценка целевой метрики с авто-выбором критерия и сжатием дисперсии CUPED / Lin
        4) Оценка вторичных метрик с контролем FDR и трехуровневыми порогами значимости (0.05 / 0.01 / 0.001)
        5) Проверка Guardrail-метрик на пробой порога безопасности
        6) Формирование честного бизнес-вердикта простым языком
        """
        clean_df, double_exp_frac = DataHygieneProcessor.remove_double_exposure(
            df, user_col=user_col, variant_col=variant_col
        )

        n_c_trig = int((clean_df[variant_col] == "control").sum())
        n_t_trig = int((clean_df[variant_col] == "treatment").sum())

        if assigned_counts is not None:
            n_c_ass, n_t_ass = assigned_counts
        else:
            n_c_ass, n_t_ass = n_c_trig, n_t_trig

        treatment_var = next((v for v in passport.variants if not v.is_control), passport.variants[-1])
        expected_treat_ratio = treatment_var.traffic_percentage / 100.0

        srm_diag = ExperimentSafeguardEngine.diagnose_srm_stage(
            assigned_control=n_c_ass,
            assigned_treatment=n_t_ass,
            triggered_control=n_c_trig,
            triggered_treatment=n_t_trig,
            expected_ratio=expected_treat_ratio,
        )
        has_srm = bool(srm_diag["assigned_srm"] or srm_diag["triggered_srm"])

        primary_res, cuped_diag = cls._evaluate_single_metric(
            df=clean_df,
            metric_def=passport.primary_metric,
            variant_col=variant_col,
            covariate_col=covariate_col,
            use_lins_method=use_lins_method,
            alpha=passport.alpha,
        )

        # Оценка вторичных метрик (с поддержкой трехуровневых порогов 0.05, 0.01, 0.001 и FDR)
        tier_alpha_map = {1: 0.05, 2: 0.01, 3: 0.001}
        raw_sec_results = []
        sec_pvals = []
        for sec_def in passport.secondary_metrics:
            if sec_def.numerator_col not in clean_df.columns:
                continue
            tier = (secondary_tiers or {}).get(sec_def.name, 2)
            t_alpha = tier_alpha_map.get(tier, 0.01)
            s_res, _ = cls._evaluate_single_metric(
                df=clean_df,
                metric_def=sec_def,
                variant_col=variant_col,
                covariate_col=None,
                alpha=t_alpha,
            )
            raw_sec_results.append((sec_def, tier, t_alpha, s_res))
            sec_pvals.append(s_res["p_value"])

        fdr_decisions = StatCriterionRouter.apply_fdr_correction(sec_pvals, alpha=passport.alpha)
        secondary_summaries: List[SecondaryMetricEvalSummary] = []
        for (sec_def, tier, t_alpha, s_res), is_sig_fdr in zip(raw_sec_results, fdr_decisions):
            secondary_summaries.append(
                SecondaryMetricEvalSummary(
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

        # Оценка Guardrail-метрик
        guardrail_summaries: List[GuardrailEvalSummary] = []
        any_guardrail_breached = False
        for g_def in passport.guardrail_metrics:
            if g_def.metric.numerator_col not in clean_df.columns:
                continue
            g_stat, _ = cls._evaluate_single_metric(
                df=clean_df,
                metric_def=g_def.metric,
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
                any_guardrail_breached = True

            guardrail_summaries.append(
                GuardrailEvalSummary(
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

        # Проверка направления изменения целевой метрики
        expected_dir = passport.primary_metric.expected_direction
        rel_diff = primary_res["relative_diff"]
        is_positive_move = (rel_diff > 0) if expected_dir == Direction.INCREASE else (rel_diff < 0)
        primary_won = bool(primary_res["is_significant"] and is_positive_move)
        primary_degraded = bool(primary_res["is_significant"] and not is_positive_move)

        swi = ArborDecisionCopilot.compute_site_wide_impact(
            local_lift_rel=rel_diff, triggered_share_alpha=triggered_share_alpha
        )

        # Дерево финального решения
        if has_srm or double_exp_frac > 0.05:
            decision_status = "INVALID_EXPERIMENT_SRM_OR_CONTAMINATION"
            can_trust = False
            verdict_ru = (
                f"⛔ ЭКСПЕРИМЕНТ СКОМПРОМЕТИРОВАН (ДАННЫМ ДОВЕРЯТЬ НЕЛЬЗЯ): "
                f"{srm_diag['diagnosis']} Доля пересечения групп (Double Exposure): {double_exp_frac*100:.2f}%. "
                f"Раскатка заблокирована до устранения технического бага сплитования."
            )
        elif any_guardrail_breached:
            decision_status = "DO_NOT_SHIP_GUARDRAIL_BREACH"
            can_trust = True
            breached_names = ", ".join(g.metric_name for g in guardrail_summaries if g.breached)
            verdict_ru = (
                f"🛑 ЗАПРЕТ РАСКАТКИ (ПРОБОЙ ЗАЩИТНЫХ МЕТРИК): Несмотря на изменение целевой метрики "
                f"'{passport.primary_metric.name}' на {rel_diff*100:+.2f}% (p={primary_res['p_value']:.4f}), "
                f"пробиты критические Guardrail-пороги безопасности по метрикам: [{breached_names}]. "
                f"Рекомендуется откат (Revert) и доработка фичи."
            )
        elif primary_won:
            decision_status = "SHIP_TO_PRODUCTION"
            can_trust = True
            ci_low_pct = (primary_res["ci_lower"] / max(abs(primary_res["mean_control"]), 1e-12)) * 100.0
            ci_high_pct = (primary_res["ci_upper"] / max(abs(primary_res["mean_control"]), 1e-12)) * 100.0
            verdict_ru = (
                f"✅ РЕКОМЕНДУЕТСЯ К РАСКАТКЕ НА 100%: Целевая метрика '{passport.primary_metric.name}' "
                f"статистически значимо улучшилась на {rel_diff*100:+.2f}% "
                f"(95% ДИ: [{ci_low_pct:+.2f}%, {ci_high_pct:+.2f}%], p-value = {primary_res['p_value']:.4f}, "
                f"критерий: {primary_res['criterion']}, сжатие дисперсии CUPED: {cuped_diag['variance_reduction_pct']:.1f}%). "
                f"Влияние на весь бизнес (Site Wide Impact): {swi['site_wide_impact_pct']:+.2f}%. "
                f"Все защитные Guardrail-метрики в норме."
            )
        elif primary_degraded:
            decision_status = "DO_NOT_SHIP_PRIMARY_DEGRADED"
            can_trust = True
            verdict_ru = (
                f"❌ ОТКЛОНИТЬ ГИПОТЕЗУ (УХУДШЕНИЕ ЦЕЛЕВОЙ МЕТРИКИ): Метрика '{passport.primary_metric.name}' "
                f"статистически значимо ухудшилась на {rel_diff*100:+.2f}% (p-value = {primary_res['p_value']:.4f})."
            )
        else:
            decision_status = "NEUTRAL_NO_SIGNIFICANT_EFFECT"
            can_trust = True
            verdict_ru = (
                f"⚪ СТАТИСТИЧЕСКИ ЗНАЧИМОГО ЭФФЕКТА НЕ ОБНАРУЖЕНО: Изменение '{passport.primary_metric.name}' "
                f"составило {rel_diff*100:+.2f}% (p-value = {primary_res['p_value']:.4f} >= {passport.alpha}). "
                f"Гипотеза не подтвердилась в рамках заданного MDE ({passport.mde_relative*100:.1f}%)."
            )

        return ExperimentEvaluationReport(
            experiment_id=passport.experiment_id,
            decision_status=decision_status,
            can_trust_data=can_trust,
            srm_diagnosis=srm_diag,
            double_exposure_removed_pct=round(double_exp_frac * 100.0, 3),
            clean_sample_size=len(clean_df),
            cuped_diagnostics=cuped_diag,
            primary_metric_result=primary_res,
            guardrail_results=guardrail_summaries,
            secondary_results=secondary_summaries,
            site_wide_impact=swi,
            executive_summary_ru=verdict_ru,
        )
