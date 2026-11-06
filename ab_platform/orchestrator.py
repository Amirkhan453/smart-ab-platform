from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

from ab_platform.schemas.passport import (
    Direction,
    Domain,
    ExperimentPassport,
    MetricDefinition,
    MetricType,
)
from ab_platform.engine import cluster_inference
from ab_platform.engine.hygiene import DataHygieneProcessor
from ab_platform.engine.cuped import VarianceReductionEngine
from ab_platform.engine.criterion_router import StatCriterionRouter
from ab_platform.engine.safeguards import ExperimentSafeguardEngine, GuardrailStatus
from ab_platform.copilot.arbor_engine import ArborDecisionCopilot


class DecisionStatus(str, Enum):
    """
    Вердикт правила решения (ТЗ v4, 5.3). SHIP_CANDIDATE — кандидат на раскатку: финальное
    решение ставит человек. SHIP_TO_PRODUCTION — устаревший синоним SHIP_CANDIDATE (до фазы 5).
    """
    INVALID_EXPERIMENT_SRM_OR_CONTAMINATION = "INVALID_EXPERIMENT_SRM_OR_CONTAMINATION"
    DO_NOT_SHIP_GUARDRAIL_BREACH = "DO_NOT_SHIP_GUARDRAIL_BREACH"
    SHIP_CANDIDATE = "SHIP_CANDIDATE"
    SHIP_TO_PRODUCTION = "SHIP_CANDIDATE"  # синоним (alias) для обратной совместимости
    DO_NOT_SHIP_PRIMARY_DEGRADED = "DO_NOT_SHIP_PRIMARY_DEGRADED"
    NEUTRAL_NO_SIGNIFICANT_EFFECT = "NEUTRAL_NO_SIGNIFICANT_EFFECT"
    NEUTRAL_BELOW_PRACTICAL_THRESHOLD = "NEUTRAL_BELOW_PRACTICAL_THRESHOLD"
    INCONCLUSIVE_NEEDS_MORE_DATA = "INCONCLUSIVE_NEEDS_MORE_DATA"
    HOLD_HUMAN_REVIEW_GUARDRAIL_UNCERTAIN = "HOLD_HUMAN_REVIEW_GUARDRAIL_UNCERTAIN"
    HOLD_HUMAN_REVIEW_EVIDENCE = "HOLD_HUMAN_REVIEW_EVIDENCE"
    RUNNING_BLINDED = "RUNNING_BLINDED"


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
    status: str = "SAFE_PROVEN"
    p_value_breach: float = 1.0


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
    evidence_level: str = "A"
    estimand: str = "itt"
    generalization_population: str = ""
    passport_version: int = 1
    passport_hash: Optional[str] = None
    analysis_day: Optional[int] = None
    blinded: bool = False


class SmartABPlatformOrchestrator:
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
        clustered = passport.domain in (Domain.RESTAURANT, Domain.RESTAURANT_TIME)
        for m in all_metrics:
            if m.metric_type == MetricType.BINARY:
                agg_rules[m.numerator_col] = "mean" if clustered else "max"
            elif m.metric_type == MetricType.RATIO:
                agg_rules[m.numerator_col] = "sum"
            else:
                # Ресторан-дни → среднее за день на ресторан (устойчиво к закрытым дням); гость → сумма.
                agg_rules[m.numerator_col] = "mean" if clustered else "sum"
            if m.denominator_col:
                agg_rules[m.denominator_col] = "sum"

        if covariate_col and covariate_col in df.columns:
            agg_rules[covariate_col] = "first"
        for col in passport.stratification_cols:
            if col in df.columns and col not in agg_rules:
                agg_rules[col] = "first"

        return df.groupby([user_col, variant_col], as_index=False).agg(agg_rules)

    @staticmethod
    def _evaluate_cluster_primary(
        pair_df: pd.DataFrame,
        passport: ExperimentPassport,
        control_id: str,
        treatment_id: str,
        variant_col: str,
        covariate_col: Optional[str],
    ) -> tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Primary ресторанного домена (ТЗ v4, 5.5): строки уже агрегированы до кластера,
        оценка — cluster_inference по estimand и inference из Паспорта.
        """
        metric = passport.primary_metric
        col = metric.numerator_col
        clusters = pd.DataFrame({
            "y": pair_df[col].to_numpy(dtype=float),
            "treat": (pair_df[variant_col] == treatment_id).astype(int).to_numpy(),
        })
        cuped = {"applied": False, "theta": 0.0, "correlation": 0.0,
                 "variance_reduction_pct": 0.0, "sensitivity_gain_factor": 1.0, "method": "none"}
        if covariate_col and covariate_col in pair_df.columns:
            x = pair_df[covariate_col].to_numpy(dtype=float)
            clusters["x"] = x
            corr = float(np.corrcoef(clusters["y"], x)[0, 1]) if np.std(x) > 0 else 0.0
            cuped = {"applied": True, "theta": 0.0, "correlation": corr,
                     "variance_reduction_pct": round(corr ** 2 * 100.0, 3),
                     "sensitivity_gain_factor": float(1.0 / np.sqrt(max(1.0 - corr ** 2, 1e-12))),
                     "method": "lin_regression"}
        blocks = [c for c in passport.stratification_cols if c in pair_df.columns]
        if blocks:
            clusters["block"] = pair_df[blocks].astype(str).agg("|".join, axis=1).to_numpy()
        res = cluster_inference.analyze_cluster_rct(
            clusters,
            estimand=passport.estimand.value,
            inference=passport.inference.value,
            alpha=passport.alpha,
            sidedness=passport.sidedness,
            direction=metric.expected_direction.value,
        )
        return res, cuped

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
        cuped_min_r2: float = 0.05,
        sidedness: str = "two_sided",
    ) -> tuple[Dict[str, Any], Dict[str, Any]]:
        direction = metric_def.expected_direction.value
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
                min_r2_threshold=cuped_min_r2,
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
                alpha=alpha, sidedness=sidedness, direction=direction,
            )
        elif metric_def.metric_type == MetricType.CONTINUOUS:
            stat_res = StatCriterionRouter.welchs_t_test(
                c_df[target_col].to_numpy(dtype=float),
                t_df[target_col].to_numpy(dtype=float),
                alpha=alpha, sidedness=sidedness, direction=direction,
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
                alpha=alpha, sidedness=sidedness, direction=direction,
            )
        elif metric_def.metric_type == MetricType.QUANTILE:
            stat_res = StatCriterionRouter.bootstrap_quantile_test(
                c_df[num_col].to_numpy(dtype=float),
                t_df[num_col].to_numpy(dtype=float),
                quantile=quantile_target,
                alpha=alpha, sidedness=sidedness, direction=direction,
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
        use_cuped: Optional[bool] = None,
        cuped_min_r2: Optional[float] = None,
        srm_alpha: Optional[float] = None,
        practical_significance_threshold: Optional[float] = None,
        assigned_counts: Optional[tuple[int, int]] = None,
        triggered_share_alpha: float = 1.0,
        secondary_tiers: Optional[Dict[str, int]] = None,
        analysis_day: Optional[int] = None,
    ) -> ExperimentEvaluationReport:
        """
        Правило решения ТЗ v4, 5.3. ``analysis_day`` — день теста на момент анализа: при
        fixed_horizon и дне меньше planned_duration_days p-value и вердикт по primary скрыты
        (RUNNING_BLINDED), а SRM и guardrails считаются. ``None`` — финальный анализ.
        """
        if analysis_day is not None and analysis_day < 0:
            raise ValueError("analysis_day не может быть отрицательным.")
        blinded = bool(
            analysis_day is not None
            and passport.analysis_mode == "fixed_horizon"
            and analysis_day < passport.planned_duration_days
        )
        effective_use_lin = passport.use_lins_method if use_lins_method is None else use_lins_method
        effective_use_cuped = passport.use_cuped if use_cuped is None else use_cuped
        effective_cuped_min_r2 = passport.cuped_min_r2 if cuped_min_r2 is None else cuped_min_r2
        effective_srm_alpha = passport.srm_alpha if srm_alpha is None else srm_alpha
        effective_practical_threshold = (
            passport.practical_significance_threshold
            if practical_significance_threshold is None
            else practical_significance_threshold
        )
        if effective_practical_threshold is None or effective_practical_threshold <= 0.0:
            effective_practical_threshold = passport.mde_relative

        effective_covariate_col = covariate_col
        covariate_col_from_passport = False
        if effective_covariate_col is None and effective_use_cuped and passport.covariate_col:
            effective_covariate_col = passport.covariate_col
            covariate_col_from_passport = True

        validate_covariate_col = effective_covariate_col
        if covariate_col_from_passport and effective_covariate_col not in df.columns:
            effective_covariate_col = None
            validate_covariate_col = None

        cls._validate_required_columns(df, passport, user_col, variant_col, validate_covariate_col)

        dedup_df, double_exp_frac = DataHygieneProcessor.remove_double_exposure(
            df, user_col=user_col, variant_col=variant_col
        )

        clean_df = cls._aggregate_to_randomization_unit(
            dedup_df, passport, user_col=user_col, variant_col=variant_col, covariate_col=effective_covariate_col
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
            raw_user_variant = df[[user_col, variant_col]].drop_duplicates()
            n_c_raw = int((raw_user_variant[variant_col] == control_id).sum())
            n_t_raw = int((raw_user_variant[variant_col] == t_id).sum())
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
                alpha_srm=effective_srm_alpha,
            )
            raw_srm_check = ExperimentSafeguardEngine.detect_srm_chi2(
                n_c_raw, n_t_raw, expected_treat_ratio, effective_srm_alpha
            )
            srm_diag["post_hygiene_recheck"] = {
                "raw_triggered_srm": bool(raw_srm_check["srm_detected"]),
                "clean_triggered_srm": bool(srm_diag["triggered_srm"]),
                "hygiene_introduced_imbalance": bool(
                    srm_diag["triggered_srm"] and not raw_srm_check["srm_detected"]
                ),
            }
            has_srm = bool(srm_diag["assigned_srm"] or srm_diag["triggered_srm"])
        else:
            obs_counts = [int((clean_df[variant_col] == v.variant_id).sum()) for v in passport.variants]
            exp_shares = [v.traffic_percentage / 100.0 for v in passport.variants]
            multi_srm = ExperimentSafeguardEngine.detect_multinomial_srm(obs_counts, exp_shares, alpha_srm=effective_srm_alpha)
            raw_user_variant = df[[user_col, variant_col]].drop_duplicates()
            raw_obs_counts = [int((raw_user_variant[variant_col] == v.variant_id).sum()) for v in passport.variants]
            raw_multi_srm = ExperimentSafeguardEngine.detect_multinomial_srm(raw_obs_counts, exp_shares, alpha_srm=effective_srm_alpha)
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
                "post_hygiene_recheck": {
                    "raw_triggered_srm": bool(raw_multi_srm["srm_detected"]),
                    "clean_triggered_srm": has_srm,
                    "hygiene_introduced_imbalance": bool(has_srm and not raw_multi_srm["srm_detected"]),
                },
            }

        variant_comparisons: Dict[str, Dict[str, Any]] = {}
        cuped_diag_map: Dict[str, Dict[str, Any]] = {}
        arm_pvals: List[float] = []
        arm_ids: List[str] = []

        for t_alloc in treatment_allocs:
            t_id = t_alloc.variant_id
            pair_df = clean_df[clean_df[variant_col].isin([control_id, t_id])]
            clustered_primary = (
                passport.domain in (Domain.RESTAURANT, Domain.RESTAURANT_TIME)
                and passport.primary_metric.metric_type == MetricType.CONTINUOUS
            )
            if clustered_primary:
                p_res, c_diag = cls._evaluate_cluster_primary(
                    pair_df, passport, control_id, t_id, variant_col, effective_covariate_col
                )
                variant_comparisons[t_id] = p_res
                cuped_diag_map[t_id] = c_diag
                arm_pvals.append(p_res["p_value"])
                arm_ids.append(t_id)
                continue
            p_res, c_diag = cls._evaluate_pair_metric(
                df_pair=pair_df,
                metric_def=passport.primary_metric,
                control_id=control_id,
                treatment_id=t_id,
                variant_col=variant_col,
                covariate_col=effective_covariate_col,
                use_lins_method=effective_use_lin,
                alpha=passport.alpha,
                cuped_min_r2=effective_cuped_min_r2,
                sidedness=passport.sidedness,
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
        inconclusive_guardrail_arms = set()
        all_secondary_raw: List[tuple] = []

        for t_alloc in treatment_allocs:
            t_id = t_alloc.variant_id
            pair_df = clean_df[clean_df[variant_col].isin([control_id, t_id])]

            arm_guardrail_evals = []
            for g_def in passport.guardrail_metrics:
                g_stat, _ = cls._evaluate_pair_metric(
                    df_pair=pair_df,
                    metric_def=g_def.metric,
                    control_id=control_id,
                    treatment_id=t_id,
                    variant_col=variant_col,
                    covariate_col=None,
                    alpha=passport.alpha,
                )
                higher_is_better = g_def.metric.expected_direction == Direction.INCREASE
                breach_eval = ExperimentSafeguardEngine.evaluate_guardrail_breach(
                    baseline_control=g_stat["mean_control"],
                    treatment_value=g_stat["mean_treatment"],
                    ci_lower_diff=g_stat["ci_lower"],
                    ci_upper_diff=g_stat["ci_upper"],
                    max_allowed_degradation_rel=g_def.max_allowed_degradation_rel,
                    higher_is_better=higher_is_better,
                    alpha_guardrail=passport.alpha,
                )
                arm_guardrail_evals.append((g_def, g_stat, breach_eval))

            breach_pvals = [be["p_value_breach"] for _, _, be in arm_guardrail_evals]
            holm_rejects = ExperimentSafeguardEngine.apply_holm_to_breach_signals(breach_pvals, alpha=passport.alpha)

            for (g_def, g_stat, breach_eval), holm_significant in zip(arm_guardrail_evals, holm_rejects):
                status = breach_eval["status"]
                if status == GuardrailStatus.BREACH_PROVEN and not holm_significant:
                    status = GuardrailStatus.INCONCLUSIVE_LOW_POWER
                final_breached = bool(status == GuardrailStatus.BREACH_PROVEN)
                action = breach_eval["action"] if final_breached else (
                    "HOLD_HUMAN_REVIEW" if status == GuardrailStatus.INCONCLUSIVE_LOW_POWER else "KEEP_RUNNING"
                )
                if final_breached:
                    breached_arms.add(t_id)
                elif status == GuardrailStatus.INCONCLUSIVE_LOW_POWER:
                    inconclusive_guardrail_arms.add(t_id)

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
                        breached=final_breached,
                        action=action,
                        rationale=g_def.tradeoff_rationale,
                        status=status.value if hasattr(status, "value") else str(status),
                        p_value_breach=breach_eval["p_value_breach"],
                    )
                )

            for sec_def in passport.secondary_metrics:
                # Тир — метка важности, а не отдельный alpha (ТЗ v4, 5.2): решение только по BH.
                tier = (secondary_tiers or {}).get(sec_def.name, 2)
                t_alpha = passport.alpha
                s_res, _ = cls._evaluate_pair_metric(
                    df_pair=pair_df,
                    metric_def=sec_def,
                    control_id=control_id,
                    treatment_id=t_id,
                    variant_col=variant_col,
                    covariate_col=None,
                    alpha=t_alpha,
                )
                all_secondary_raw.append((t_id, sec_def, tier, t_alpha, s_res))

        all_sec_pvals = [s_res["p_value"] for (_, _, _, _, s_res) in all_secondary_raw]
        fdr_decisions = (
            StatCriterionRouter.apply_fdr_correction(all_sec_pvals, alpha=passport.alpha)
            if all_sec_pvals
            else []
        )
        for (t_id, sec_def, tier, t_alpha, s_res), is_sig_fdr in zip(all_secondary_raw, fdr_decisions):
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
                    is_significant_tiered=bool(is_sig_fdr),
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
        primary_significant_positive = bool(
            primary_res["is_significant"] and is_positive_move and (best_arm_id not in breached_arms)
        )
        practical_threshold_for_ship = passport.practical_significance_threshold
        meets_practical = bool(abs(rel_diff) >= practical_threshold_for_ship)
        primary_won = primary_significant_positive and meets_practical
        primary_degraded = bool(primary_res["is_significant"] and not is_positive_move)
        evidence = passport.evidence_level
        any_guardrail_breached = len(breached_arms) == len(arm_ids)

        swi = ArborDecisionCopilot.compute_site_wide_impact(
            local_lift_rel=rel_diff, triggered_share_alpha=triggered_share_alpha
        )

        ci_conf_pct = int(round((1.0 - passport.alpha) * 100))
        winning_variant: Optional[str] = None
        if has_srm or double_exp_frac > passport.max_double_exposure_share:
            decision_status = DecisionStatus.INVALID_EXPERIMENT_SRM_OR_CONTAMINATION
            can_trust = False
            verdict_ru = (
                f"⛔ ЭКСПЕРИМЕНТ СКОМПРОМЕТИРОВАН (ДАННЫМ ДОВЕРЯТЬ НЕЛЬЗЯ): "
                f"{srm_diag['diagnosis']} Доля пересечения групп (Double Exposure): {double_exp_frac*100:.2f}%."
            )
        elif any_guardrail_breached:
            decision_status = DecisionStatus.DO_NOT_SHIP_GUARDRAIL_BREACH
            can_trust = True
            breached_names = ", ".join(sorted({g.metric_name for g in guardrail_summaries if g.breached}))
            verdict_ru = (
                f"🛑 ЗАПРЕТ РАСКАТКИ (ПРОБОЙ ЗАЩИТНЫХ МЕТРИК, доказан после поправки Холма): Пробиты "
                f"критические Guardrail-пороги безопасности по метрикам: [{breached_names}]. "
                f"Рекомендуется откат (Revert)."
            )
        elif blinded:
            decision_status = DecisionStatus.RUNNING_BLINDED
            can_trust = True
            verdict_ru = (
                f"⏳ ТЕСТ ИДЁТ (день {analysis_day} из {passport.planned_duration_days}): p-value и вердикт по "
                f"целевой метрике скрыты до горизонта (fixed horizon, ТЗ v4 5.4). SRM и guardrails в норме."
            )
        elif best_arm_id in inconclusive_guardrail_arms:
            decision_status = DecisionStatus.HOLD_HUMAN_REVIEW_GUARDRAIL_UNCERTAIN
            can_trust = True
            inconclusive_names = ", ".join(sorted({
                g.metric_name for g in guardrail_summaries
                if g.variant_id == best_arm_id and g.status == GuardrailStatus.INCONCLUSIVE_LOW_POWER.value
            }))
            verdict_ru = (
                f"🟡 ТРЕБУЕТСЯ РЕШЕНИЕ ЧЕЛОВЕКА (НЕДОСТАТОЧНО МОЩНОСТИ ПО GUARDRAIL): По варианту "
                f"'{best_arm_id}' безопасность не доказана и пробой не доказан для метрик: "
                f"[{inconclusive_names}]. Автоматический SHIP заблокирован."
            )
        elif primary_won and not passport.auto_ship_allowed:
            decision_status = DecisionStatus.HOLD_HUMAN_REVIEW_EVIDENCE
            can_trust = True
            verdict_ru = (
                f"🟡 ТРЕБУЕТСЯ РЕШЕНИЕ ЧЕЛОВЕКА (УРОВЕНЬ ДОКАЗАТЕЛЬНОСТИ {evidence.value}): эффект "
                f"{rel_diff*100:+.2f}% (p-value = {primary_res['p_value']:.4f}) получен без рандомизации "
                f"(метод: {passport.quasi_method.value if passport.quasi_method else passport.experiment_type.value}). "
                f"Автоматический SHIP запрещён (ТЗ v4, EVD-1)."
            )
        elif primary_significant_positive and not meets_practical:
            decision_status = DecisionStatus.NEUTRAL_BELOW_PRACTICAL_THRESHOLD
            can_trust = True
            verdict_ru = (
                f"⚪ ЭФФЕКТ ЗНАЧИМ, НО МЕНЬШЕ ПОРОГА ПРАКТИЧЕСКОЙ ЗНАЧИМОСТИ: вариант '{best_arm_id}' дал "
                f"{rel_diff*100:+.2f}% (p-value = {primary_res['p_value']:.4f}) при пороге "
                f"{practical_threshold_for_ship*100:.2f}% из Паспорта. Раскатка не рекомендуется."
            )
        elif primary_won:
            decision_status = DecisionStatus.SHIP_CANDIDATE
            can_trust = True
            winning_variant = best_arm_id
            ci_low_pct = (primary_res["ci_lower"] / max(abs(primary_res["mean_control"]), 1e-12)) * 100.0
            ci_high_pct = (primary_res["ci_upper"] / max(abs(primary_res["mean_control"]), 1e-12)) * 100.0
            verdict_ru = (
                f"✅ КАНДИДАТ НА РАСКАТКУ (Вариант '{best_arm_id}', решение за человеком): Целевая метрика "
                f"'{passport.primary_metric.name}' статистически значимо улучшилась на {rel_diff*100:+.2f}% "
                f"({ci_conf_pct}% ДИ: [{ci_low_pct:+.2f}%, {ci_high_pct:+.2f}%], p-value = {primary_res['p_value']:.4f}, "
                f"критерий: {primary_res['criterion']}, сжатие дисперсии CUPED: {cuped_diag['variance_reduction_pct']:.1f}%). "
                f"Влияние на весь бизнес (Site Wide Impact): {swi['site_wide_impact_pct']:+.2f}%."
            )
        elif primary_degraded:
            decision_status = DecisionStatus.DO_NOT_SHIP_PRIMARY_DEGRADED
            can_trust = True
            verdict_ru = (
                f"❌ ОТКЛОНИТЬ ГИПОТЕЗУ (УХУДШЕНИЕ ЦЕЛЕВОЙ МЕТРИКИ): В варианте '{best_arm_id}' метрика "
                f"'{passport.primary_metric.name}' ухудшилась на {rel_diff*100:+.2f}% (p-value = {primary_res['p_value']:.4f})."
            )
        else:
            denom_abs = max(abs(primary_res["mean_control"]), 1e-12)
            ci_lower_rel = primary_res["ci_lower"] / denom_abs
            ci_upper_rel = primary_res["ci_upper"] / denom_abs
            ci_covers_zero = bool(ci_lower_rel <= 0.0 <= ci_upper_rel)
            if expected_dir == Direction.INCREASE:
                ci_covers_practical_effect = bool(ci_upper_rel >= effective_practical_threshold)
            else:
                ci_covers_practical_effect = bool(ci_lower_rel <= -effective_practical_threshold)

            if ci_covers_zero and ci_covers_practical_effect:
                decision_status = DecisionStatus.INCONCLUSIVE_NEEDS_MORE_DATA
                can_trust = True
                verdict_ru = (
                    f"🔸 НЕДОСТАТОЧНО ДАННЫХ ДЛЯ ВЫВОДА: Лучший вариант '{best_arm_id}' дал "
                    f"{rel_diff*100:+.2f}% (p-value = {primary_res['p_value']:.4f}), но {ci_conf_pct}% ДИ "
                    f"[{ci_lower_rel*100:+.2f}%, {ci_upper_rel*100:+.2f}%] одновременно накрывает и 0, и порог "
                    f"практической значимости ({effective_practical_threshold*100:.2f}%). Нужно больше данных."
                )
            else:
                decision_status = DecisionStatus.NEUTRAL_NO_SIGNIFICANT_EFFECT
                can_trust = True
                verdict_ru = (
                    f"⚪ СТАТИСТИЧЕСКИ ЗНАЧИМОГО ЭФФЕКТА НЕ ОБНАРУЖЕНО: Лучший вариант '{best_arm_id}' показал "
                    f"{rel_diff*100:+.2f}% (p-value = {primary_res['p_value']:.4f} >= {passport.alpha}), "
                    f"{ci_conf_pct}% ДИ [{ci_lower_rel*100:+.2f}%, {ci_upper_rel*100:+.2f}%] уже порога "
                    f"практической значимости ({effective_practical_threshold*100:.2f}%)."
                )

        if blinded:
            primary_res = {"blinded": True, "criterion": primary_res.get("criterion")}
            variant_comparisons = {t_id: {"blinded": True} for t_id in variant_comparisons}
            secondary_summaries = []
            winning_variant = None

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
            evidence_level=evidence.value,
            estimand=passport.estimand.value,
            generalization_population=passport.generalization_population or "",
            passport_version=passport.version,
            passport_hash=passport.passport_hash,
            analysis_day=analysis_day,
            blinded=blinded,
        )
