import math
from unittest import mock

import numpy as np
import pandas as pd
import pytest

from ab_platform.schemas.passport import (
    Direction,
    ExperimentPassport,
    GuardrailDefinition,
    MetricDefinition,
    MetricType,
    VariantAllocation,
)
from ab_platform.engine.safeguards import ExperimentSafeguardEngine, GuardrailStatus
from ab_platform.engine.aa_validator import AAValidationEngine, HistoricalAAValidator
from ab_platform.orchestrator import DecisionStatus, SmartABPlatformOrchestrator


def _make_passport(n_guardrails=1, n_secondary=0, alpha=0.05, practical_threshold=0.0, n_variants=2):
    primary = MetricDefinition(name="revenue", metric_type=MetricType.CONTINUOUS, numerator_col="revenue")
    guardrails = [
        GuardrailDefinition(
            metric=MetricDefinition(
                name=f"latency_{i}",
                metric_type=MetricType.CONTINUOUS,
                numerator_col=f"latency_{i}",
                expected_direction=Direction.DECREASE,
            ),
            max_allowed_degradation_rel=0.02,
            tradeoff_rationale="perf guardrail",
        )
        for i in range(n_guardrails)
    ]
    secondary = [
        MetricDefinition(name=f"sec_{i}", metric_type=MetricType.CONTINUOUS, numerator_col=f"sec_{i}")
        for i in range(n_secondary)
    ]
    variants = [VariantAllocation(variant_id="control", traffic_percentage=100.0 / n_variants, is_control=True)]
    for i in range(1, n_variants):
        variants.append(VariantAllocation(variant_id=f"t{i}", traffic_percentage=100.0 / n_variants, is_control=False))

    return ExperimentPassport(
        experiment_id="exp_phase1",
        test_key="exp_phase1_key",
        hypothesis_statement="Если мы изменим X, то revenue вырастет, потому что гипотеза R",
        variants=variants,
        primary_metric=primary,
        guardrail_metrics=guardrails,
        secondary_metrics=secondary,
        alpha=alpha,
        use_cuped=False,
        practical_significance_threshold=practical_threshold,
        mde_relative=0.02,
        baseline_value=100.0,
        planned_duration_days=14,
        daily_eligible_units=1000,
    )


def _make_df(n=2000, seed=1, n_secondary=0):
    rng = np.random.default_rng(seed)
    uids = np.arange(n)
    variant = np.where(uids % 2 == 0, "control", "t1")
    data = {
        "user_id": uids,
        "variant": variant,
        "revenue": rng.normal(100, 10, n),
        "latency_0": rng.normal(200, 5, n),
    }
    for i in range(n_secondary):
        data[f"sec_{i}"] = rng.normal(1, 0.2, n)
    return pd.DataFrame(data)


def _fake_pair_metric(scenario):
    def fake(cls, df_pair, metric_def, control_id, treatment_id, variant_col="variant",
             covariate_col=None, use_lins_method=False, alpha=0.05, quantile_target=0.9, cuped_min_r2=0.05):
        cfg = scenario[metric_def.name]
        stat_res = dict(cfg)
        stat_res.setdefault("criterion", "welch_t_test")
        cuped_diag = {
            "applied": False, "theta": 0.0, "correlation": 0.0,
            "variance_reduction_pct": 0.0, "sensitivity_gain_factor": 1.0,
        }
        return stat_res, cuped_diag
    return classmethod(fake)


class TestSRMChi2:
    def test_small_shift_detected_at_large_n(self):
        result = ExperimentSafeguardEngine.detect_srm_chi2(497000, 503000, 0.5, 0.001)
        assert result["srm_detected"] is True
        assert result["chi2_pvalue"] < 0.001

    def test_healthy_split_not_flagged(self):
        result = ExperimentSafeguardEngine.detect_srm_chi2(50000, 50010, 0.5, 0.001)
        assert result["srm_detected"] is False

    def test_min_detectable_srm_shift_present_and_shrinks_with_n(self):
        shift_small_n = ExperimentSafeguardEngine.min_detectable_srm_shift(10000, 0.5, 0.001)
        shift_large_n = ExperimentSafeguardEngine.min_detectable_srm_shift(10_000_000, 0.5, 0.001)
        assert shift_large_n < shift_small_n

    def test_diagnose_srm_stage_separates_assigned_and_triggered(self):
        diag = ExperimentSafeguardEngine.diagnose_srm_stage(500000, 500000, 497000, 503000, alpha_srm=0.001)
        assert diag["assigned_srm"] is False
        assert diag["triggered_srm"] is True
        assert "min_detectable_srm_shift" in diag


class TestGuardrailNonInferiority:
    def _eval(self, degradation_rel, se_rel, delta=0.02, baseline=100.0):
        point_diff = -degradation_rel * baseline
        se_abs = se_rel * baseline
        return ExperimentSafeguardEngine.evaluate_guardrail_breach(
            baseline_control=baseline,
            treatment_value=baseline + point_diff,
            ci_lower_diff=point_diff - 1.96 * se_abs,
            ci_upper_diff=point_diff + 1.96 * se_abs,
            max_allowed_degradation_rel=delta,
            higher_is_better=True,
        )

    def test_safe_proven(self):
        res = self._eval(degradation_rel=0.001, se_rel=0.002)
        assert res["status"] == GuardrailStatus.SAFE_PROVEN
        assert res["guardrail_breached"] is False

    def test_breach_proven(self):
        res = self._eval(degradation_rel=0.05, se_rel=0.005)
        assert res["status"] == GuardrailStatus.BREACH_PROVEN
        assert res["guardrail_breached"] is True

    def test_inconclusive_low_power(self):
        res = self._eval(degradation_rel=0.02, se_rel=0.02)
        assert res["status"] == GuardrailStatus.INCONCLUSIVE_LOW_POWER
        assert res["guardrail_breached"] is False

    def test_holm_correction_controls_family(self):
        rejects = ExperimentSafeguardEngine.apply_holm_to_breach_signals([0.001, 0.04, 0.3], alpha=0.05)
        assert rejects == [True, False, False]


class TestOrchestratorDecisionStatuses:
    def test_ship_with_dynamic_ci_text(self):
        passport99 = _make_passport(alpha=0.01)
        df = _make_df()
        scenario = {
            "revenue": dict(mean_control=100.0, mean_treatment=103.0, relative_diff=0.03,
                             p_value=0.001, ci_lower=1.0, ci_upper=5.0, is_significant=True),
            "latency_0": dict(mean_control=200.0, mean_treatment=199.5, relative_diff=-0.0025,
                               p_value=0.4, ci_lower=-3.0, ci_upper=2.0, is_significant=False),
        }
        with mock.patch.object(SmartABPlatformOrchestrator, "_evaluate_pair_metric", _fake_pair_metric(scenario)):
            report = SmartABPlatformOrchestrator.evaluate_experiment(passport99, df)
        assert report.decision_status == DecisionStatus.SHIP_TO_PRODUCTION
        assert "99% ДИ" in report.executive_summary_ru

    def test_guardrail_inconclusive_blocks_auto_ship(self):
        passport = _make_passport()
        df = _make_df()
        scenario = {
            "revenue": dict(mean_control=100.0, mean_treatment=103.0, relative_diff=0.03,
                             p_value=0.001, ci_lower=1.0, ci_upper=5.0, is_significant=True),
            "latency_0": dict(mean_control=100.0, mean_treatment=103.0, relative_diff=0.03,
                               p_value=0.2, ci_lower=-1.0, ci_upper=7.0, is_significant=False),
        }
        with mock.patch.object(SmartABPlatformOrchestrator, "_evaluate_pair_metric", _fake_pair_metric(scenario)):
            report = SmartABPlatformOrchestrator.evaluate_experiment(passport, df)
        assert report.decision_status == DecisionStatus.HOLD_HUMAN_REVIEW_GUARDRAIL_UNCERTAIN
        assert report.winning_variant is None
        assert any(g.status == GuardrailStatus.INCONCLUSIVE_LOW_POWER.value for g in report.guardrail_results)

    def test_guardrail_breach_after_holm_blocks_ship(self):
        passport = _make_passport()
        df = _make_df()
        scenario = {
            "revenue": dict(mean_control=100.0, mean_treatment=103.0, relative_diff=0.03,
                             p_value=0.001, ci_lower=1.0, ci_upper=5.0, is_significant=True),
            "latency_0": dict(mean_control=100.0, mean_treatment=110.0, relative_diff=0.10,
                               p_value=0.001, ci_lower=8.0, ci_upper=12.0, is_significant=True),
        }
        with mock.patch.object(SmartABPlatformOrchestrator, "_evaluate_pair_metric", _fake_pair_metric(scenario)):
            report = SmartABPlatformOrchestrator.evaluate_experiment(passport, df)
        assert report.decision_status == DecisionStatus.DO_NOT_SHIP_GUARDRAIL_BREACH
        assert report.decision_status == "DO_NOT_SHIP_GUARDRAIL_BREACH"

    def test_neutral_vs_inconclusive_split(self):
        passport = _make_passport()
        df = _make_df()
        guardrail_ok = dict(mean_control=200.0, mean_treatment=199.5, relative_diff=-0.0025,
                             p_value=0.4, ci_lower=-3.0, ci_upper=2.0, is_significant=False)

        scenario_neutral = {
            "revenue": dict(mean_control=100.0, mean_treatment=100.2, relative_diff=0.002,
                             p_value=0.6, ci_lower=-0.5, ci_upper=0.9, is_significant=False),
            "latency_0": guardrail_ok,
        }
        with mock.patch.object(SmartABPlatformOrchestrator, "_evaluate_pair_metric", _fake_pair_metric(scenario_neutral)):
            report_n = SmartABPlatformOrchestrator.evaluate_experiment(passport, df)
        assert report_n.decision_status == DecisionStatus.NEUTRAL_NO_SIGNIFICANT_EFFECT

        scenario_inconclusive = {
            "revenue": dict(mean_control=100.0, mean_treatment=101.0, relative_diff=0.01,
                             p_value=0.3, ci_lower=-1.0, ci_upper=5.0, is_significant=False),
            "latency_0": guardrail_ok,
        }
        with mock.patch.object(SmartABPlatformOrchestrator, "_evaluate_pair_metric", _fake_pair_metric(scenario_inconclusive)):
            report_i = SmartABPlatformOrchestrator.evaluate_experiment(passport, df)
        assert report_i.decision_status == DecisionStatus.INCONCLUSIVE_NEEDS_MORE_DATA

    def test_secondary_bh_is_family_wide_across_arms(self):
        passport = _make_passport(n_secondary=2, n_variants=3)
        rng = np.random.default_rng(5)
        n = 4500
        uids = np.arange(n)
        variant3 = np.array(["control", "t1", "t2"])[uids % 3]
        df = pd.DataFrame({
            "user_id": uids, "variant": variant3,
            "revenue": rng.normal(100, 10, n), "latency_0": rng.normal(200, 5, n),
            "sec_0": rng.normal(1, 0.2, n), "sec_1": rng.normal(1, 0.2, n),
        })
        scenario = {
            "revenue": dict(mean_control=100.0, mean_treatment=103.0, relative_diff=0.03,
                             p_value=0.001, ci_lower=1.0, ci_upper=5.0, is_significant=True),
            "latency_0": dict(mean_control=200.0, mean_treatment=199.5, relative_diff=-0.0025,
                               p_value=0.4, ci_lower=-3.0, ci_upper=2.0, is_significant=False),
            "sec_0": dict(mean_control=1.0, mean_treatment=1.2, relative_diff=0.2,
                          p_value=0.03, ci_lower=0.01, ci_upper=0.3, is_significant=True),
            "sec_1": dict(mean_control=1.0, mean_treatment=1.01, relative_diff=0.01,
                          p_value=0.8, ci_lower=-0.1, ci_upper=0.1, is_significant=False),
        }
        from ab_platform.engine import criterion_router as cr
        real_fdr = cr.StatCriterionRouter.apply_fdr_correction
        calls = []
        def counting(pvals, alpha=0.05):
            calls.append(len(pvals))
            return real_fdr(pvals, alpha=alpha)
        with mock.patch.object(cr.StatCriterionRouter, "apply_fdr_correction", staticmethod(counting)):
            with mock.patch.object(SmartABPlatformOrchestrator, "_evaluate_pair_metric", _fake_pair_metric(scenario)):
                report = SmartABPlatformOrchestrator.evaluate_experiment(passport, df)
        assert calls == [4]
        assert len(report.secondary_results) == 4

    def test_srm_rechecked_after_hygiene_removal(self):
        passport = _make_passport()
        rng = np.random.default_rng(3)
        n_single = 90000
        single_ids = np.arange(n_single)
        single_variant = np.where(single_ids % 2 == 0, "control", "t1")
        df_single = pd.DataFrame({
            "user_id": single_ids, "variant": single_variant,
            "revenue": rng.normal(100, 10, n_single), "latency_0": rng.normal(200, 5, n_single),
        })
        n_dup = 20000
        dup_ids = np.arange(n_single, n_single + n_dup)
        dup_control = pd.DataFrame({"user_id": dup_ids, "variant": "control",
                                     "revenue": rng.normal(100, 10, n_dup), "latency_0": rng.normal(200, 5, n_dup)})
        dup_treat = pd.DataFrame({"user_id": dup_ids, "variant": "t1",
                                   "revenue": rng.normal(100, 10, n_dup), "latency_0": rng.normal(200, 5, n_dup)})
        n_extra = 1000
        extra_ids = np.arange(n_single + n_dup, n_single + n_dup + n_extra)
        extra_rows = pd.DataFrame({"user_id": extra_ids, "variant": "control",
                                    "revenue": rng.normal(100, 10, n_extra), "latency_0": rng.normal(200, 5, n_extra)})
        df_full = pd.concat([df_single, dup_control, dup_treat, extra_rows], ignore_index=True)

        scenario = {
            "revenue": dict(mean_control=100.0, mean_treatment=103.0, relative_diff=0.03,
                             p_value=0.001, ci_lower=1.0, ci_upper=5.0, is_significant=True),
            "latency_0": dict(mean_control=200.0, mean_treatment=199.5, relative_diff=-0.0025,
                               p_value=0.4, ci_lower=-3.0, ci_upper=2.0, is_significant=False),
        }
        with mock.patch.object(SmartABPlatformOrchestrator, "_evaluate_pair_metric", _fake_pair_metric(scenario)):
            report = SmartABPlatformOrchestrator.evaluate_experiment(passport, df_full)

        recheck = report.srm_diagnosis["post_hygiene_recheck"]
        assert recheck["raw_triggered_srm"] is False
        assert recheck["clean_triggered_srm"] is True
        assert recheck["hygiene_introduced_imbalance"] is True
        assert report.decision_status == DecisionStatus.INVALID_EXPERIMENT_SRM_OR_CONTAMINATION
        assert report.can_trust_data is False

    def test_passport_covariate_default_missing_does_not_crash(self):
        passport = _make_passport()
        object.__setattr__(passport, "use_cuped", True)
        df = _make_df()
        assert "pre_revenue" not in df.columns
        scenario = {
            "revenue": dict(mean_control=100.0, mean_treatment=103.0, relative_diff=0.03,
                             p_value=0.001, ci_lower=1.0, ci_upper=5.0, is_significant=True),
            "latency_0": dict(mean_control=200.0, mean_treatment=199.5, relative_diff=-0.0025,
                               p_value=0.4, ci_lower=-3.0, ci_upper=2.0, is_significant=False),
        }
        with mock.patch.object(SmartABPlatformOrchestrator, "_evaluate_pair_metric", _fake_pair_metric(scenario)):
            report = SmartABPlatformOrchestrator.evaluate_experiment(passport, df)
        assert report.decision_status == DecisionStatus.SHIP_TO_PRODUCTION

    def test_explicit_missing_covariate_col_still_raises(self):
        passport = _make_passport()
        df = _make_df()
        with pytest.raises(ValueError):
            SmartABPlatformOrchestrator.evaluate_experiment(passport, df, covariate_col="definitely_missing")


class TestAAValidatorWithCuped:
    def test_historical_aa_validator_alias_exported(self):
        assert HistoricalAAValidator is AAValidationEngine

    def test_aa_simulation_fpr_within_range(self):
        rng = np.random.default_rng(7)
        n = 3000
        df = pd.DataFrame({"user_id": np.arange(n), "revenue": rng.normal(100, 20, n)})
        res = AAValidationEngine.run_aa_simulation(df, "revenue", user_col="user_id", iterations=2000, alpha=0.05, seed=7)
        assert res["is_fpr_within_expected_range"] is True
        assert res["is_uniform_distribution"] is True

    def test_aa_simulation_with_cuped_covariate(self):
        rng = np.random.default_rng(7)
        n = 3000
        revenue = rng.normal(100, 20, n)
        pre_revenue = revenue * 0.6 + rng.normal(0, 10, n)
        df = pd.DataFrame({"user_id": np.arange(n), "revenue": revenue, "pre_revenue": pre_revenue})
        res = AAValidationEngine.run_aa_simulation(
            df, "revenue", user_col="user_id", iterations=1500, alpha=0.05,
            covariate_col="pre_revenue", cuped_min_r2=0.05, seed=7,
        )
        assert res["cuped_applied_share"] is not None
        assert res["cuped_applied_share"] > 0.9

    def test_covariate_balance_smd_flags_imbalance(self):
        rng = np.random.default_rng(9)
        n = 2000
        variant = rng.choice(["control", "treatment"], n)
        pre = rng.normal(100, 10, n)
        df_balanced = pd.DataFrame({"variant": variant, "pre_revenue": pre})
        res_balanced = AAValidationEngine.check_covariate_balance_smd(df_balanced, ["pre_revenue"], "variant")
        assert res_balanced["is_balanced"] is True

        df_unbalanced = df_balanced.copy()
        df_unbalanced.loc[df_unbalanced["variant"] == "treatment", "pre_revenue"] += 50
        res_unbalanced = AAValidationEngine.check_covariate_balance_smd(df_unbalanced, ["pre_revenue"], "variant")
        assert res_unbalanced["is_balanced"] is False
