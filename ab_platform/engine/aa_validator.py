import math
from typing import Optional, Sequence
import numpy as np
import pandas as pd
from scipy import stats
from ab_platform.schemas.passport import MetricDefinition, MetricType


class AAValidationEngine:
    """
    A/A на боевом пути (ТЗ v4, PRE-2; принцип 6): назначение — тем же хэшем и правилом сплита,
    что и ``DeterministicAssignmentEngine.assign_user`` (новая соль на итерацию), анализ — той же
    функцией, что и в оркестраторе. ``assignment="bernoulli"`` — быстрый режим для интерфейса.
    Для ресторанов — ``run_cluster_aa_simulation`` через весь ``evaluate_experiment``.
    """

    @staticmethod
    def _fpr_summary(p_values: np.ndarray, alpha: float, iterations: int) -> dict:
        n_runs = int(len(p_values))
        if n_runs == 0:
            raise ValueError("Ни одна итерация A/A не дала валидного p-value; проверьте входные данные.")
        fpr = float(np.mean(p_values < alpha))
        margin = 1.96 * math.sqrt(alpha * (1.0 - alpha) / n_runs)
        lo, hi = alpha - margin, alpha + margin
        ks_stat, ks_p = stats.kstest(p_values, "uniform")
        in_range = bool(lo <= fpr <= hi)
        uniform = bool(ks_p > 0.01)
        return {
            "iterations_requested": iterations,
            "iterations_run": n_runs,
            "false_positive_rate": round(fpr, 4),
            "expected_alpha": alpha,
            "fpr_expected_ci_lower": round(lo, 5),
            "fpr_expected_ci_upper": round(hi, 5),
            "fpr_deviation_abs": round(abs(fpr - alpha), 4),
            "is_fpr_within_expected_range": in_range,
            "ks_statistic": float(ks_stat),
            "ks_pvalue": float(ks_p),
            "is_uniform_distribution": uniform,
            "is_valid": bool(in_range and uniform),
            "is_fpr_acceptable": in_range,
        }

    @staticmethod
    def run_aa_simulation(
        df: pd.DataFrame,
        metric_col: str,
        user_col: str = "user_id",
        iterations: int = 10000,
        alpha: float = 0.05,
        covariate_col: Optional[str] = None,
        cuped_min_r2: float = 0.05,
        seed: int = 42,
        assignment: str = "hash",
        metric_def: Optional[MetricDefinition] = None,
        control_split_pct: float = 50.0,
    ) -> dict:
        if metric_col not in df.columns:
            raise ValueError(f"Колонка '{metric_col}' отсутствует в датасете.")
        if user_col not in df.columns:
            raise ValueError(f"Колонка '{user_col}' отсутствует в датасете.")
        if covariate_col is not None and covariate_col not in df.columns:
            raise ValueError(f"Колонка ковариаты '{covariate_col}' отсутствует в датасете.")
        if iterations < 1:
            raise ValueError("iterations должен быть >= 1.")
        if assignment not in ("hash", "bernoulli"):
            raise ValueError("assignment: 'hash' (боевой хэш) или 'bernoulli'.")

        from ab_platform.engine.assignment import DeterministicAssignmentEngine as ASG
        from ab_platform.orchestrator import SmartABPlatformOrchestrator as ORCH

        cols = [user_col, metric_col] + ([covariate_col] if covariate_col else [])
        unit_df = df[cols].drop_duplicates(subset=[user_col]).reset_index(drop=True)
        n = len(unit_df)
        if n < 10:
            raise ValueError("Недостаточно уникальных единиц рандомизации для A/A симуляции (минимум 10).")
        if metric_def is None:
            vals = unit_df[metric_col].dropna().unique()
            is_binary = len(vals) <= 2 and set(np.asarray(vals, dtype=float)) <= {0.0, 1.0}
            metric_def = MetricDefinition(
                name=metric_col, metric_type=MetricType.BINARY if is_binary else MetricType.CONTINUOUS,
                numerator_col=metric_col, winsorize_percentile=None,
            )

        ids = unit_df[user_col].astype(str).tolist()
        rng = np.random.default_rng(seed)
        p_values = []
        cuped_applied_count = 0
        for i in range(iterations):
            if assignment == "hash":
                z = ASG.buckets_for(f"aa:{seed}:{i}", ids)
                is_control = ASG.control_from_bucket(z, control_split_pct)
            else:
                is_control = rng.random(n) * 100.0 < control_split_pct
            if is_control.sum() < 2 or (~is_control).sum() < 2:
                continue
            frame = unit_df.assign(variant=np.where(is_control, "control", "treatment"))
            res, cuped_diag = ORCH._evaluate_pair_metric(
                frame, metric_def, "control", "treatment", variant_col="variant",
                covariate_col=covariate_col, alpha=alpha, cuped_min_r2=cuped_min_r2,
            )
            if cuped_diag.get("applied"):
                cuped_applied_count += 1
            if not np.isnan(res["p_value"]):
                p_values.append(res["p_value"])

        out = AAValidationEngine._fpr_summary(np.asarray(p_values, dtype=float), alpha, iterations)
        out["assignment"] = assignment
        out["criterion"] = metric_def.metric_type.value
        out["cuped_applied_share"] = round(cuped_applied_count / iterations, 4) if covariate_col else None
        return out

    @staticmethod
    def run_cluster_aa_simulation(
        df: pd.DataFrame,
        passport,
        cluster_col: str,
        iterations: int = 2000,
        seed: int = 42,
        day_col: Optional[str] = "date",
    ) -> dict:
        """
        A/A ресторанного дизайна через весь ``evaluate_experiment`` (кластерный путь Паспорта):
        рестораны перераспределяются по рукам полной рандомизацией (внутри страт, если
        ``passport.stratification_cols``). Отрицательный контроль: на тех же назначениях считается
        запрещённый анализ — Welch по ресторан-дням как независимым наблюдениям (FPR раздут).
        """
        from ab_platform.orchestrator import SmartABPlatformOrchestrator as ORCH

        metric = passport.primary_metric.numerator_col
        for col in (cluster_col, metric):
            if col not in df.columns:
                raise ValueError(f"Колонка '{col}' отсутствует в датасете.")
        clusters = df[[cluster_col] + list(passport.stratification_cols)].drop_duplicates(cluster_col)
        if len(clusters) < 4:
            raise ValueError("Нужно минимум 4 кластера.")
        strata = (clusters[list(passport.stratification_cols)].astype(str).agg("|".join, axis=1).to_numpy()
                  if passport.stratification_cols else np.zeros(len(clusters)))
        share_t = 1.0 - next(v.traffic_percentage for v in passport.variants if v.is_control) / 100.0
        t_id = next(v.variant_id for v in passport.variants if not v.is_control)
        c_id = next(v.variant_id for v in passport.variants if v.is_control)
        rng = np.random.default_rng(seed)
        cl_ids = clusters[cluster_col].to_numpy()
        row_cluster = pd.Index(cl_ids).get_indexer(df[cluster_col])
        p_values, naive_p = [], []
        for _ in range(iterations):
            treat = np.zeros(len(cl_ids), dtype=bool)
            for s in np.unique(strata):
                idx = np.flatnonzero(strata == s)
                k = int(round(len(idx) * share_t))
                treat[rng.permutation(idx)[:k]] = True
            frame = df.assign(variant=np.where(treat[row_cluster], t_id, c_id))
            rep = ORCH.evaluate_experiment(passport, frame, user_col=cluster_col)
            p_values.append(rep.primary_metric_result["p_value"])
            y = frame[metric].to_numpy(dtype=float)
            is_t = treat[row_cluster]
            naive_p.append(stats.ttest_ind(y[is_t], y[~is_t], equal_var=False).pvalue)
        out = AAValidationEngine._fpr_summary(np.asarray(p_values, dtype=float), passport.alpha, iterations)
        out["criterion"] = rep.primary_metric_result.get("criterion")
        out["naive_storeday_fpr"] = round(float(np.mean(np.asarray(naive_p) < passport.alpha)), 4)
        out["n_clusters"] = int(len(cl_ids))
        return out

    @staticmethod
    def check_covariate_balance_smd(
        df: pd.DataFrame,
        covariate_cols: Sequence[str],
        variant_col: str = "variant",
        smd_threshold: float = 0.1,
    ) -> dict:
        for col in list(covariate_cols) + [variant_col]:
            if col not in df.columns:
                raise ValueError(f"Колонка '{col}' отсутствует в датасете для проверки баланса.")
        if len(covariate_cols) == 0:
            raise ValueError("covariate_cols не может быть пустым.")

        variants = sorted(df[variant_col].dropna().unique().tolist(), key=str)
        if len(variants) < 2:
            raise ValueError("Для проверки баланса SMD нужно минимум 2 варианта.")

        smd_by_covariate: dict = {}
        max_abs_smd = 0.0

        for cov in covariate_cols:
            pair_smd: dict = {}
            for i in range(len(variants)):
                for j in range(i + 1, len(variants)):
                    v1, v2 = variants[i], variants[j]
                    x1 = df.loc[df[variant_col] == v1, cov].to_numpy(dtype=float)
                    x2 = df.loc[df[variant_col] == v2, cov].to_numpy(dtype=float)
                    x1 = x1[~np.isnan(x1)]
                    x2 = x2[~np.isnan(x2)]
                    if len(x1) < 2 or len(x2) < 2:
                        continue
                    mean1, mean2 = float(np.mean(x1)), float(np.mean(x2))
                    var1, var2 = float(np.var(x1, ddof=1)), float(np.var(x2, ddof=1))
                    pooled_sd = math.sqrt((var1 + var2) / 2.0)
                    smd = (mean1 - mean2) / pooled_sd if pooled_sd > 1e-12 else 0.0
                    pair_smd[f"{v1}_vs_{v2}"] = round(float(smd), 4)
                    max_abs_smd = max(max_abs_smd, abs(smd))
            smd_by_covariate[cov] = pair_smd

        return {
            "smd_by_covariate": smd_by_covariate,
            "max_abs_smd": round(max_abs_smd, 4),
            "threshold": smd_threshold,
            "is_balanced": bool(max_abs_smd < smd_threshold),
        }


HistoricalAAValidator = AAValidationEngine
