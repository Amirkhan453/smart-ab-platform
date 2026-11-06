"""
Кластерный инференс для ресторанного RCT (ТЗ v4, 5.5; STA-2, RST-6).

Принцип 8: единица инференса = единица рандомизации. Строки (ресторан-дни) сначала
агрегируются до кластера, затем:

- ``lin_cr2`` — регрессия Lin (treat, центрированные ковариаты, treat × ковариаты,
  индикаторы пар / страт) с CR2-ковариацией (Bell–McCaffrey) и степенями свободы
  Imbens–Kolesár; estimand ``per_restaurant_mean``;
- ``randomization_test`` — перестановочный тест по боевой схеме (перестановки внутри пар
  и страт, точный перебор знаков для ≤ 13 пар), CUPED с θ, не зависящим от назначения;
  ДИ — обращением теста при постоянном аддитивном эффекте;
- ``wild_cluster_bootstrap`` — WCR-bootstrap t-статистики (Rademacher; Webb при < 12 кластерах);
- ``network_total_ratio`` — относительный эффект на суммарную выручку (отношение сумм,
  с поправкой на пре-период), дельта-метод на кластерных суммах;
- ``analyze_cluster_rct`` — выбор метода по Паспорту.

Ссылки: Lin (2013); Bell, McCaffrey (2002); Pustejovsky, Tipton (2018); Imbens, Kolesár (2016);
Cameron, Gelbach, Miller (2008); Webb (2014).
"""
from __future__ import annotations

import itertools
from typing import Dict, Literal, Optional, Sequence

import numpy as np
import pandas as pd
from scipy import stats

Sidedness = Literal["two_sided", "one_sided"]
DirectionT = Literal["increase", "decrease"]

EXACT_PAIR_LIMIT = 13          # 2^13 = 8192 перестановок знаков — точный перебор
WEBB_CLUSTER_LIMIT = 12        # при меньшем числе кластеров — 6-точечные веса Webb
WILD_MIN_PER_ARM = 10          # ниже — предупреждение: рандомизационный тест точнее (замер фазы 1)


# ---------------------------------------------------------------- агрегация
def aggregate_clusters(
    df: pd.DataFrame,
    cluster_col: str,
    arm_col: str,
    y_col: str,
    treatment_value,
    covariate_col: Optional[str] = None,
    block_col: Optional[str] = None,
    se_cluster_col: Optional[str] = None,
    how: Literal["mean", "sum"] = "mean",
) -> pd.DataFrame:
    """
    Ресторан-дни → строка на кластер: ``y`` (среднее или сумма за период), ``x`` (ковариата
    пре-периода, должна быть постоянной в кластере), ``treat`` (0/1), ``block``, ``se_cluster``.
    Ошибка, если рука или ковариата не постоянны внутри кластера.
    """
    need = [cluster_col, arm_col, y_col] + [c for c in (covariate_col, block_col, se_cluster_col) if c]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise ValueError(f"Нет колонок {missing}")
    if df[y_col].isna().any():
        raise ValueError(f"В '{y_col}' есть пропуски: задайте правило их обработки до анализа")
    g = df.groupby(cluster_col, sort=True)
    if (g[arm_col].nunique() > 1).any():
        raise ValueError("Рука меняется внутри кластера: единица рандомизации не совпадает с кластером")
    out = pd.DataFrame({
        "cluster": g.size().index,
        "y": (g[y_col].mean() if how == "mean" else g[y_col].sum()).to_numpy(dtype=float),
        "n_rows": g.size().to_numpy(),
        "treat": (g[arm_col].first() == treatment_value).astype(int).to_numpy(),
    })
    if covariate_col:
        if (g[covariate_col].nunique(dropna=False) > 1).any():
            raise ValueError("Ковариата пре-периода должна быть постоянной внутри кластера")
        out["x"] = g[covariate_col].first().to_numpy(dtype=float)
        if out["x"].isna().any():
            raise ValueError("Пропуски в ковариате пре-периода")
    if block_col:
        out["block"] = g[block_col].first().to_numpy()
    if se_cluster_col:
        out["se_cluster"] = g[se_cluster_col].first().to_numpy()
    return out


# ---------------------------------------------------------------- помощники
def _side_p(t: float, df: Optional[float], sidedness: Sidedness, direction: DirectionT) -> float:
    dist = stats.t(df) if df is not None and np.isfinite(df) else stats.norm
    if sidedness == "one_sided":
        return float(dist.sf(t if direction == "increase" else -t))
    return float(2.0 * dist.sf(abs(t)))


def _crit(alpha: float, df: Optional[float], sidedness: Sidedness) -> float:
    q = 1.0 - alpha / 2.0 if sidedness == "two_sided" else 1.0 - alpha
    dist = stats.t(df) if df is not None and np.isfinite(df) else stats.norm
    return float(dist.ppf(q))


def _check(sidedness: str, direction: str, alpha: float) -> None:
    if sidedness not in ("two_sided", "one_sided"):
        raise ValueError("sidedness: 'two_sided' или 'one_sided'")
    if direction not in ("increase", "decrease"):
        raise ValueError("direction: 'increase' или 'decrease'")
    if not 0 < alpha < 0.5:
        raise ValueError("alpha в (0, 0.5)")


def _block_codes(blocks: Optional[Sequence]) -> Optional[np.ndarray]:
    if blocks is None:
        return None
    return pd.factorize(pd.Series(blocks), sort=True)[0]


def lin_design(
    treat: np.ndarray, covariates: Optional[np.ndarray] = None, blocks: Optional[Sequence] = None,
) -> np.ndarray:
    """Матрица Lin: [1, treat, Xc, treat·Xc, индикаторы блоков без первого]; коэффициент treat — ATE."""
    t = np.asarray(treat, dtype=float)
    cols = [np.ones_like(t), t]
    if covariates is not None:
        X = np.asarray(covariates, dtype=float).reshape(len(t), -1)
        Xc = X - X.mean(axis=0)
        cols += [Xc, t[:, None] * Xc]
    codes = _block_codes(blocks)
    if codes is not None and codes.max() > 0:
        cols.append((codes[:, None] == np.arange(1, codes.max() + 1)[None, :]).astype(float))
    return np.column_stack([c if c.ndim == 2 else c[:, None] for c in cols])


def _inv_sqrt_psd(M: np.ndarray, tol: float = 1e-10) -> np.ndarray:
    w, V = np.linalg.eigh((M + M.T) / 2.0)
    inv = np.where(w > tol, 1.0 / np.sqrt(np.clip(w, tol, None)), 0.0)
    return (V * inv) @ V.T


# ---------------------------------------------------------------- CR2
def ols_cr2(
    y: np.ndarray, X: np.ndarray, cluster_ids: Optional[Sequence] = None, coef_index: int = 1,
) -> Dict[str, float]:
    """
    OLS с CR2-дисперсией (Bell–McCaffrey) для коэффициента ``coef_index`` и степенями
    свободы Imbens–Kolesár (2016). ``cluster_ids=None`` — каждая строка свой кластер (HC2).
    """
    y = np.asarray(y, dtype=float)
    X = np.asarray(X, dtype=float)
    n, k = X.shape
    if n <= k:
        raise ValueError(f"Наблюдений ({n}) не больше числа параметров ({k})")
    XtX_inv = np.linalg.pinv(X.T @ X)
    beta = XtX_inv @ X.T @ y
    e = y - X @ beta
    ids = np.arange(n) if cluster_ids is None else pd.factorize(pd.Series(cluster_ids))[0]
    G = int(ids.max()) + 1
    if G < 2:
        raise ValueError("Нужно минимум 2 кластера")
    c = np.zeros(k)
    c[coef_index] = 1.0
    a = XtX_inv @ c                                  # k
    H = X @ XtX_inv @ X.T
    IminusH = np.eye(n) - H
    Gmat = np.zeros((n, G))
    var = 0.0
    for g in range(G):
        idx = np.flatnonzero(ids == g)
        Xg = X[idx]
        A = _inv_sqrt_psd(np.eye(len(idx)) - H[np.ix_(idx, idx)])
        u = A @ (Xg @ a)                             # n_g: веса остатков кластера в оценке
        var += float(u @ e[idx]) ** 2
        Gmat[:, g] = IminusH[:, idx] @ u
    GtG = Gmat.T @ Gmat
    tr = float(np.trace(GtG))
    df = tr ** 2 / float(np.sum(GtG ** 2)) if tr > 0 else float("nan")
    se = float(np.sqrt(max(var, 0.0)))
    return {"estimate": float(beta[coef_index]), "se": se, "df": float(df), "n": n, "n_clusters": G,
            "beta": beta, "residuals": e}


def lin_cr2(
    y: np.ndarray, treat: np.ndarray, covariates: Optional[np.ndarray] = None, blocks: Optional[Sequence] = None,
    se_clusters: Optional[Sequence] = None, alpha: float = 0.05, sidedness: Sidedness = "two_sided",
    direction: DirectionT = "increase",
) -> Dict[str, float]:
    """Эффект Lin (ATE на кластер) с CR2 и t-распределением с df Imbens–Kolesár."""
    _check(sidedness, direction, alpha)
    treat = np.asarray(treat, dtype=int)
    if treat.min() == treat.max():
        raise ValueError("Нужны обе руки")
    X = lin_design(treat, covariates, blocks)
    r = ols_cr2(y, X, se_clusters, coef_index=1)
    t = r["estimate"] / r["se"] if r["se"] > 0 else 0.0
    q = _crit(alpha, r["df"], sidedness)
    return {
        "criterion": "lin_cr2_bell_mccaffrey", "estimate": r["estimate"], "se": r["se"], "df": r["df"],
        "statistic": t, "p_value": _side_p(t, r["df"], sidedness, direction),
        "ci_lower": r["estimate"] - q * r["se"], "ci_upper": r["estimate"] + q * r["se"],
        "n_clusters": r["n_clusters"],
    }


# ---------------------------------------------------------------- рандомизационный тест
def _permutation_assignments(treat: np.ndarray, codes: np.ndarray, n_perm: int, rng: np.random.Generator) -> np.ndarray:
    """Случайные назначения с тем же числом тестовых в каждом блоке (полная рандомизация внутри блока)."""
    n = len(treat)
    order_blocks = np.argsort(codes, kind="stable")
    sorted_codes = codes[order_blocks]
    starts = np.searchsorted(sorted_codes, np.arange(codes.max() + 1))
    n_t = np.bincount(codes, weights=treat, minlength=codes.max() + 1).astype(int)
    pos = np.arange(n) - starts[sorted_codes]
    pattern = (pos < n_t[sorted_codes]).astype(np.int8)       # для строк, отсортированных по блоку
    keys = codes[None, :] + rng.random((n_perm, n))
    order = np.argsort(keys, axis=1)
    T = np.empty((n_perm, n), dtype=np.int8)
    np.put_along_axis(T, order, np.broadcast_to(pattern, (n_perm, n)), axis=1)
    return T


def _block_contrast(treat: np.ndarray, codes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Стратифицированная разность средних — линейная функция назначения при фиксированных
    числах тестовых в блоках: stat(y; T) = (T * a) @ y + b @ y, где a, b не зависят от T.
    """
    B = codes.max() + 1
    n_b = np.bincount(codes, minlength=B).astype(float)
    n_t = np.bincount(codes, weights=treat, minlength=B).astype(float)
    n_c = n_b - n_t
    if np.any(n_t == 0) or np.any(n_c == 0):
        raise ValueError("В каждом блоке нужны обе руки")
    w = n_b / n_b.sum()
    a = (w / n_t + w / n_c)[codes]
    b = -(w / n_c)[codes]
    return a, b


def randomization_test(
    y: np.ndarray, treat: np.ndarray, covariate: Optional[np.ndarray] = None, blocks: Optional[Sequence] = None,
    n_perm: int = 10_000, alpha: float = 0.05, sidedness: Sidedness = "two_sided",
    direction: DirectionT = "increase", seed: int = 0, ci: bool = True,
) -> Dict[str, float]:
    """
    Перестановочный тест по схеме рандомизации: блоки (пары / страты) с фиксированным числом
    тестовых. Статистика — стратифицированная разность средних y − θ(x − x̄), где θ оценён по
    регрессии y ~ x + блоки без руки (не зависит от назначения, тест точен при нулевой гипотезе).
    Для пар (все блоки размера 2) при ≤ 13 парах — точный перебор знаков.
    """
    _check(sidedness, direction, alpha)
    y = np.asarray(y, dtype=float)
    treat = np.asarray(treat, dtype=int)
    n = len(y)
    if n < 4:
        raise ValueError("Слишком мало кластеров для рандомизационного теста")
    codes = _block_codes(blocks) if blocks is not None else np.zeros(n, dtype=int)
    y_adj = y.copy()
    if covariate is not None:
        Z = lin_design(np.zeros(n), covariate, blocks if blocks is not None else None)
        Z = np.delete(Z, 1, axis=1)                               # без колонки руки
        coef = np.linalg.lstsq(Z, y, rcond=None)[0]
        k_x = np.asarray(covariate, dtype=float).reshape(n, -1).shape[1]
        Xc = np.asarray(covariate, dtype=float).reshape(n, -1)
        Xc = Xc - Xc.mean(axis=0)
        y_adj = y - Xc @ coef[1:1 + k_x]
    a, b = _block_contrast(treat, codes)
    s_obs = float((treat * a) @ y_adj + b @ y_adj)

    sizes = np.bincount(codes)
    pairs = bool(np.all(sizes == 2))
    k = len(sizes)
    if pairs and k <= EXACT_PAIR_LIMIT:
        # Точный перебор: в каждой паре тестовым может быть любой из двух ресторанов.
        order = np.argsort(codes, kind="stable").reshape(k, 2)
        combos = np.array(list(itertools.product((0, 1), repeat=k)), dtype=np.int8)
        T = np.zeros((len(combos), n), dtype=np.int8)
        rows = np.arange(len(combos))[:, None]
        T[rows, order[np.arange(k), combos]] = 1
        exact = True
    else:
        rng = np.random.default_rng(seed)
        T = _permutation_assignments(treat, codes, n_perm, rng)
        exact = False

    s_y = (T * a) @ y_adj + b @ y_adj
    s_t = (T * a) @ treat + b @ treat          # статистика наблюдаемого назначения при перестановке T

    def pval(tau: float) -> float:
        perm = s_y - tau * s_t
        obs = s_obs - tau
        if sidedness == "one_sided":
            sign = 1.0 if direction == "increase" else -1.0
            hits = np.sum(sign * perm >= sign * obs - 1e-12)
        else:
            hits = np.sum(np.abs(perm) >= abs(obs) - 1e-12)
        return float(hits / len(perm)) if exact else float((hits + 1) / (len(perm) + 1))

    p = pval(0.0)
    lo = hi = float("nan")
    if ci:
        spread = float(np.std(s_y)) or 1.0
        lo = _invert(pval, s_obs, -1, alpha, spread) if not (sidedness == "one_sided" and direction == "decrease") else -np.inf
        hi = _invert(pval, s_obs, +1, alpha, spread) if not (sidedness == "one_sided" and direction == "increase") else np.inf
    return {
        "criterion": "randomization_test_exact" if exact else "randomization_test_mc", "estimate": s_obs,
        "p_value": p, "ci_lower": lo, "ci_upper": hi, "n_permutations": int(len(s_y)), "exact": exact,
        "min_attainable_p": (2.0 / 2 ** k if pairs and sidedness == "two_sided" else 1.0 / 2 ** k) if exact else 1.0 / (len(s_y) + 1),
        "n_clusters": n,
    }


def _invert(pval, est: float, side: int, alpha: float, spread: float, iters: int = 60) -> float:
    """Граница ДИ: самое дальнее tau в сторону ``side``, при котором тест ещё не отвергает H0: эффект = tau."""
    inside, step = est, spread
    outside = est + side * step
    expand = 0
    while pval(outside) >= alpha and expand < 40:
        inside, outside = outside, outside + side * step * (2 ** expand)
        expand += 1
    for _ in range(iters):
        mid = (inside + outside) / 2.0
        if pval(mid) >= alpha:
            inside = mid
        else:
            outside = mid
    return float(inside)


# ---------------------------------------------------------------- wild cluster bootstrap
def wild_cluster_bootstrap(
    y: np.ndarray, X: np.ndarray, cluster_ids: Optional[Sequence] = None, coef_index: int = 1,
    n_boot: int = 9_999, alpha: float = 0.05, sidedness: Sidedness = "two_sided",
    direction: DirectionT = "increase", seed: int = 0, weights: Optional[str] = None,
) -> Dict[str, float]:
    """
    WCR wild cluster bootstrap-t (Cameron, Gelbach, Miller 2008): остатки модели с ограничением
    β = 0 умножаются на веса кластера. t-статистика — с CR2 (поправки A_g не зависят от y,
    поэтому векторизуется): на ресторанной панели с 8–10 кластерами на руку CR1-вариант
    завышал FPR до 8%. Веса Rademacher; при < 12 кластерах — Webb.
    """
    _check(sidedness, direction, alpha)
    y = np.asarray(y, dtype=float)
    X = np.asarray(X, dtype=float)
    n, k = X.shape
    ids = np.arange(n) if cluster_ids is None else pd.factorize(pd.Series(cluster_ids))[0]
    G = int(ids.max()) + 1
    if G < 4:
        raise ValueError("Wild bootstrap требует минимум 4 кластера")
    weights = weights or ("webb" if G < WEBB_CLUSTER_LIMIT else "rademacher")
    XtX_inv = np.linalg.pinv(X.T @ X)
    a = XtX_inv[:, coef_index]
    H = X @ XtX_inv @ X.T
    w = np.zeros(n)                                   # CR2-веса остатков: u_g = A_g X_g a
    for g in range(G):
        idx = np.flatnonzero(ids == g)
        w[idx] = _inv_sqrt_psd(np.eye(len(idx)) - H[np.ix_(idx, idx)]) @ (X[idx] @ a)
    S = np.zeros((G, n))
    S[ids, np.arange(n)] = 1.0                        # агрегатор по кластерам

    def t_stats(Y: np.ndarray) -> np.ndarray:        # Y: n × B
        beta = XtX_inv @ X.T @ Y
        E = Y - X @ beta
        scores = S @ (w[:, None] * E)                 # G × B
        se = np.sqrt(np.sum(scores ** 2, axis=0))
        return beta[coef_index] / np.where(se > 0, se, np.inf)

    t_obs = float(t_stats(y[:, None])[0])
    Xr = np.delete(X, coef_index, axis=1)
    beta_r = np.linalg.lstsq(Xr, y, rcond=None)[0]
    fit_r, u_r = Xr @ beta_r, y - Xr @ beta_r
    rng = np.random.default_rng(seed)
    t_boot = []
    for start in range(0, n_boot, 2000):
        B = min(2000, n_boot - start)
        if weights == "webb":
            vals = np.array([-np.sqrt(1.5), -1.0, -np.sqrt(0.5), np.sqrt(0.5), 1.0, np.sqrt(1.5)])
            V = vals[rng.integers(0, 6, (G, B))]
        else:
            V = rng.choice((-1.0, 1.0), size=(G, B))
        Ystar = fit_r[:, None] + u_r[:, None] * V[ids]
        t_boot.append(t_stats(Ystar))
    t_boot = np.concatenate(t_boot)
    if sidedness == "one_sided":
        sign = 1.0 if direction == "increase" else -1.0
        p = float((np.sum(sign * t_boot >= sign * t_obs) + 1) / (len(t_boot) + 1))
    else:
        p = float((np.sum(np.abs(t_boot) >= abs(t_obs)) + 1) / (len(t_boot) + 1))
    return {"criterion": f"wild_cluster_bootstrap_{weights}", "statistic": t_obs, "p_value": p,
            "n_boot": int(len(t_boot)), "n_clusters": G}


# ---------------------------------------------------------------- network_total
def network_total_ratio(
    y: np.ndarray, treat: np.ndarray, x: Optional[np.ndarray] = None, alpha: float = 0.05,
    sidedness: Sidedness = "two_sided", direction: DirectionT = "increase",
) -> Dict[str, float]:
    """
    Относительный эффект на суммарный объём (estimand ``network_total``, веса — объём):
    R = (Σ_T y / Σ_T x) / (Σ_C y / Σ_C x) − 1 (с пре-периодом) или ȳ_T / ȳ_C − 1 (без него).
    SE — дельта-метод для log R на кластерных суммах.
    """
    _check(sidedness, direction, alpha)
    y = np.asarray(y, dtype=float)
    t = np.asarray(treat, dtype=int).astype(bool)
    xv = np.ones_like(y) if x is None else np.asarray(x, dtype=float)
    if np.any(xv <= 0) or np.any(y < 0):
        raise ValueError("network_total: объёмы должны быть неотрицательны, пре-период > 0")

    def log_ratio_var(yy: np.ndarray, xx: np.ndarray) -> tuple[float, float]:
        m = len(yy)
        my, mx = yy.mean(), xx.mean()
        if my <= 0:
            raise ValueError("Нулевой объём в руке")
        vy, vx, cxy = yy.var(ddof=1), xx.var(ddof=1), np.cov(yy, xx, ddof=1)[0, 1]
        v = (vy / my ** 2 + vx / mx ** 2 - 2 * cxy / (my * mx)) / m if x is not None else vy / (my ** 2 * m)
        return float(np.log(my / mx)), float(max(v, 0.0))

    lt, vt = log_ratio_var(y[t], xv[t])
    lc, vc = log_ratio_var(y[~t], xv[~t])
    d = lt - lc
    se = float(np.sqrt(vt + vc))
    z = d / se if se > 0 else 0.0
    q = _crit(alpha, None, sidedness)
    return {
        "criterion": "network_total_ratio_delta", "estimate": float(np.expm1(d)), "log_estimate": d, "se_log": se,
        "statistic": z, "p_value": _side_p(z, None, sidedness, direction),
        "ci_lower": float(np.expm1(d - q * se)), "ci_upper": float(np.expm1(d + q * se)),
        "n_clusters": int(len(y)),
    }


# ---------------------------------------------------------------- точка входа
def analyze_cluster_rct(
    clusters: pd.DataFrame, estimand: str = "per_restaurant_mean", inference: str = "asymptotic",
    alpha: float = 0.05, sidedness: Sidedness = "two_sided", direction: DirectionT = "increase",
    n_perm: int = 10_000, n_boot: int = 9_999, seed: int = 0,
) -> Dict[str, float]:
    """
    Анализ таблицы ``aggregate_clusters``. Возвращает словарь в формате критериев
    ``StatCriterionRouter`` (mean_control, mean_treatment, absolute_diff, relative_diff, ci, p_value).

    - ``per_restaurant_mean``: оценка и ДИ — Lin + CR2; p-value — по ``inference``
      (asymptotic → CR2 / t(df IK); randomization → перестановки, ДИ обращением теста;
      wild_bootstrap → WCR, ДИ — CR2).
    - ``network_total``: отношение сумм с поправкой на пре-период, дельта-метод
      (для малого числа кластеров p-value — перестановками того же отношения).
    """
    _check(sidedness, direction, alpha)
    for col in ("y", "treat"):
        if col not in clusters.columns:
            raise ValueError(f"Нет колонки {col!r}: используйте aggregate_clusters")
    y = clusters["y"].to_numpy(dtype=float)
    t = clusters["treat"].to_numpy(dtype=int)
    x = clusters["x"].to_numpy(dtype=float) if "x" in clusters.columns else None
    blocks = clusters["block"].to_numpy() if "block" in clusters.columns else None
    se_cl = clusters["se_cluster"].to_numpy() if "se_cluster" in clusters.columns else None
    n_t, n_c = int(t.sum()), int((1 - t).sum())
    if n_t < 2 or n_c < 2:
        raise ValueError("В каждой руке нужно минимум 2 кластера")
    mean_c, mean_t = float(y[t == 0].mean()), float(y[t == 1].mean())
    base = {"mean_control": mean_c, "mean_treatment": mean_t, "n_clusters_control": n_c,
            "n_clusters_treatment": n_t, "estimand": estimand, "inference": inference,
            "sidedness": sidedness, "direction": direction}

    if estimand == "network_total":
        r = network_total_ratio(y, t, x, alpha, sidedness, direction)
        out = {**base, "criterion": r["criterion"], "relative_diff": r["estimate"],
               "absolute_diff": r["estimate"] * mean_c, "p_value": r["p_value"],
               "ci_lower": r["ci_lower"] * mean_c, "ci_upper": r["ci_upper"] * mean_c,
               "ci_lower_rel": r["ci_lower"], "ci_upper_rel": r["ci_upper"]}
        if inference == "randomization":
            out["p_value"] = _ratio_permutation_p(y, t, x, blocks, n_perm, sidedness, direction, seed)
            out["criterion"] += "+randomization_p"
        elif inference == "wild_bootstrap":
            raise ValueError("network_total при малом числе кластеров: используйте inference = randomization")
        out["is_significant"] = bool(out["p_value"] < alpha)
        return out
    if estimand != "per_restaurant_mean":
        raise ValueError(f"Неизвестный estimand {estimand!r}")

    lin = lin_cr2(y, t, x, blocks, se_cl, alpha, sidedness, direction)
    out = {**base, "criterion": lin["criterion"], "absolute_diff": lin["estimate"],
           "se": lin["se"], "df": lin["df"], "p_value": lin["p_value"],
           "ci_lower": lin["ci_lower"], "ci_upper": lin["ci_upper"], "statistic": lin["statistic"]}
    if inference == "randomization":
        if se_cl is not None:
            raise ValueError("Рандомизационный тест предполагает рандомизацию по строкам таблицы кластеров")
        rt = randomization_test(y, t, x, blocks, n_perm, alpha, sidedness, direction, seed)
        out.update(criterion=rt["criterion"], p_value=rt["p_value"], ci_lower=rt["ci_lower"],
                   ci_upper=rt["ci_upper"], absolute_diff=rt["estimate"], n_permutations=rt["n_permutations"],
                   min_attainable_p=rt["min_attainable_p"])
    elif inference == "wild_bootstrap":
        X = lin_design(t, x, blocks)
        wb = wild_cluster_bootstrap(y, X, se_cl, 1, n_boot, alpha, sidedness, direction, seed)
        out.update(criterion=f"{wb['criterion']}+ci_lin_cr2", p_value=wb["p_value"], n_boot=wb["n_boot"])
        if min(n_t, n_c) < WILD_MIN_PER_ARM:
            out["warning"] = (
                f"Wild bootstrap при {min(n_t, n_c)} кластерах на руку слегка завышает FPR (≈ 7% при 8 на руку "
                "на синтетике); для таких дизайнов рекомендуется inference = randomization."
            )
    elif inference != "asymptotic":
        raise ValueError(f"Неизвестный inference {inference!r}")
    out["relative_diff"] = out["absolute_diff"] / mean_c if abs(mean_c) > 1e-12 else 0.0
    out["is_significant"] = bool(out["p_value"] < alpha)
    return out


def _ratio_permutation_p(y, t, x, blocks, n_perm, sidedness, direction, seed) -> float:
    xv = np.ones_like(y) if x is None else x
    codes = _block_codes(blocks) if blocks is not None else np.zeros(len(y), dtype=int)

    def stat(T: np.ndarray) -> np.ndarray:
        Tb = T.astype(bool)
        st = (Tb * y).sum(1) / (Tb * xv).sum(1)
        sc = (~Tb * y).sum(1) / (~Tb * xv).sum(1)
        return np.log(st / sc)

    obs = float(stat(t[None, :])[0])
    T = _permutation_assignments(t, codes, n_perm, np.random.default_rng(seed))
    perm = stat(T)
    if sidedness == "one_sided":
        sign = 1.0 if direction == "increase" else -1.0
        hits = np.sum(sign * perm >= sign * obs - 1e-12)
    else:
        hits = np.sum(np.abs(perm) >= abs(obs) - 1e-12)
    return float((hits + 1) / (len(perm) + 1))
