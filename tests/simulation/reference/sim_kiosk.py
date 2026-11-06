"""Киоск: рандомизация по заказу внутри ресторан-дня. Проверка FPR и MDE против формулы."""
import numpy as np
from scipy import stats
rng = np.random.default_rng(7)
R, D, K = 150, 14, 150          # ресторанов, дней, заказов на ресторан-день
cells = R * D
def one(effect):
    cell_eff = rng.normal(0, 0.3, cells)                      # уровень ресторан-дня (log)
    sigma = np.sqrt(np.log(1 + 0.7**2))                       # CV чека внутри ячейки = 0.7
    y = np.exp(np.repeat(cell_eff, K) + rng.normal(-sigma**2/2, sigma, cells * K)) * 400
    t = rng.integers(0, 2, cells * K).astype(bool)            # монетка на заказ
    y = np.where(t, y * (1 + effect), y)
    cell = np.repeat(np.arange(cells), K)
    # (1) простая разность средних, Welch
    p1 = stats.ttest_ind(y[t], y[~t], equal_var=False).pvalue
    d1 = y[t].mean() - y[~t].mean()
    # (2) с фиксированными эффектами ресторан-дня (вычитаем среднее ячейки)
    yc = y - np.bincount(cell, y)[cell] / K
    d2 = yc[t].mean() - yc[~t].mean()
    se2 = np.sqrt(yc[t].var(ddof=1) / t.sum() + yc[~t].var(ddof=1) / (~t).sum())
    p2 = 2 * stats.norm.sf(abs(d2 / se2))
    return p1, p2, d1, d2, y[~t].mean()
res = np.array([one(0.0) for _ in range(400)])
print("A/A FPR welch=%.3f  FE=%.3f" % ((res[:, 0] < .05).mean(), (res[:, 1] < .05).mean()))
base = res[:, 4].mean()
for i, name in [(2, "welch"), (3, "FE")]:
    print(name, "MDE_sim=%.2f%%" % (100 * 2.8016 * res[:, i].std(ddof=1) / base))
n = cells * K / 2
cv_total = np.sqrt((1 + 0.7**2) * np.exp(0.3**2) - 1)
print("formula MDE (CV within=0.7) = %.2f%%, (CV total=%.2f) = %.2f%%" % (
    100 * 2.8016 * 0.7 * np.sqrt(2 / n), cv_total, 100 * 2.8016 * cv_total * np.sqrt(2 / n)))
