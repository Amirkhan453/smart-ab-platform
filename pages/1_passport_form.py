"""
Форма Паспорта (ТЗ v4.1, UI-0): создать, отправить на рецензию, утвердить и запустить гостевой тест без кода.
Всё — через HTTP API с личным ключом (роли и аудит те же). Адрес и ключ — из ABP_API_URL и ABP_API_KEY
или из полей на боковой панели; ключ живёт только в сессии браузера.
"""
import os

import streamlit as st

from ab_platform.design import TREE, choose_design
from ab_platform.metrics import APP_GUARDRAIL_NAMES
from ab_platform.schemas.targeting import PLATFORMS
from ab_platform.ui_api import ApiError, PlatformApi
from ab_platform.ui_form import DEFAULT_CV, build_passport, estimate_duration, values_from_passport

st.set_page_config(page_title="Паспорт теста", page_icon="📝", layout="wide")
st.title("📝 Паспорт теста")
st.caption("Гостевой A/B в приложении: поля Паспорта → рецензия → утверждение → предстартовая проверка → запуск.")

# ---------------- подключение ----------------
with st.sidebar:
    st.header("Подключение")
    url = st.text_input("Адрес API", os.environ.get("ABP_API_URL", "http://localhost:8000"), key="api_url")
    key = st.text_input("Личный ключ", os.environ.get("ABP_API_KEY", ""), type="password", key="api_key")
    if st.button("Подключиться", key="connect"):
        try:
            st.session_state["api"] = PlatformApi(url, key)
        except ValueError as exc:
            st.error(str(exc))

api = st.session_state.get("api")
if api is None:
    st.info("Укажите адрес API и личный ключ на боковой панели.")
    st.stop()


def call(fn, *args, **kw):
    """Вызов API: отказ сервера показывается текстом причины, страница не падает."""
    try:
        return fn(*args, **kw)
    except ApiError as exc:
        st.error(f"Сервер отклонил ({exc.status}):\n\n{exc.detail}")
    except ValueError as exc:
        st.error(str(exc))
    return None


metrics = call(api.metrics)
if metrics is None:
    st.stop()
by_ref = {m["ref"]: m for m in metrics}
refs = sorted(by_ref)
title = lambda r: f"{by_ref[r]['title']} ({r}, {by_ref[r]['metric_type']})"  # noqa: E731

# ---------------- дерево выбора дизайна ----------------
with st.expander("Подсказка: какой дизайн нужен (дерево ТЗ v4, 6.1)", expanded=False):
    answers = {n["key"]: st.checkbox(n["question"], key=f"tree_{n['key']}") for n in TREE}
    d = choose_design(answers)
    if d["method"] is None:
        st.warning(d["rule"])
    else:
        st.markdown(f"**{d['method']}** — уровень доказательности {d['evidence_level']}, "
                    f"{'автоматический SHIP возможен' if d['auto_ship_allowed'] else 'автоматический SHIP запрещён'}. "
                    f"Правило: {d['rule']}. Где делается: {d['phase']}.")
        if d["experiment_type"] != "classic_ab":
            st.warning("Эта форма создаёт только гостевой A/B. Для такого дизайна Паспорт пока не создать.")

# ---------------- выбор Паспорта ----------------
passports = call(api.passports) or []
if "_select" in st.session_state:  # после сохранения форма переходит к сохранённому Паспорту
    st.session_state["choice"] = st.session_state.pop("_select")
for kind, text in st.session_state.pop("_flash", []):
    getattr(st, kind)(text)
choice = st.selectbox("Паспорт", ["— новый —"] + [p["experiment_id"] for p in passports], key="choice")
current = call(api.passport, choice) if choice != "— новый —" else None
if current:
    st.markdown(f"Статус: **{current['status']}**, версия {current['version']}, владелец {current['owner']}"
                + (f", утвердил {current['approved_by']}" if current.get("approved_by") else ""))
    for w in current.get("design_warnings", []):
        st.warning(w)
v = values_from_passport(current) if current else {}
sfx = f"_{current['experiment_id']}" if current else "_new"  # свои поля у каждого Паспорта: значения не смешиваются

# ---------------- поля ----------------
st.subheader("Поля Паспорта")
c1, c2 = st.columns(2)
with c1:
    exp_id = st.text_input("ID теста (experiment_id)", v.get("experiment_id", ""), key="experiment_id" + sfx,
                           disabled=current is not None)
    test_key = st.text_input("Ключ фичи (test_key)", v.get("test_key", ""), key="test_key" + sfx)
    layer = st.text_input("Слой", v.get("layer_id", "ui"), key="layer_id" + sfx)
    hypothesis = st.text_area("Гипотеза: «Если мы X, то метрика Y изменится на Z%, потому что R»",
                              v.get("hypothesis_statement", ""), key="hypothesis" + sfx)
    unit = st.selectbox("Единица рандомизации", ["user_id", "device_id"],
                        index=["user_id", "device_id"].index(v.get("randomization_unit", "user_id")), key="unit" + sfx)
    share = st.number_input("Доля тестовой группы, %", 1.0, 99.0, float(v.get("treatment_share_pct", 50.0)),
                            key="share" + sfx)
    ramp = st.text_input("План раскатки, % дорожки", v.get("ramp_plan", "10, 50, 100"), key="ramp" + sfx)
with c2:
    primary = st.selectbox("Главная метрика", refs, format_func=title, key="primary" + sfx,
                           index=refs.index(v.get("primary_ref") or "revenue_per_guest@1")
                           if (v.get("primary_ref") or "revenue_per_guest@1") in refs else 0)
    default_guards = v.get("guardrail_refs") or [r for r in refs if by_ref[r]["name"] in APP_GUARDRAIL_NAMES]
    guards = st.multiselect("Защитные метрики", refs, default=[r for r in default_guards if r in refs],
                            format_func=title, key="guards" + sfx)
    degradation = st.number_input("Допустимое ухудшение защитных метрик, %", 0.1, 100.0,
                                  float(v.get("max_degradation_pct", 10.0)), key="degradation" + sfx)
    rationale = st.text_input("Какой риск страхуют защитные метрики", v.get("tradeoff_rationale", "") or
                              "падения и ошибки оплаты в варианте", key="rationale" + sfx)
    secondary = st.multiselect("Вторичные метрики", refs, default=[r for r in v.get("secondary_refs", []) if r in refs],
                               format_func=title, key="secondary" + sfx)
    platforms = st.multiselect("Платформы (пусто — все)", list(PLATFORMS), default=v.get("platforms", []),
                               key="platforms" + sfx)
    version_min = st.text_input("Версия приложения от (пусто — любая)", v.get("app_version_min", ""), key="vmin" + sfx)

st.subheader("Мощность и длительность")
c3, c4, c5 = st.columns(3)
with c3:
    baseline = st.number_input("Базовое значение главной метрики", 0.0001, 1e9, float(v.get("baseline_value", 450.0)),
                               key="baseline" + sfx)
    mde = st.number_input("MDE, % (относительный)", 0.1, 100.0, float(v.get("mde_pct", 3.0)), key="mde" + sfx)
    cv = st.number_input("Коэффициент вариации (σ / среднее; для бинарной не нужен)", 0.01, 50.0, DEFAULT_CV,
                         key="cv" + sfx)
with c4:
    daily = st.number_input("Гостей в день в слое (до таргетинга)", 1, 10_000_000, int(v.get("daily_eligible_units", 10_000)),
                            key="daily" + sfx)
    audience = st.number_input("Доля аудитории под таргетингом, %", 0.1, 100.0, float(v.get("audience_share_pct", 100.0)),
                               key="audience" + sfx)
    duration = st.number_input("Плановая длительность, дней", 7, 365, int(v.get("planned_duration_days", 28)), step=7,
                               key="duration" + sfx)
with c5:
    alpha = st.number_input("α", 0.001, 0.2, float(v.get("alpha", 0.05)), format="%.3f", key="alpha" + sfx)
    power = st.number_input("Мощность", 0.51, 0.99, float(v.get("power", 0.8)), key="power" + sfx)
    window = st.number_input("Окно атрибуции, дней", 1, 60, int(v.get("attribution_window_days", 7)), key="window" + sfx)
    cuped = st.checkbox("CUPED (ковариата — та же метрика до показа)", bool(v.get("use_cuped", True)), key="cuped" + sfx)
    rho = st.number_input("Корреляция до/после для CUPED", 0.0, 0.95, 0.5 if cuped else 0.0, key="rho" + sfx)

target_daily = int(daily * (audience / 100 if platforms or version_min else 1.0))
try:
    est = estimate_duration(by_ref[primary]["metric_type"], baseline, mde / 100, max(target_daily, 1), alpha=alpha,
                            power=power, cv=cv, pre_post_correlation=rho if cuped else 0.0,
                            treatment_share=share / 100, direction=by_ref[primary]["direction"])
except ValueError as exc:
    st.error(f"Калькулятор: {exc}")
else:
    st.markdown(f"Целевая аудитория: **{target_daily:,}** гостей в день. Нужно **{est['n_per_variant']:,}** на вариант, "
                f"это **{est['days']} дн.** (целыми неделями).".replace(",", " "))
    if est["days"] > duration:
        st.warning(f"Плановой длительности {duration} дн. не хватает: нужно {est['days']} дн. {est['recommendation']}")

values = dict(experiment_id=exp_id, test_key=test_key, layer_id=layer, hypothesis_statement=hypothesis,
              randomization_unit=unit, treatment_share_pct=share, primary_ref=primary, secondary_refs=secondary,
              guardrail_refs=guards, max_degradation_pct=degradation, tradeoff_rationale=rationale,
              baseline_value=baseline, mde_pct=mde, alpha=alpha, power=power, planned_duration_days=duration,
              daily_eligible_units=daily, platforms=platforms, app_version_min=version_min,
              audience_share_pct=audience, ramp_plan=ramp, attribution_window_days=window, use_cuped=cuped)

# ---------------- команды ----------------
st.subheader("Команды")
def passport_body():
    """Тело Паспорта: определения метрик — из реестра через API."""
    defs = {}
    for r in {primary, *guards, *secondary}:
        m = call(api.metric, r)
        if m is None:
            return None
        defs[r] = m["definition"]
    return call(build_passport, values, defs)


editable = current is None or current["status"] == "draft"
if st.button("Сохранить черновик", key="save", disabled=not editable):
    body = passport_body()
    saved = call(api.save_draft, body) if body else None
    if saved:
        st.session_state["_flash"] = [("success", f"Черновик {saved['experiment_id']} v{saved['version']} сохранён.")]
        st.session_state["_select"] = saved["experiment_id"]
        st.rerun()
if current is not None and current["status"] not in ("draft", "review"):
    if st.button("Новая версия с этими полями", key="new_version"):
        body = passport_body()
        if body:
            body.pop("experiment_id")
            nv = call(api.new_version, exp_id, body)
            if nv:
                st.session_state["_flash"] = [("success", f"Создана версия {nv['version']} (черновик).")]
                st.rerun()

ACTIONS = [("submit", "На рецензию"), ("approve", "Утвердить"), ("return", "Вернуть в черновик"),
           ("preflight", "Предстартовая проверка"), ("start", "Запустить"), ("pause", "Пауза"),
           ("resume", "Продолжить"), ("stop", "Остановить")]
if current is not None:
    cols = st.columns(len(ACTIONS))
    comment = st.text_input("Комментарий при возврате в черновик", key="comment")
    for col, (action, label) in zip(cols, ACTIONS):
        if col.button(label, key=f"cmd_{action}"):
            out = call(api.command, current["experiment_id"], action, {"comment": comment} if action == "return" else None)
            if out is not None:
                if action == "preflight":
                    msg = "пройдена" if out.get("passed") else "не пройдена: " + "; ".join(out.get("problems") or [])
                    flash = ("success" if out.get("passed") else "error", f"Предстартовая проверка {msg}.")
                else:
                    flash = ("success", f"Готово: статус {out.get('status')}.")
                st.session_state["_flash"] = [flash]
                st.rerun()  # статус и кнопки — уже по новому состоянию

# ---------------- отчёт и тревоги ----------------
if current is not None:
    st.subheader("Последний отчёт и тревоги")
    runs = [r for r in (call(api.runs, current["experiment_id"]) or []) if r["kind"] != "preflight"]
    if runs:
        last = max(runs, key=lambda r: r["created_at"])
        st.markdown(call(api.report, last["run_id"]) or "")
    else:
        st.caption("Анализов ещё не было.")
    alerts = call(api.alerts, current["experiment_id"]) or []
    for a in alerts:
        st.warning(f"{a['created_at'][:16]} · {a['kind']}: {a['message']}")
    if not alerts:
        st.caption("Тревог нет.")
