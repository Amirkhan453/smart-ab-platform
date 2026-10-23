import sqlite3
import numpy as np
import pandas as pd
import streamlit as st

from ab_platform.schemas.passport import (
    AnalysisUnit,
    Direction,
    ExperimentPassport,
    ExperimentType,
    GuardrailDefinition,
    MetricDefinition,
    MetricType,
    RandomizationUnit,
    RiskProfile,
    VariantAllocation,
)
from ab_platform.engine.power import PowerAndSampleSizeEngine
from ab_platform.engine.aa_validator import HistoricalAAValidator
from ab_platform.engine.assignment import DeterministicAssignmentEngine
from ab_platform.engine.safeguards import ExperimentSafeguardEngine
from ab_platform.engine.switchback import SwitchbackAndClusterEngine
from ab_platform.engine.offline_geo import OfflineRetailGeoEngine
from ab_platform.engine.causal_quasi import CausalAndMLDebiasingEngine
from ab_platform.engine.bandits import MultiArmedBanditEngine
from ab_platform.copilot.arbor_engine import ArborDecisionCopilot, CopilotSessionState
from ab_platform.connectors.sql_builder import WarehouseSQLBuilder
from ab_platform.simulation.synthetic import SyntheticDataGenerator
from ab_platform.orchestrator import SmartABPlatformOrchestrator


st.set_page_config(
    page_title="Smart A/B & Causal Platform with AI Copilot",
    page_icon="🧪",
    layout="wide",
)

st.title("🧪 Умная A/B и Causal Inference Платформа с ИИ-Ассистентом")
st.caption(
    "Автопилот проверки бизнес-гипотез: граф решений Arbor • Калькуляторы Эвана Миллера и Switchback • "
    "CUPED & Метод Лина • SPRT SRM-мониторинг • Оффлайн Гео-тесты (Synthetic Control) • PSM / DiD / Бандиты"
)

with st.sidebar:
    st.header("⚙️ Источник данных и Симулятор")
    data_source_mode = st.radio(
        "Выберите источник данных:",
        ["Синтетический симулятор бизнеса", "Загрузить собственный CSV"],
    )

    if data_source_mode == "Синтетический симулятор бизнеса":
        sim_n_users = st.slider("Количество пользователей в выборке", 2000, 100000, 20000, step=2000)
        sim_split = st.slider("Доля тестовой группы (Treatment)", 0.10, 0.90, 0.50, step=0.05)
        sim_true_lift = st.slider("Истинный продуктовый эффект (Lift, %)", -10.0, 25.0, 5.0, step=0.5) / 100.0
        sim_corr = st.slider("Корреляция с пред-периодом (для CUPED)", 0.0, 0.95, 0.70, step=0.05)
        sim_outliers = st.slider("Доля выбросов / 'китов' (%)", 0.0, 2.0, 0.2, step=0.1) / 100.0
        sim_double_exp = st.slider("Доля багов Double Exposure (%)", 0.0, 5.0, 0.5, step=0.1) / 100.0
        sim_guardrail_drop = st.checkbox("Симулировать пробой Guardrail-метрики (ошибки страницы)", value=False)

        active_df = SyntheticDataGenerator.generate_ab_dataset(
            n_users=sim_n_users,
            treatment_ratio=sim_split,
            true_lift_rel=sim_true_lift,
            pre_post_corr=sim_corr,
            outlier_fraction=sim_outliers,
            double_exposure_fraction=sim_double_exp,
            random_seed=42,
        )
        rng_guard = np.random.default_rng(42)
        base_err = 0.02
        treat_err = 0.035 if sim_guardrail_drop else 0.0205
        active_df["page_error"] = np.where(
            active_df["variant"] == "treatment",
            (rng_guard.random(len(active_df)) < treat_err).astype(int),
            (rng_guard.random(len(active_df)) < base_err).astype(int),
        )
    else:
        uploaded_file = st.file_uploader("Загрузите CSV-файл с логами эксперимента", type=["csv"])
        if uploaded_file is not None:
            active_df = pd.read_csv(uploaded_file)
        else:
            st.info("Пока CSV не загружен, используется демо-датасет (10 000 юзеров).")
            active_df = SyntheticDataGenerator.generate_ab_dataset(n_users=10000, true_lift_rel=0.04)
            active_df["page_error"] = 0

    st.divider()
    st.markdown(f"**Строк в датасете:** `{len(active_df):,}`")
    st.markdown(f"**Колонки:** `{', '.join(active_df.columns)}`")


tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🤖 1. ИИ-Копилот & Паспорт",
    "📐 2. Выборка & MDE",
    "🗄️ 3. БД, SQL & А/А-тест",
    "🛡️ 4. Сплит & SRM-Защита",
    "📊 5. Автопилот Итогов",
    "🧭 6. Оффлайн, Switchback & Causal",
])

with tab1:
    st.subheader("🤖 ИИ-Ассистент (Граф решений Arbor) и Пре-Регистрация Эксперимента")
    col_q, col_res = st.columns([1, 1])

    with col_q:
        st.markdown("#### Шаг 1. Ответьте на вопросы о специфике бизнеса и рисках")
        q_can_rand = st.selectbox(
            "1. Можем ли мы технически рандомизировать трафик/точки?",
            options=[True, False],
            format_func=lambda x: "Да, запускаем управляемый эксперимент" if x else "Нет, анализируем исторический запуск (Quasi)",
        )
        q_offline = st.selectbox(
            "2. Где проводится эксперимент?",
            options=[False, True],
            format_func=lambda x: "Онлайн (сайт, мобильное приложение, бэкенд)" if not x else "Оффлайн (физические магазины, города, филиалы)",
        )
        q_net_eff = st.selectbox(
            "3. Есть ли сетевые эффекты (конкуренция за курьеров/товары/аукцион или соц. граф)?",
            options=[False, True],
            format_func=lambda x: "Нет, пользователи независимы (SUTVA выполняется)" if not x else "Да, группы влияют друг на друга (нарушение SUTVA)",
        )
        q_inter_type = st.selectbox(
            "4. Тип сетевой интерференции (если есть):",
            options=["temporal_marketplace", "social_or_b2b_cluster"],
            format_func=lambda x: "Динамический баланс спроса/предложения (доставка, такси, прайсинг)"
            if x == "temporal_marketplace"
            else "Кластеры пользователей / B2B-компании / соц. граф",
        )
        q_regret = st.selectbox(
            "5. Главная цель запуска:",
            options=[False, True],
            format_func=lambda x: "Строгое статистическое доказательство эффекта (A/B)"
            if not x
            else "Быстрая максимизация конверсии на короткой акции (Бандит)",
        )
        q_cannibal = st.checkbox("Есть риск каннибализации смежных товаров/категорий", value=True)
        q_novelty = st.checkbox("Ожидается эффект новизны (Novelty Effect) в первые дни", value=False)
        q_lag = st.number_input("Лаг созревания целевой метрики клиента (дней)", min_value=0, max_value=90, value=2)

    copilot = ArborDecisionCopilot()
    session_state = CopilotSessionState()
    session_state = copilot.advance_tree(
        session_state,
        {
            "can_randomize": q_can_rand,
            "is_offline": q_offline,
            "has_network_effects": q_net_eff,
            "interference_type": q_inter_type,
            "prefer_regret_minimization": q_regret,
            "has_cannibalization_risk": q_cannibal,
            "reaction_lag_days": q_lag,
            "daily_traffic": len(active_df) // 14,
            "mde_relative": 0.03,
        },
    )

    with col_res:
        st.markdown("#### Вердикт ИИ-Копилота (Arbor Engine)")
        st.success(f"**Рекомендуемый дизайн:** `{session_state.recommended_design.value}`")
        st.info(
            f"**Рекомендуемая единица рандомизации:** `{session_state.recommended_randomization_unit.value}`  \n"
            f"**Сигнал достаточности информации ($\\hat{{p}}_{{\\text{{suff}}}}$):** `{session_state.p_suff * 100:.0f}%`  \n"
            f"**Заключение узла:** {copilot.nodes[session_state.current_node].question_prompt}"
        )
        if session_state.warnings:
            st.warning("**Обнаруженные методические риски:**\n" + "\n".join(f"- {w}" for w in session_state.warnings))

    st.divider()
    st.markdown("#### Шаг 2. Сформировать и Зафиксировать Паспорт Эксперимента (Pre-Registration Lock)")
    c1, c2, c3 = st.columns(3)
    with c1:
        p_exp_id = st.text_input("ID Эксперимента", value="exp_checkout_2026")
        p_layer_id = st.text_input("Слой изоляции (Plane / Layer ID)", value="checkout_ui_layer")
        p_hyp = st.text_area(
            "Структурированная гипотеза",
            value="Если мы ускорим шаг оплаты, то выручка на пользователя вырастет на 4%, так как снизится отвал из корзины",
        )
    with c2:
        p_metric_type = st.selectbox(
            "Тип целевой метрики (Primary Metric)",
            options=[MetricType.CONTINUOUS, MetricType.BINARY, MetricType.RATIO, MetricType.QUANTILE],
            format_func=lambda m: {
                MetricType.CONTINUOUS: "Непрерывная (post_revenue -> Welch t-test + CUPED)",
                MetricType.BINARY: "Бинарная конверсия (converted -> Z-test)",
                MetricType.RATIO: "Ratio-метрика (orders / sessions -> Дельта-метод)",
                MetricType.QUANTILE: "Квантиль p90 (post_revenue -> Bootstrap)",
            }[m],
        )
        p_mde = st.number_input("Целевой относительный MDE (%)", min_value=0.5, max_value=50.0, value=4.0, step=0.5) / 100.0
        p_dur = st.number_input("Планируемая длительность (дней, мин. 7)", min_value=7, max_value=90, value=14)
    with c3:
        p_guard_thresh = (
            st.number_input("Порог допустимого ухудшения Guardrail-метрики ошибок (%)", min_value=1.0, max_value=50.0, value=15.0)
            / 100.0
        )
        p_winsor = st.slider("Квантиль винзоризации выбросов", 0.950, 1.000, 0.995, step=0.001)
        p_daily = st.number_input("Доступный дневной трафик (юзеров/день)", min_value=100, value=5000, step=500)

    num_col_map = {
        MetricType.CONTINUOUS: "post_revenue",
        MetricType.BINARY: "converted",
        MetricType.RATIO: "orders",
        MetricType.QUANTILE: "post_revenue",
    }
    den_col_map = {
        MetricType.CONTINUOUS: None,
        MetricType.BINARY: None,
        MetricType.RATIO: "sessions",
        MetricType.QUANTILE: None,
    }

    try:
        current_passport = ExperimentPassport(
            experiment_id=p_exp_id,
            test_key=f"key_{p_exp_id}",
            layer_id=p_layer_id,
            hypothesis_statement=p_hyp,
            experiment_type=ExperimentType.CLASSIC_AB if not q_net_eff else session_state.recommended_design,
            randomization_unit=RandomizationUnit.USER_ID if not q_net_eff else session_state.recommended_randomization_unit,
            analysis_unit=AnalysisUnit.PAGEVIEW if p_metric_type == MetricType.RATIO else AnalysisUnit.USER,
            variants=[
                VariantAllocation(variant_id="control", traffic_percentage=50.0, is_control=True),
                VariantAllocation(variant_id="treatment", traffic_percentage=50.0, is_control=False),
            ],
            primary_metric=MetricDefinition(
                name=f"primary_{p_metric_type.value}",
                metric_type=p_metric_type,
                numerator_col=num_col_map[p_metric_type],
                denominator_col=den_col_map[p_metric_type],
                expected_direction=Direction.INCREASE,
                winsorize_percentile=p_winsor,
            ),
            secondary_metrics=[
                MetricDefinition(
                    name="secondary_conversion_rate",
                    metric_type=MetricType.BINARY,
                    numerator_col="converted",
                    expected_direction=Direction.INCREASE,
                ),
                MetricDefinition(
                    name="orders_per_session_ratio",
                    metric_type=MetricType.RATIO,
                    numerator_col="orders",
                    denominator_col="sessions",
                    expected_direction=Direction.INCREASE,
                ),
            ],
            guardrail_metrics=[
                GuardrailDefinition(
                    metric=MetricDefinition(
                        name="page_error_rate",
                        metric_type=MetricType.BINARY,
                        numerator_col="page_error",
                        expected_direction=Direction.DECREASE,
                    ),
                    max_allowed_degradation_rel=p_guard_thresh,
                    tradeoff_rationale="Защита от роста технических ошибок на странице чекаута",
                )
            ],
            baseline_value=float(active_df[num_col_map[p_metric_type]].mean()),
            mde_relative=p_mde,
            planned_duration_days=int(p_dur),
            daily_eligible_units=int(p_daily),
            risk_profile=RiskProfile(
                has_network_effects=q_net_eff,
                has_cannibalization_risk=q_cannibal,
                reaction_lag_days=int(q_lag),
                novelty_effect_expected=q_novelty,
            ),
        )
        st.session_state["passport"] = current_passport
        st.success("✅ Паспорт Эксперимента прошел все методологические проверки и зафиксирован!")
    except Exception as exc:
        st.error(f"❌ Ошибка валидации Паспорта Эксперимента: {exc}")


with tab2:
    st.subheader("📐 Калькулятор Выборки, MDE и Длительности (Evan Miller • Calculator.net • Switchback)")
    calc_mode = st.radio(
        "Выберите режим калькулятора:",
        [
            "1. Классический A/B и CUPED (Точная формула Эвана Миллера / Welch / Дельта-метод)",
            "2. Аудит доверительного интервала и погрешности (Режим Calculator.net + FPC)",
            "3. Калькулятор Switchback и Кластерных экспериментов",
        ],
        horizontal=True,
    )

    if calc_mode.startswith("1."):
        cc1, cc2, cc3 = st.columns(3)
        with cc1:
            is_bin = st.checkbox("Бинарная метрика (Конверсия %)", value=True)
            base_val = st.number_input(
                "Базовое значение (Baseline: доля 0..1 для конверсии или среднее для чека)",
                min_value=0.001,
                value=0.20 if is_bin else 1000.0,
                step=0.01 if is_bin else 50.0,
            )
            var_val = st.number_input("Дисперсия метрики (для непрерывных/Ratio)", min_value=1.0, value=90000.0)
        with cc2:
            mde_mode = st.selectbox("Формат MDE", ["Относительный (%)", "Абсолютный (п.п. или ед.)"])
            mde_input = st.number_input("Значение MDE", min_value=0.1, value=5.0, step=0.5)
            bin_formula = st.selectbox("Формула для конверсий", ["evan_miller", "pooled", "baseline"])
        with cc3:
            c_alpha = st.slider("Уровень значимости (Alpha)", 0.01, 0.10, 0.05, step=0.01)
            c_power = st.slider("Мощность (1 - Beta)", 0.70, 0.95, 0.80, step=0.05)
            c_cuped_r = st.slider("Ожидаемая корреляция ковариаты CUPED (r)", 0.0, 0.90, 0.0, step=0.05)
            c_traffic = st.number_input("Дневной трафик (юзеров/день)", min_value=100, value=10000, step=1000)

        res_power = PowerAndSampleSizeEngine.calculate_sample_size(
            baseline_mean=base_val,
            mde_relative=(mde_input / 100.0) if mde_mode.startswith("Относительный") else None,
            mde_absolute=(mde_input / 100.0 if is_bin else mde_input) if not mde_mode.startswith("Относительный") else None,
            metric_variance=None if is_bin else var_val,
            is_binary=is_bin,
            binary_method=bin_formula,
            alpha=c_alpha,
            power=c_power,
            daily_traffic=int(c_traffic),
            pre_post_correlation_r=c_cuped_r,
        )

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Выборка на 1 вариант", f"{res_power.n_per_variant:,}")
        m2.metric("Общая выборка (A + B)", f"{res_power.total_sample_size:,}")
        m3.metric("Длительность теста", f"{res_power.estimated_days} дн.")
        m4.metric("Сжатие выборки CUPED", f"{res_power.variance_reduction_pct:.1f}%")

        st.info(res_power.recommendation)
        st.markdown("**Таблица компромиссов (Какой минимальный эффект мы заметим по неделям):**")
        st.dataframe(pd.DataFrame(res_power.tradeoff_table), use_container_width=True)

    elif calc_mode.startswith("2."):
        sc1, sc2 = st.columns(2)
        with sc1:
            s_n = st.number_input("Размер выборки (Sample Size)", min_value=10, value=20000, step=1000)
            s_p = st.number_input("Доля признака / Конверсия (%)", min_value=0.1, max_value=99.9, value=3.0, step=0.5) / 100.0
        with sc2:
            s_cl = st.slider("Доверительная вероятность (Confidence Level)", 0.80, 0.99, 0.95, step=0.01)
            s_pop = st.number_input("Размер всей генеральной совокупности (0 = бесконечная)", min_value=0, value=0, step=5000)

        audit_res = PowerAndSampleSizeEngine.calculate_survey_margin_of_error(
            sample_size=int(s_n),
            proportion=s_p,
            confidence_level=s_cl,
            finite_population_n=int(s_pop) if s_pop > s_n else None,
        )
        a1, a2, a3 = st.columns(3)
        a1.metric("Предельная ошибка (Margin of Error)", f"±{audit_res.margin_of_error_pct_points:.2f}%")
        a2.metric("Доверительный интервал доли", f"[{audit_res.ci_lower*100:.2f}%, {audit_res.ci_upper*100:.2f}%]")
        a3.metric("Достижимый отн. MDE при A/B 50/50", f"{audit_res.achievable_ab_mde_rel_if_split_50_50*100:.2f}%")

    else:
        sb1, sb2 = st.columns(2)
        with sb1:
            sb_j = st.number_input("Число гео-кластеров / зон (J)", min_value=2, value=10)
            sb_h = st.number_input("Число временных окон переключения (H)", min_value=2, value=28)
            sb_n = st.number_input("Среднее число заказов/юзеров в ячейке (n_bar)", min_value=5.0, value=150.0)
        with sb2:
            sb_cv = st.slider("Коэффициент вариации размера ячеек (CV)", 0.0, 3.0, 1.2, step=0.1)
            sb_base = st.number_input("Базовое среднее метрики", min_value=1.0, value=500.0)
            sb_var = st.number_input("Общая дисперсия метрики", min_value=1.0, value=25000.0)

        sb_power = PowerAndSampleSizeEngine.calculate_switchback_mde(
            num_clusters_j=int(sb_j),
            num_time_windows_h=int(sb_h),
            mean_cell_size_n=sb_n,
            cell_size_cv=sb_cv,
            baseline_mean=sb_base,
            total_variance=sb_var,
        )
        k1, k2, k3 = st.columns(3)
        k1.metric("Всего ячеек рандомизации (J x H)", f"{sb_power.required_total_cells}")
        k2.metric("Относительный MDE Switchback", f"{sb_power.mde_relative*100:.2f}%")
        k3.metric("Макро-вес Lambda для CUPAC", f"{sb_power.macro_inflation_factor_lambda:.1f}")
        if sb_power.warning_message:
            st.warning(sb_power.warning_message)


with tab3:
    st.subheader("🗄️ Авто-генератор SQL к БД пользователя и Исторический А/А-Валидатор")
    col_sql, col_aa = st.columns([1, 1])

    with col_sql:
        st.markdown("#### 1. Генерация SQL-запроса (Exposure Ordering + CUPED Pre-Window)")
        sql_dialect = st.selectbox("Диалект СУБД:", ["postgres", "clickhouse", "sqlite"])
        sql_metric = st.text_input("Имя целевой метрики в таблице событий", value="revenue")
        sql_pre_days = st.slider("Окно пред-периода для сбора ковариаты CUPED (дней)", 7, 60, 14)

        generated_sql = WarehouseSQLBuilder.build_user_level_cuped_query(
            experiment_id="exp_checkout_2026",
            metric_name=sql_metric,
            pre_window_days=sql_pre_days,
            dialect=sql_dialect,
        )
        st.code(generated_sql, language="sql")

    with col_aa:
        st.markdown("#### 2. Проверка честности стат-критерия (Исторический А/А-тест)")
        aa_col = st.selectbox("Выберите колонку для А/А-симуляции:", ["post_revenue", "pre_revenue", "orders", "converted"])
        aa_iters = st.slider("Количество Монте-Карло итераций А/А-теста", 100, 1000, 300, step=100)

        if st.button("🚀 Запустить исторический А/А-тест"):
            aa_res = HistoricalAAValidator.run_aa_simulation(
                active_df, metric_col=aa_col, iterations=aa_iters, alpha=0.05
            )
            a_c1, a_c2, a_c3 = st.columns(3)
            a_c1.metric("Фактический FPR (ожид. ~5%)", f"{aa_res['false_positive_rate']*100:.2f}%")
            a_c2.metric("KS p-value равномерности", f"{aa_res['ks_pvalue']:.4f}")
            a_c3.metric("Распределение p-value", "Равномерное ✅" if aa_res["is_uniform_distribution"] else "Искажено ⚠️")


with tab4:
    st.subheader("🛡️ Детерминированное Сплитование (10 000 бакетов) и Защита от SRM (SPRT)")
    sp_col1, sp_col2 = st.columns(2)

    with sp_col1:
        st.markdown("#### Проверка бакетирования и ортогональности слоев")
        test_uid = st.text_input("Тестовый User ID", value="user_000777")
        ramp_pct = st.slider("Процент раскатки эксперимента (Exposure Rate %)", 5.0, 100.0, 50.0, step=5.0)
        holdout_pct = st.slider("Глобальный Holdout-слой (%)", 0.0, 10.0, 2.0, step=0.5)

        assign_res = DeterministicAssignmentEngine.assign_user(
            user_id=test_uid,
            experiment_id="exp_checkout_2026",
            exposure_rate_pct=ramp_pct,
            control_split_pct=50.0,
            global_holdout_pct=holdout_pct,
        )
        st.json({
            "user_id": assign_res.user_id,
            "bucket_z_0_9999": assign_res.bucket_z,
            "is_in_global_holdout": assign_res.is_in_global_holdout,
            "is_exposed": assign_res.is_exposed,
            "assigned_variant": assign_res.variant_id,
        })

    with sp_col2:
        st.markdown("#### Детектор SRM по стадиям воронки (Assigned vs Triggered)")
        ass_c = st.number_input("Назначено в Control (Assigned)", value=50000, step=500)
        ass_t = st.number_input("Назначено в Treatment (Assigned)", value=50020, step=500)
        trig_c = st.number_input("Активировали фичу в Control (Triggered)", value=42000, step=500)
        trig_t = st.number_input("Активировали фичу в Treatment (Triggered)", value=41950, step=500)

        srm_stage_res = ExperimentSafeguardEngine.diagnose_srm_stage(
            assigned_control=int(ass_c),
            assigned_treatment=int(ass_t),
            triggered_control=int(trig_c),
            triggered_treatment=int(trig_t),
        )
        if srm_stage_res["assigned_srm"] or srm_stage_res["triggered_srm"]:
            st.error(srm_stage_res["diagnosis"])
        else:
            st.success(srm_stage_res["diagnosis"])


with tab5:
    st.subheader("📊 Автопилот Итогов Эксперимента (Hygiene + CUPED/Lin + Auto-Router + Guardrails)")
    use_lin = st.checkbox("Использовать раздельную регрессию Лина (Lin's Method) вместо стандартного CUPED", value=False)
    trig_share = st.slider("Доля затронутого трафика от всего продукта (для расчета Site Wide Impact)", 0.05, 1.0, 0.50, step=0.05)

    if "passport" in st.session_state:
        report = SmartABPlatformOrchestrator.evaluate_experiment(
            passport=st.session_state["passport"],
            df=active_df,
            user_col="user_id",
            variant_col="variant",
            covariate_col="pre_revenue",
            use_lins_method=use_lin,
            triggered_share_alpha=trig_share,
        )

        if report.decision_status == "SHIP_TO_PRODUCTION":
            st.success(report.executive_summary_ru)
        elif "INVALID" in report.decision_status or "BREACH" in report.decision_status or "DEGRADED" in report.decision_status:
            st.error(report.executive_summary_ru)
        else:
            st.warning(report.executive_summary_ru)

        r1, r2, r3, r4 = st.columns(4)
        r1.metric("Относительный эффект (Primary)", f"{report.primary_metric_result['relative_diff']*100:+.2f}%")
        r2.metric("p-value (Авто-критерий)", f"{report.primary_metric_result['p_value']:.4f}")
        r3.metric("Сжатие дисперсии CUPED", f"{report.cuped_diagnostics['variance_reduction_pct']:.1f}%")
        r4.metric("Site Wide Impact (SWI)", f"{report.site_wide_impact['site_wide_impact_pct']:+.2f}%")

        st.markdown("#### 🛡️ Статус защитных Guardrail-метрик")
        st.dataframe(pd.DataFrame([vars(g) for g in report.guardrail_results]), use_container_width=True)

        st.markdown("#### 📈 Вторичные метрики (с поправкой Бенджамини-Хохберга FDR и Tiered Alpha)")
        st.dataframe(pd.DataFrame([vars(s) for s in report.secondary_results]), use_container_width=True)
    else:
        st.warning("Сначала проверьте и зафиксируйте Паспорт Эксперимента во вкладке 1.")


with tab6:
    st.subheader("🧭 Оффлайн-ритейл (Synthetic Control), Switchback CUPAC, Квази-тесты (PSM/DiD) и Бандиты")
    adv_mode = st.selectbox(
        "Выберите продвинутый модуль:",
        [
            "1. Оффлайн-ритейл: Подбор пар магазинов (Matched Markets) и Synthetic Control",
            "2. Switchback Power-Optimal CUPAC (Анализ с подавлением межъячеечного шума)",
            "3. Квази-эксперименты без рандомизации: Propensity Score Matching (PSM), Doubly Robust (AIPW) и DiD",
            "4. Многорукие и Контекстные Бандиты (Thompson Sampling, UCB1, Contextual LinUCB)",
        ],
    )

    if adv_mode.startswith("1."):
        rng_geo = np.random.default_rng(101)
        days = np.arange(1, 61)
        base_trend = 500.0 + 50.0 * np.sin(days / 4.0)
        geo_rows = []
        for t in days:
            tr = base_trend[t - 1]
            d1 = tr + rng_geo.normal(0, 2.0)
            d2 = tr * 0.6 + 100.0 + rng_geo.normal(0, 2.0)
            d3 = 300.0 - tr * 0.2 + rng_geo.normal(0, 5.0)
            s_t = 0.7 * d1 + 0.3 * d2 + (45.0 if t >= 40 else 0.0) + rng_geo.normal(0, 1.5)
            for sid, val in [("store_Treatment", s_t), ("store_Donor1", d1), ("store_Donor2", d2), ("store_Donor3", d3)]:
                geo_rows.append({"date": int(t), "store_id": sid, "revenue": float(val)})
        geo_df = pd.DataFrame(geo_rows)

        sc_out = OfflineRetailGeoEngine.estimate_synthetic_control(
            geo_df, "store_id", "date", "revenue", ["store_Treatment"], intervention_start_time=40
        )
        g1, g2, g3 = st.columns(3)
        g1.metric("Оценка эффекта акции (ATE)", f"+{sc_out.ate_absolute:.1f} ед. ({sc_out.ate_relative_pct:+.2f}%)")
        g2.metric("Ошибка склейки до теста (Pre-RMSE)", f"{sc_out.pre_intervention_rmse:.2f}")
        g3.metric("p-value", f"{sc_out.p_value_placebo_or_ttest:.5f}")
        st.write("**Оптимальные веса донорских магазинов в Синтетическом Контроле:**", sc_out.donor_weights)

    elif adv_mode.startswith("2."):
        rng_sb = np.random.default_rng(42)
        sb_rows = []
        for c_i in range(40):
            is_tr = (c_i % 2) == 1
            shock = rng_sb.normal(0, 25.0)
            for _ in range(int(rng_sb.integers(40, 100))):
                xf = shock + rng_sb.normal(0, 5.0)
                yv = 100.0 + 0.9 * xf + (8.0 if is_tr else 0.0) + rng_sb.normal(0, 3.0)
                sb_rows.append({"cell_id": f"cell_{c_i}", "variant": "treatment" if is_tr else "control", "x1": xf, "y": yv})
        sb_res = SwitchbackAndClusterEngine.analyze_power_optimal_cupac(
            pd.DataFrame(sb_rows), "cell_id", "variant", "y", ["x1"]
        )
        s1, s2, s3 = st.columns(3)
        s1.metric("Оценка ATE (Power-Optimal CUPAC)", f"{sb_res.ate_power_cupac:+.2f}")
        s2.metric("Сжатие межъячеечной дисперсии", f"{sb_res.variance_reduction_pct:.1f}%")
        s3.metric("p-value после CUPAC", f"{sb_res.p_value_power_cupac:.5f}")

    elif adv_mode.startswith("3."):
        rng_q = np.random.default_rng(77)
        n_q = 1200
        x1, x2 = rng_q.normal(0, 1, n_q), rng_q.normal(0, 1, n_q)
        p_t = 1.0 / (1.0 + np.exp(-(1.1 * x1 + 0.8 * x2)))
        t_q = (rng_q.random(n_q) < p_t).astype(int)
        y_q = 50.0 + 10.0 * x1 + 7.0 * x2 + 12.0 * t_q + rng_q.normal(0, 2.0, n_q)
        q_df = pd.DataFrame({"treatment": t_q, "x1": x1, "x2": x2, "y": y_q})

        psm_res = CausalAndMLDebiasingEngine.propensity_score_matching_and_aipw(
            q_df, "treatment", "y", ["x1", "x2"]
        )
        q1, q2, q3 = st.columns(3)
        q1.metric("Наивная (смещенная) разница", f"{psm_res['naive_biased_diff']:+.2f}")
        q2.metric("Очищенный эффект PSM (ATT)", f"{psm_res['att_psm_matched']:+.2f}")
        q3.metric("Doubly Robust AIPW (ATE)", f"{psm_res['ate_doubly_robust']:+.2f} (p={psm_res['p_value_doubly_robust']:.4f})")

    else:
        b_succ = np.array([25, 55, 110])
        b_trials = np.array([200, 200, 200])
        ts_out = MultiArmedBanditEngine.thompson_sampling_allocation(b_succ, b_trials)
        st.write("**Оптимальные доли трафика по Thompson Sampling для 3 креативов:**", ts_out["optimal_traffic_shares"])
        st.success(f"Рекомендуемый креатив-победитель: Вариант #{ts_out['recommended_arm'] + 1}")
