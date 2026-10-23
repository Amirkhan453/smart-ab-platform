import sqlite3
import numpy as np
from ab_platform.connectors.sql_builder import WarehouseSQLBuilder
from ab_platform.engine.assignment import DeterministicAssignmentEngine
from ab_platform.engine.safeguards import ExperimentSafeguardEngine


def test_sql_builder_exposure_ordering_and_cuped_extraction():
    """
    Проверяем на живой in-memory SQLite базе:
    1) Конверсия ДО показа (exposure) не засчитывается в post_metric_value, но идет в pre_metric_value (CUPED)
    2) Конверсия ПОСЛЕ показа засчитывается в post_metric_value
    3) Пользователи с Double Exposure (2 разных варианта) автоматически отфильтровываются
    """
    conn = sqlite3.connect(":memory:")
    cur = conn.cursor()
    cur.execute(
        "CREATE TABLE exposure_events (event_id TEXT, user_id TEXT, experiment_id TEXT, variant_id TEXT, exposed_at TEXT);"
    )
    cur.execute(
        "CREATE TABLE conversion_events (event_id TEXT, user_id TEXT, metric_name TEXT, metric_value REAL, event_timestamp TEXT);"
    )

    # u1: нормальный юзер в control (покупка на 100 до теста и на 250 после показа)
    # u2: юзер с Double Exposure (и в control, и в treatment) -> должен быть исключен
    cur.executemany(
        "INSERT INTO exposure_events VALUES (?, ?, ?, ?, ?);",
        [
            ("e1", "u1", "exp_checkout", "control", "2026-10-10 12:00:00"),
            ("e2", "u2", "exp_checkout", "control", "2026-10-10 12:05:00"),
            ("e3", "u2", "exp_checkout", "treatment", "2026-10-10 12:06:00"),
        ],
    )
    cur.executemany(
        "INSERT INTO conversion_events VALUES (?, ?, ?, ?, ?);",
        [
            ("c1", "u1", "revenue", 100.0, "2026-10-05 10:00:00"),  # До exposure -> пойдет в pre_metric_value
            ("c2", "u1", "revenue", 250.0, "2026-10-10 12:30:00"),  # После exposure -> пойдет в post_metric_value
            ("c3", "u2", "revenue", 999.0, "2026-10-10 13:00:00"),
        ],
    )
    conn.commit()

    df = WarehouseSQLBuilder.execute_on_sqlite_connection(conn, "exp_checkout", "revenue")
    assert len(df) == 1
    assert df.iloc[0]["user_id"] == "u1"
    assert df.iloc[0]["pre_metric_value"] == 100.0
    assert df.iloc[0]["post_metric_value"] == 250.0
    assert df.iloc[0]["converted"] == 1
    conn.close()


def test_deterministic_assignment_monotonic_rampup_and_orthogonality():
    """
    Проверяем:
    1) Если юзер попал в эксперимент при раскатке 20%, то при увеличении раскатки до 80% его вариант НЕ меняется
    2) Два разных эксперимента статистически независимы (ортогональны)
    """
    for i in range(500):
        uid = f"user_{i}"
        res_20 = DeterministicAssignmentEngine.assign_user(uid, "exp_search_v1", exposure_rate_pct=20.0)
        res_80 = DeterministicAssignmentEngine.assign_user(uid, "exp_search_v1", exposure_rate_pct=80.0)
        if res_20.is_exposed:
            assert res_80.is_exposed is True
            assert res_20.variant_id == res_80.variant_id

    ortho = DeterministicAssignmentEngine.validate_orthogonality_and_uniformity(
        "exp_ui_layer", "exp_algo_layer", n_sim_users=10000
    )
    assert ortho["is_orthogonal_and_valid"] is True


def test_sprt_srm_stage_diagnosis_and_guardrails():
    """
    Проверяем:
    1) Детекцию SRM на этапе Triggered при честном Assigned (кейс eBay с латентностью и ботами)
    2) Авто-остановку теста при пробое Guardrail-метрики
    """
    diag = ExperimentSafeguardEngine.diagnose_srm_stage(
        assigned_control=50000,
        assigned_treatment=50050,   # Честный сплит 50/50
        triggered_control=40000,
        triggered_treatment=36000,  # Сильный отвал в тесте из-за бага/латентности
    )
    assert diag["assigned_srm"] is False
    assert diag["triggered_srm"] is True
    assert "Trigger/Execution Stage SRM" in diag["diagnosis"]

    psi_res = ExperimentSafeguardEngine.calculate_psi(
        np.array([1000, 2000, 3000]), np.array([1010, 1990, 3005])
    )
    assert psi_res["is_balanced"] is True

    guard_res = ExperimentSafeguardEngine.evaluate_guardrail_breach(
        baseline_control=0.10,
        treatment_value=0.092,  # Падение конверсии на -8% при пороге 3%
        p_value=0.001,
        max_allowed_degradation_rel=0.03,
        higher_is_better=True,
    )
    assert guard_res["guardrail_breached"] is True
    assert guard_res["action"] == "AUTO_STOP_AND_REVERT"
