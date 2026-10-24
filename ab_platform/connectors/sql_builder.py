import re
import sqlite3
from typing import Any, Dict, Tuple
import pandas as pd


class WarehouseSQLBuilder:
    """
    Защищенный от SQL-инъекций генератор аналитических запросов (PostgreSQL / ClickHouse / SQLite):
    - Строгая проверка всех идентификаторов и ключей по белому списку (Regex Whitelist)
    - Параметризованные запросы (Prepared Statements с биндингом параметров ? / %(param)s)
    - Строгое правило порядка событий: conversion.event_timestamp >= exposure.exposed_at
    """

    SAFE_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
    SAFE_KEY_TOKEN_RE = re.compile(r"^[A-Za-z0-9_.:-]+$")

    @classmethod
    def validate_safe_identifier(cls, name: str, field_label: str = "identifier") -> str:
        if not isinstance(name, str) or not cls.SAFE_IDENTIFIER_RE.match(name):
            raise ValueError(
                f"Security Error [SQL Injection Guard]: Недопустимое имя SQL-идентификатора "
                f"в поле '{field_label}': {name!r}. Разрешены только [A-Za-z0-9_]."
            )
        return name

    @classmethod
    def validate_safe_key(cls, key_val: str, field_label: str = "key") -> str:
        if not isinstance(key_val, str) or not cls.SAFE_KEY_TOKEN_RE.match(key_val):
            raise ValueError(
                f"Security Error [SQL Injection Guard]: Обнаружены опасные символы в '{field_label}': {key_val!r}."
            )
        return key_val

    @classmethod
    def build_parameterized_cuped_query(
        cls,
        experiment_id: str,
        metric_name: str,
        pre_window_days: int = 14,
        dialect: str = "postgres",
    ) -> Tuple[str, Dict[str, Any]]:
        cls.validate_safe_key(experiment_id, "experiment_id")
        cls.validate_safe_identifier(metric_name, "metric_name")
        if not isinstance(pre_window_days, int) or pre_window_days <= 0 or pre_window_days > 3650:
            raise ValueError("pre_window_days должен быть целым числом в диапазоне [1, 3650].")
        if dialect not in ("postgres", "clickhouse", "sqlite"):
            raise ValueError(f"Неподдерживаемый диалект SQL: {dialect!r}")

        if dialect in ("postgres", "clickhouse"):
            exp_ph = "%(experiment_id)s"
            met_ph = "%(metric_name)s"
            pre_interval_expr = f"fe.first_exposed_at - INTERVAL '{int(pre_window_days)}' DAY"
        else:
            exp_ph = ":experiment_id"
            met_ph = ":metric_name"
            pre_interval_expr = "datetime(fe.first_exposed_at, :pre_offset)"

        sql = f"""
WITH first_exposures AS (
    SELECT
        user_id,
        MIN(variant_id) AS variant,
        COUNT(DISTINCT variant_id) AS exposed_variants_cnt,
        MIN(exposed_at) AS first_exposed_at
    FROM exposure_events
    WHERE experiment_id = {exp_ph}
    GROUP BY user_id
),
post_metrics AS (
    SELECT
        fe.user_id,
        SUM(c.metric_value) AS post_metric_value,
        COUNT(c.event_id) AS post_events_count
    FROM first_exposures fe
    INNER JOIN conversion_events c
        ON fe.user_id = c.user_id
        AND c.metric_name = {met_ph}
        AND c.event_timestamp >= fe.first_exposed_at
    GROUP BY fe.user_id
),
pre_covariates AS (
    SELECT
        fe.user_id,
        SUM(c.metric_value) AS pre_metric_value
    FROM first_exposures fe
    INNER JOIN conversion_events c
        ON fe.user_id = c.user_id
        AND c.metric_name = {met_ph}
        AND c.event_timestamp < fe.first_exposed_at
        AND c.event_timestamp >= {pre_interval_expr}
    GROUP BY fe.user_id
)
SELECT
    fe.user_id,
    fe.variant,
    fe.exposed_variants_cnt,
    COALESCE(pre.pre_metric_value, 0.0) AS pre_metric_value,
    COALESCE(post.post_metric_value, 0.0) AS post_metric_value,
    CASE WHEN COALESCE(post.post_events_count, 0) > 0 THEN 1 ELSE 0 END AS converted
FROM first_exposures fe
LEFT JOIN pre_covariates pre ON fe.user_id = pre.user_id
LEFT JOIN post_metrics post ON fe.user_id = post.user_id
WHERE fe.exposed_variants_cnt = 1;
""".strip()

        params: Dict[str, Any] = {
            "experiment_id": experiment_id,
            "metric_name": metric_name,
            "pre_offset": f"-{int(pre_window_days)} days",
        }
        return sql, params

    @classmethod
    def build_user_level_cuped_query(
        cls,
        experiment_id: str,
        metric_name: str,
        pre_window_days: int = 14,
        dialect: str = "postgres",
    ) -> str:
        cls.validate_safe_key(experiment_id, "experiment_id")
        cls.validate_safe_identifier(metric_name, "metric_name")
        sql_template, _ = cls.build_parameterized_cuped_query(
            experiment_id=experiment_id,
            metric_name=metric_name,
            pre_window_days=pre_window_days,
            dialect=dialect,
        )
        if dialect in ("postgres", "clickhouse"):
            return (
                sql_template.replace("%(experiment_id)s", f"'{experiment_id}'")
                .replace("%(metric_name)s", f"'{metric_name}'")
            )
        return (
            sql_template.replace(":experiment_id", f"'{experiment_id}'")
            .replace(":metric_name", f"'{metric_name}'")
            .replace(":pre_offset", f"'-{int(pre_window_days)} days'")
        )

    @classmethod
    def build_sufficient_stats_query(cls, experiment_id: str, metric_name: str) -> str:
        user_query = cls.build_user_level_cuped_query(
            experiment_id=experiment_id, metric_name=metric_name, dialect="sqlite"
        ).rstrip(";")

        return f"""
WITH user_level AS (
{user_query}
)
SELECT
    variant,
    COUNT(user_id) AS user_count,
    SUM(post_metric_value) AS sum_value,
    SUM(post_metric_value * post_metric_value) AS sum_squared_value,
    SUM(converted) AS conversion_count
FROM user_level
GROUP BY variant;
""".strip()

    @classmethod
    def execute_on_sqlite_connection(
        cls,
        conn: sqlite3.Connection,
        experiment_id: str,
        metric_name: str,
        pre_window_days: int = 14,
    ) -> pd.DataFrame:
        sql_template, params = cls.build_parameterized_cuped_query(
            experiment_id=experiment_id,
            metric_name=metric_name,
            pre_window_days=pre_window_days,
            dialect="sqlite",
        )
        return pd.read_sql_query(sql_template, conn, params=params)
