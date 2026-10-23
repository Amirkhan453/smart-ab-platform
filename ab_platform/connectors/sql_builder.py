import sqlite3
from typing import Dict, Optional
import pandas as pd


class WarehouseSQLBuilder:
    """
    Генератор безопасных аналитических SQL-запросов (ClickHouse / PostgreSQL / SQLite):
    - Строгое правило порядка событий: conversion.event_timestamp >= exposure.exposed_at
    - Triggered Analysis: в расчет попадают только пользователи с зафиксированным событием exposure
    - Автоматический сбор ковариаты X_pre за пред-экспериментальное окно для CUPED
    - Сжатие данных в достаточные статистики (count, sum, sum_of_squares) на стороне БД
    """

    @staticmethod
    def build_user_level_cuped_query(
        experiment_id: str,
        metric_name: str,
        pre_window_days: int = 14,
        dialect: str = "postgres",
    ) -> str:
        """
        Генерирует SQL-запрос, который агрегирует поюзерную витрину эксперимента:
        1) Находит первую точку входа пользователя в эксперимент (first_exposed_at)
        2) Суммирует целевую метрику строго ПОСЛЕ first_exposed_at (защита от инфляции конверсий)
        3) Суммирует ту же метрику ДО first_exposed_at за pre_window_days (ковариата для CUPED)
        """
        if dialect in ("postgres", "clickhouse"):
            pre_interval_expr = f"fe.first_exposed_at - INTERVAL '{pre_window_days}' DAY"
        else:
            pre_interval_expr = f"datetime(fe.first_exposed_at, '-{pre_window_days} days')"

        sql = f"""
WITH first_exposures AS (
    SELECT
        user_id,
        MIN(variant_id) AS variant,
        COUNT(DISTINCT variant_id) AS exposed_variants_cnt,
        MIN(exposed_at) AS first_exposed_at
    FROM exposure_events
    WHERE experiment_id = '{experiment_id}'
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
        AND c.metric_name = '{metric_name}'
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
        AND c.metric_name = '{metric_name}'
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
        return sql

    @staticmethod
    def build_sufficient_stats_query(experiment_id: str, metric_name: str) -> str:
        """
        Генерирует запрос для получения сжатых достаточных статистик (user_count, sum, sum_squared)
        по каждому варианту для мгновенного расчета t-теста Уэлча без выгрузки сырых строк из БД.
        """
        user_query = WarehouseSQLBuilder.build_user_level_cuped_query(
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
        """Выполняет сгенерированный SQL-запрос на подключенной БД и возвращает готовый DataFrame."""
        query = cls.build_user_level_cuped_query(
            experiment_id=experiment_id,
            metric_name=metric_name,
            pre_window_days=pre_window_days,
            dialect="sqlite",
        )
        return pd.read_sql_query(query, conn)
