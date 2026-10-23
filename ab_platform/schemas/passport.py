from datetime import datetime, timezone
from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field, field_validator, model_validator


class ExperimentType(str, Enum):
    CLASSIC_AB = "classic_ab"             # Онлайн-сплит по независимым юзерам
    CLUSTER_AB = "cluster_ab"             # Сплит по кластерам (магазины, B2B-компании, соц. граф)
    SWITCHBACK = "switchback"             # Сплит Регион x Время (такси, доставка, динамические цены)
    GEO_EXPERIMENT = "geo_experiment"     # Оффлайн гео-сплит / Matched Markets / Synthetic Control
    QUASI_EXPERIMENT = "quasi_experiment" # PSM / DiD на исторических данных без рандомизации
    BANDIT = "multi_armed_bandit"         # Динамическая аллокация трафика для быстрой оптимизации


class RandomizationUnit(str, Enum):
    USER_ID = "user_id"
    DEVICE_ID = "device_id"
    SESSION_ID = "session_id"
    STORE_ID = "store_id"
    GEO_REGION_ID = "geo_region_id"
    CLUSTER_TIME_CELL = "cluster_time_cell"
    B2B_ACCOUNT_ID = "b2b_account_id"


class AnalysisUnit(str, Enum):
    USER = "user"
    SESSION = "session"
    PAGEVIEW = "pageview"
    ORDER = "order"
    STORE_DAY = "store_day"
    CLUSTER_CELL = "cluster_cell"


class MetricType(str, Enum):
    BINARY = "binary"           # Конверсия (0/1), CR
    CONTINUOUS = "continuous"   # Выручка на юзера (ARPU), время в приложении
    RATIO = "ratio"             # CTR (клики/просмотры), AOV (выручка/заказы)
    QUANTILE = "quantile"       # p90, p95 времени ответа или чека


class Direction(str, Enum):
    INCREASE = "increase"
    DECREASE = "decrease"


class MetricDefinition(BaseModel):
    """Строгое определение метрики до старта теста (защита от p-hacking и cherry-picking)."""
    name: str = Field(..., description="Системное имя метрики, например 'checkout_cr'")
    metric_type: MetricType
    numerator_col: str = Field(..., description="Колонка числителя или самой метрики в БД")
    denominator_col: Optional[str] = Field(
        None, description="Колонка знаменателя (обязательна для RATIO-метрик)"
    )
    expected_direction: Direction = Direction.INCREASE
    winsorize_percentile: Optional[float] = Field(
        0.995, ge=0.90, le=1.0, description="Порог винзоризации выбросов (например, 0.995 или 0.999)"
    )

    @model_validator(mode="after")
    def check_ratio_denominator(self) -> "MetricDefinition":
        if self.metric_type == MetricType.RATIO and not self.denominator_col:
            raise ValueError(
                f"Метрика '{self.name}' имеет тип RATIO: необходимо указать 'denominator_col'."
            )
        return self


class GuardrailDefinition(BaseModel):
    """Защитная метрика с жестким числовым порогом деградации для авто-остановки теста."""
    metric: MetricDefinition
    max_allowed_degradation_rel: float = Field(
        ..., gt=0.0, description="Максимально допустимое относительное ухудшение (например, 0.02 = 2%)"
    )
    tradeoff_rationale: str = Field(
        ..., description="Какой бизнес-риск или каннибализацию страхует этот guardrail"
    )
    auto_stop_on_breach: bool = True


class VariantAllocation(BaseModel):
    variant_id: str = Field(..., description="Например: 'control', 'treatment_1'")
    traffic_percentage: float = Field(..., gt=0.0, le=100.0)
    is_control: bool = False


class RiskProfile(BaseModel):
    """Профиль бизнес-рисков, заполняемый ИИ-ассистентом при интервьюировании."""
    has_network_effects: bool = Field(
        False, description="Влияют ли пользователи группы B на ресурсы/поведение группы A (SUTVA)"
    )
    has_cannibalization_risk: bool = Field(
        False, description="Может ли фича перетянуть выручку из соседних товаров/категорий"
    )
    reaction_lag_days: int = Field(
        0, ge=0, description="Лаг созревания целевого действия клиента в днях"
    )
    novelty_effect_expected: bool = Field(
        False, description="Ожидается ли эффект новизны в первые дни после выкатки"
    )


class ExperimentPassport(BaseModel):
    """
    Главный неизменяемый контракт эксперимента (Pre-Registration Lock).
    Защищает бизнес от методологических ошибок еще ДО запуска сплитования.
    """
    experiment_id: str
    test_key: str = Field(..., description="Уникальный ключ фичи для ортогонального хэширования")
    layer_id: str = Field("default_layer", description="Слой (Plane) для изоляции пересекающихся тестов")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    # 1. Структурированная гипотеза
    hypothesis_statement: str = Field(
        ..., min_length=15, description="Формат: Если мы [X], то [метрика Y] изменится на [Z%], потому что [R]"
    )

    # 2. Архитектура рандомизации и анализа
    experiment_type: ExperimentType = ExperimentType.CLASSIC_AB
    randomization_unit: RandomizationUnit = RandomizationUnit.USER_ID
    analysis_unit: AnalysisUnit = AnalysisUnit.USER
    variants: List[VariantAllocation]

    # 3. Метрики и пре-регистрация MDE
    primary_metric: MetricDefinition
    secondary_metrics: List[MetricDefinition] = Field(default_factory=list)
    guardrail_metrics: List[GuardrailDefinition] = Field(
        ..., min_length=1, description="Минимум 1 Guardrail-метрика обязательна для безопасности бизнеса"
    )

    baseline_value: float = Field(..., gt=0.0, description="Историческое базовое значение целевой метрики")
    mde_relative: float = Field(
        ..., gt=0.0, description="Минимально детектируемый относительный эффект (например, 0.03 для +3%)"
    )
    alpha: float = Field(0.05, gt=0.0, lt=0.5, description="Ошибка I рода (FPR)")
    power: float = Field(0.80, gt=0.5, lt=0.999, description="Статистическая мощность (1 - Beta)")

    # 4. Ограничения длительности и трафика
    planned_duration_days: int = Field(..., ge=1, description="Планируемая длительность теста в днях")
    daily_eligible_units: int = Field(..., gt=0, description="Доступный трафик единиц рандомизации в день")
    allow_short_duration_override: bool = Field(
        False, description="Технический флаг только для экстренных smoke-тестов (<7 дней)"
    )

    # 5. Профиль рисков
    risk_profile: RiskProfile = Field(default_factory=RiskProfile)

    @field_validator("variants")
    @classmethod
    def validate_variants_allocation(cls, variants: List[VariantAllocation]) -> List[VariantAllocation]:
        if len(variants) < 2:
            raise ValueError("Эксперимент должен содержать минимум 2 варианта (Control и Treatment).")
        total_pct = sum(v.traffic_percentage for v in variants)
        if abs(total_pct - 100.0) > 1e-4:
            raise ValueError(f"Сумма долей трафика вариантов должна быть ровно 100%, получено: {total_pct}%.")
        controls = [v for v in variants if v.is_control]
        if len(controls) != 1:
            raise ValueError("Должен быть ровно один контрольный вариант (is_control=True).")
        return variants

    @model_validator(mode="after")
    def validate_methodology_and_risks(self) -> "ExperimentPassport":
        # Правило 1: Защита от нарушения SUTVA (сетевых эффектов)
        if (
            self.risk_profile.has_network_effects
            and self.experiment_type == ExperimentType.CLASSIC_AB
            and self.randomization_unit in (RandomizationUnit.USER_ID, RandomizationUnit.DEVICE_ID)
        ):
            raise ValueError(
                "Методологическая ошибка (SUTVA Violation): При наличии сетевых эффектов (конкуренция за "
                "курьеров/товары/соц. граф) классический поюзерный A/B даст смещенный результат. "
                "Используйте experiment_type='switchback', 'cluster_ab' или 'geo_experiment'."
            )

        # Правило 2: Если единица рандомизации и анализа не совпадают (например, юзеры vs просмотры страниц),
        # для корректного учета внутрипользовательской корреляции требуется RATIO-метрика (Дельта-метод)
        if (
            self.randomization_unit == RandomizationUnit.USER_ID
            and self.analysis_unit in (AnalysisUnit.PAGEVIEW, AnalysisUnit.SESSION, AnalysisUnit.ORDER)
            and self.primary_metric.metric_type != MetricType.RATIO
        ):
            raise ValueError(
                f"Единица рандомизации ({self.randomization_unit.value}) шире единицы анализа "
                f"({self.analysis_unit.value}). Чтобы избежать занижения дисперсии из-за повторных действий "
                "одного юзера, целевая метрика должна иметь тип RATIO (для применения Дельта-метода/кластеризации)."
            )

        # Правило 3: Защита от недельной сезонности и эффекта новизны (минимум 7 дней)
        if self.planned_duration_days < 7 and not self.allow_short_duration_override:
            raise ValueError(
                f"Задана длительность {self.planned_duration_days} дн. Тест должен идти минимум 7 дней "
                "(полный недельный цикл) для исключения эффекта дня недели. Если применяется эффект новизны, "
                "рекомендуется минимум 14 дней."
            )

        if self.risk_profile.novelty_effect_expected and self.planned_duration_days < 14:
            raise ValueError(
                "При ожидаемом эффекте новизны (Novelty Effect) минимальная длительность теста должна "
                "составлять не менее 14 дней для стабилизации поведения пользователей."
            )

        # Правило 4: Лаг созревания метрики не может превышать окно наблюдения
        if self.risk_profile.reaction_lag_days >= self.planned_duration_days:
            raise ValueError(
                f"Лаг реакции клиентов ({self.risk_profile.reaction_lag_days} дн.) больше или равен "
                f"длительности теста ({self.planned_duration_days} дн.). Используйте быстрые суррогатные "
                "(прокси) метрики или увеличьте длительность."
            )

        return self


