import hashlib
import json
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ab_platform.schemas.targeting import Targeting


class ExperimentType(str, Enum):
    CLASSIC_AB = "classic_ab"
    CLUSTER_AB = "cluster_ab"
    SWITCHBACK = "switchback"
    GEO_EXPERIMENT = "geo_experiment"
    QUASI_EXPERIMENT = "quasi_experiment"
    # BANDIT исключён в v2.1: многорукие бандиты вне границ текущей версии платформы.


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
    BINARY = "binary"
    CONTINUOUS = "continuous"
    RATIO = "ratio"
    QUANTILE = "quantile"


class Direction(str, Enum):
    INCREASE = "increase"
    DECREASE = "decrease"


class Domain(str, Enum):
    """Домен рандомизации (ТЗ v4, раздел 3)."""
    GUEST = "guest"
    SESSION = "session"
    RESTAURANT = "restaurant"
    RESTAURANT_TIME = "restaurant_time"


class Estimand(str, Enum):
    """Оцениваемая величина (ТЗ v4, 5.6)."""
    ITT = "itt"
    TRIGGERED = "triggered"
    PER_RESTAURANT_MEAN = "per_restaurant_mean"
    NETWORK_TOTAL = "network_total"


class ClusterLevel(str, Enum):
    NONE = "none"
    RESTAURANT = "restaurant"
    TRADE_ZONE = "trade_zone"
    CITY = "city"


class InferenceMethod(str, Enum):
    ASYMPTOTIC = "asymptotic"
    RANDOMIZATION = "randomization"
    WILD_BOOTSTRAP = "wild_bootstrap"


class Pairing(str, Enum):
    NONE = "none"
    PAIRS = "pairs"
    STRATA = "strata"


class QuasiMethod(str, Enum):
    SC = "sc"
    ASCM = "ascm"
    SDID = "sdid"
    DID = "did"
    STAGGERED_DID = "staggered_did"
    MATCHING = "matching"
    IPW = "ipw"
    AIPW = "aipw"
    ITS = "its"
    CAUSAL_IMPACT = "causal_impact"
    C_ARIMA = "c_arima"


class EvidenceLevel(str, Enum):
    """Уровень доказательности (ТЗ v4, 6.1). Авто-SHIP допустим только для A и A−."""
    A = "A"
    A_MINUS = "A-"
    B = "B"
    B_MINUS = "B-"
    C = "C"
    D = "D"


AUTO_SHIP_EVIDENCE = frozenset({EvidenceLevel.A, EvidenceLevel.A_MINUS})

_QUASI_EVIDENCE = {
    QuasiMethod.SC: EvidenceLevel.B, QuasiMethod.ASCM: EvidenceLevel.B, QuasiMethod.SDID: EvidenceLevel.B,
    QuasiMethod.DID: EvidenceLevel.B_MINUS, QuasiMethod.STAGGERED_DID: EvidenceLevel.B_MINUS,
    QuasiMethod.MATCHING: EvidenceLevel.C, QuasiMethod.IPW: EvidenceLevel.C, QuasiMethod.AIPW: EvidenceLevel.C,
    QuasiMethod.ITS: EvidenceLevel.D, QuasiMethod.CAUSAL_IMPACT: EvidenceLevel.D, QuasiMethod.C_ARIMA: EvidenceLevel.D,
}

_DOMAIN_BY_UNIT = {
    RandomizationUnit.USER_ID: Domain.GUEST,
    RandomizationUnit.DEVICE_ID: Domain.GUEST,
    RandomizationUnit.B2B_ACCOUNT_ID: Domain.GUEST,
    RandomizationUnit.SESSION_ID: Domain.SESSION,
    RandomizationUnit.STORE_ID: Domain.RESTAURANT,
    RandomizationUnit.GEO_REGION_ID: Domain.RESTAURANT,
    RandomizationUnit.CLUSTER_TIME_CELL: Domain.RESTAURANT_TIME,
}

# Минимум кластеров на руку для асимптотического инференса и минимум пар (ТЗ v4, 5.5, раздел 9).
MIN_CLUSTERS_PER_ARM_ASYMPTOTIC = 50
MIN_PAIRS = 6
FORBIDDEN_LEVERS_MVP = frozenset({"price", "цена"})
MAX_VARIANT_PARAMS_BYTES = 8192


class PassportStatus(str, Enum):
    """Жизненный цикл (ТЗ v4, 2.2)."""
    DRAFT = "draft"
    REVIEW = "review"
    APPROVED = "approved"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPED = "stopped"
    ANALYSIS = "analysis"
    DECISION = "decision"
    ARCHIVED = "archived"


ALLOWED_TRANSITIONS = {
    PassportStatus.DRAFT: {PassportStatus.REVIEW},
    PassportStatus.REVIEW: {PassportStatus.DRAFT, PassportStatus.APPROVED},
    PassportStatus.APPROVED: {PassportStatus.RUNNING},
    PassportStatus.RUNNING: {PassportStatus.PAUSED, PassportStatus.STOPPED},
    PassportStatus.PAUSED: {PassportStatus.RUNNING, PassportStatus.STOPPED},
    PassportStatus.STOPPED: {PassportStatus.ANALYSIS},
    PassportStatus.ANALYSIS: {PassportStatus.DECISION},
    PassportStatus.DECISION: {PassportStatus.ARCHIVED},
    PassportStatus.ARCHIVED: set(),
}


class MetricDefinition(BaseModel):
    """Строгое определение метрики до старта теста (защита от p-hacking и cherry-picking)."""
    name: str = Field(..., min_length=1, description="Системное имя метрики")
    metric_type: MetricType
    numerator_col: str = Field(..., min_length=1, description="Колонка числителя или самой метрики в БД")
    denominator_col: Optional[str] = Field(
        None, description="Колонка знаменателя (обязательна для RATIO-метрик)"
    )
    expected_direction: Direction = Direction.INCREASE
    winsorize_percentile: Optional[float] = Field(
        0.995, ge=0.90, le=1.0, description="Порог винзоризации выбросов (например, 0.995)"
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
        ..., min_length=3, description="Какой бизнес-риск или каннибализацию страхует этот guardrail"
    )
    auto_stop_on_breach: bool = True


class VariantAllocation(BaseModel):
    variant_id: str = Field(..., min_length=1, description="Например: 'control', 'treatment_1'")
    traffic_percentage: float = Field(..., gt=0.0, le=100.0)
    is_control: bool = False
    params: Dict[str, Any] = Field(
        default_factory=dict, description="Параметры варианта для приложения (ТЗ v4.1, PRM-1); входят в хэш Паспорта"
    )

    @field_validator("params")
    @classmethod
    def _params_are_json(cls, v: Dict[str, Any]) -> Dict[str, Any]:
        try:
            text = json.dumps(v, ensure_ascii=False, allow_nan=False, sort_keys=True)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"params должны быть JSON без NaN и бесконечностей: {exc}") from exc
        if len(text.encode("utf-8")) > MAX_VARIANT_PARAMS_BYTES:
            raise ValueError(f"params больше {MAX_VARIANT_PARAMS_BYTES} байт: это настройки, а не контент.")
        return json.loads(text)  # копия из простых типов JSON: кортежи → списки, порядок ключей не важен


class RiskProfile(BaseModel):
    """Профиль бизнес-рисков, заполняемый ИИ-ассистентом при интервьюировании."""
    has_network_effects: bool = False
    has_cannibalization_risk: bool = False
    reaction_lag_days: int = Field(0, ge=0)
    novelty_effect_expected: bool = False


class ExperimentPassport(BaseModel):
    """
    Главный неизменяемый контракт эксперимента (Pre-Registration Lock), ТЗ v4, раздел 9.

    Модель immutable (frozen=True): любое изменение параметров требует новой версии
    (`new_version`), а не мутации объекта. `model_copy` переопределён так, что результат
    заново проходит все валидаторы. Целостность версии подтверждается хэшем `passport_hash`
    (`compute_hash()` по канонической JSON-сериализации всех полей, кроме хэша, статуса и
    времени создания). Статусы меняются только по допустимым переходам (`with_status`).

    Поля v4 (`domain`, `estimand`, `cluster_level`, `generalization_population`) по умолчанию
    выводятся из единицы рандомизации, чтобы Паспорта v2 оставались валидными.
    """
    model_config = ConfigDict(frozen=True)

    experiment_id: str = Field(..., min_length=1)
    project_id: str = Field("default", min_length=1, description="Идентификатор арендатора/проекта (задел под multi-tenant)")
    version: int = Field(1, ge=1, description="Версия Паспорта; новая версия при любом изменении после Approved")
    status: PassportStatus = Field(PassportStatus.DRAFT, description="Статус жизненного цикла (ТЗ v4, 2.2)")
    passport_hash: Optional[str] = Field(
        None, description="SHA-256 хэш канонической версии Паспорта; вычисляется методом compute_hash()"
    )
    owner: Optional[str] = Field(None, min_length=1, description="Владелец гипотезы (ТЗ v4, 2.1)")
    approved_by: Optional[str] = Field(
        None, min_length=1, description="Кто утвердил; проставляется при переходе в Approved и не может совпадать с владельцем"
    )
    test_key: str = Field(..., min_length=1, description="Уникальный ключ фичи для ортогонального хэширования")
    layer_id: str = Field("default_layer", min_length=1, description="Слой (Plane) для изоляции экспериментов")
    layer_slot_start: int = Field(0, ge=0, lt=10000, description="Начальный бакет-слот в слое [0..10000)")
    layer_slot_end: int = Field(10000, gt=0, le=10000, description="Конечный бакет-слот в слое (0..10000]")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    hypothesis_statement: str = Field(
        ..., min_length=15, description="Формат: Если мы [X], то [метрика Y] изменится на [Z%], потому что [R]"
    )

    experiment_type: ExperimentType = ExperimentType.CLASSIC_AB
    randomization_unit: RandomizationUnit = RandomizationUnit.USER_ID
    analysis_unit: AnalysisUnit = AnalysisUnit.USER
    variants: List[VariantAllocation]

    # ---- Приложение (ТЗ v4.1: TGT-1, TGT-5, IDN-1, LNK-2) ----
    targeting: Targeting = Field(default_factory=Targeting, description="Условия применимости; часть хэша Паспорта")
    targeting_audience_share: Optional[float] = Field(
        None, gt=0.0, le=1.0,
        description="Доля daily_eligible_units, подходящая под таргетинг; обязательна, если условия заданы",
    )
    exposure_surface: Literal["post_login", "pre_login"] = Field(
        "post_login", description="Где гость видит изменение: после входа в приложение или до него"
    )
    ramp_plan: List[float] = Field(
        default_factory=lambda: [100.0], description="План раскатки, % дорожки по возрастанию; старт — с первого шага"
    )

    # ---- Дизайн v4 (ТЗ v4, раздел 9) ----
    domain: Optional[Domain] = Field(None, description="Домен рандомизации; по умолчанию — из randomization_unit")
    estimand: Optional[Estimand] = Field(None, description="Оцениваемая величина; по умолчанию — по домену")
    cluster_level: Optional[ClusterLevel] = Field(
        None, description="Уровень кластеризации SE; для ресторанного домена по умолчанию — restaurant"
    )
    inference: Optional[InferenceMethod] = Field(
        None,
        description=(
            "Метод инференса; по умолчанию для ресторанного домена — randomization (точен по построению; "
            "асимптотический CR2 на истории с тяжёлыми хвостами при 50 на руку давал FPR 6–7,7%, фаза 1), "
            "для гостевого — asymptotic"
        ),
    )
    n_clusters: Optional[int] = Field(
        None, gt=0, description="Число кластеров в пуле (ресторанов); для ресторанного домена по умолчанию — daily_eligible_units"
    )
    pairing: Pairing = Pairing.NONE
    generalization_population: Optional[str] = Field(None, description="На кого обобщается эффект")
    levers: List[str] = Field(default_factory=list, description="Рычаги воздействия (ТЗ v4, 4.2)")
    includes_franchise: bool = Field(False, description="Есть ли в пуле рестораны франчайзи")
    why_not_randomized: Optional[str] = Field(None, description="Обязательно для квазиэкспериментов (принцип 7)")
    quasi_method: Optional[QuasiMethod] = None
    max_double_exposure_share: float = Field(
        0.05, ge=0.0, lt=1.0, description="Доля double exposure, выше которой эксперимент INVALID"
    )

    primary_metric: MetricDefinition
    secondary_metrics: List[MetricDefinition] = Field(default_factory=list)
    guardrail_metrics: List[GuardrailDefinition] = Field(
        ..., min_length=1, description="Минимум 1 Guardrail-метрика обязательна для безопасности бизнеса"
    )

    use_cuped: bool = True
    use_lins_method: bool = False
    covariate_col: Optional[str] = Field("pre_revenue", description="Колонка пред-периода для CUPED")
    cuped_min_r2: float = Field(0.05, ge=0.0, lt=1.0, description="Минимальный R^2 ковариаты для применения CUPED")
    pre_period_window_days: int = Field(14, ge=1, description="Длина пре-периода для ковариаты CUPED, дней")

    baseline_value: float = Field(..., gt=0.0, description="Историческое базовое значение целевой метрики")
    mde_relative: float = Field(..., gt=0.0, description="Минимально детектируемый относительный эффект")
    alpha: float = Field(0.05, gt=0.0, lt=0.5, description="Ошибка I рода (FPR)")
    power: float = Field(0.80, gt=0.5, lt=0.999, description="Статистическая мощность (1 - Beta)")
    sidedness: Literal["two_sided", "one_sided"] = Field(
        "two_sided", description="Сторонность теста для primary-метрики"
    )
    analysis_mode: Literal["fixed_horizon", "sequential"] = Field(
        "fixed_horizon", description="Режим просмотра результатов: фиксированный горизонт или sequential/always-valid"
    )
    practical_significance_threshold: float = Field(
        0.0, ge=0.0, description="Минимальный относительный эффект, который считается практически значимым"
    )
    srm_alpha: float = Field(0.001, gt=0.0, lt=0.5, description="Порог значимости хи-квадрат теста SRM")

    planned_duration_days: int = Field(..., ge=1, description="Планируемая длительность теста в днях")
    daily_eligible_units: int = Field(..., gt=0, description="Доступный трафик единиц рандомизации в день")
    allow_short_duration_override: bool = False
    maturation_days: int = Field(0, ge=0, description="Период дозревания метрик после старта, дней")
    attribution_window_days: int = Field(7, ge=0, description="Окно атрибуции конверсии к показу, дней")

    stratification_cols: List[str] = Field(default_factory=list, description="Колонки страт для блочной рандомизации")
    bot_max_events_per_unit: Optional[int] = Field(
        None, gt=0, description="Порог числа событий на единицу рандомизации для variant-blind фильтрации ботов"
    )

    risk_profile: RiskProfile = Field(default_factory=RiskProfile)

    # ---------------- значения по умолчанию v4 ----------------
    @model_validator(mode="before")
    @classmethod
    def _fill_v4_defaults(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        data = dict(data)
        unit = data.get("randomization_unit", RandomizationUnit.USER_ID)
        try:
            unit = RandomizationUnit(unit)
        except ValueError:
            return data  # ошибку типа выдаст основная валидация
        domain = data.get("domain")
        if domain is None:
            domain = _DOMAIN_BY_UNIT[unit]
            data["domain"] = domain
        domain = Domain(domain)
        clustered = domain in (Domain.RESTAURANT, Domain.RESTAURANT_TIME)
        if data.get("estimand") is None:
            data["estimand"] = Estimand.PER_RESTAURANT_MEAN if clustered else Estimand.ITT
        if data.get("inference") is None:
            data["inference"] = InferenceMethod.RANDOMIZATION if clustered else InferenceMethod.ASYMPTOTIC
        if data.get("cluster_level") is None:
            data["cluster_level"] = ClusterLevel.RESTAURANT if clustered else ClusterLevel.NONE
        if data.get("generalization_population") is None:
            data["generalization_population"] = (
                "Собственные рестораны сети из пула теста" if clustered
                else "Пользователи приложения (идентифицированные гости), не все гости сети"
            )
        return data

    @field_validator("variants")
    @classmethod
    def validate_variants_allocation(cls, variants: List[VariantAllocation]) -> List[VariantAllocation]:
        if len(variants) < 2:
            raise ValueError("Эксперимент должен содержать минимум 2 варианта (Control и Treatment).")
        ids = [v.variant_id for v in variants]
        if len(set(ids)) != len(ids):
            raise ValueError("Идентификаторы вариантов (variant_id) должны быть уникальными.")
        total_pct = sum(v.traffic_percentage for v in variants)
        if abs(total_pct - 100.0) > 1e-2:
            raise ValueError(f"Сумма долей трафика вариантов должна быть ровно 100%, получено: {total_pct}%.")
        controls = [v for v in variants if v.is_control]
        if len(controls) != 1:
            raise ValueError("Должен быть ровно один контрольный вариант (is_control=True).")
        return variants

    @model_validator(mode="after")
    def validate_methodology_and_risks(self) -> "ExperimentPassport":
        if self.layer_slot_start >= self.layer_slot_end:
            raise ValueError("layer_slot_start должен быть строго меньше layer_slot_end.")

        if (
            self.risk_profile.has_network_effects
            and self.experiment_type == ExperimentType.CLASSIC_AB
            and self.randomization_unit in (RandomizationUnit.USER_ID, RandomizationUnit.DEVICE_ID)
        ):
            raise ValueError(
                "Методологическая ошибка (SUTVA Violation): При наличии сетевых эффектов "
                "классический поюзерный A/B даст смещённый результат. "
                "Используйте experiment_type='switchback', 'cluster_ab' или 'geo_experiment'."
            )

        if (
            self.randomization_unit == RandomizationUnit.USER_ID
            and self.analysis_unit in (AnalysisUnit.PAGEVIEW, AnalysisUnit.SESSION, AnalysisUnit.ORDER)
            and self.primary_metric.metric_type != MetricType.RATIO
        ):
            raise ValueError(
                f"Единица рандомизации ({self.randomization_unit.value}) шире единицы анализа "
                f"({self.analysis_unit.value}). Целевая метрика должна иметь тип RATIO."
            )

        if not self.allow_short_duration_override:
            if self.planned_duration_days < 7 or self.planned_duration_days % 7 != 0:
                raise ValueError(
                    f"Задана длительность {self.planned_duration_days} дн. Тест должен идти целое число "
                    "полных недель (7, 14, 21, ...) для исключения эффекта дня недели."
                )

        if self.risk_profile.novelty_effect_expected and self.planned_duration_days < 14:
            raise ValueError(
                "При ожидаемом эффекте новизны (Novelty Effect) минимальная длительность теста должна "
                "составлять не менее 14 дней."
            )

        if self.risk_profile.reaction_lag_days >= self.planned_duration_days:
            raise ValueError(
                f"Лаг реакции клиентов ({self.risk_profile.reaction_lag_days} дн.) больше или равен "
                f"длительности теста ({self.planned_duration_days} дн.)."
            )

        self._validate_v4_design()
        self._validate_app_design()
        return self

    def _validate_app_design(self) -> None:
        """Валидаторы ТЗ v4.1: IDN-1, TGT-1, TGT-5, план раскатки, «второй человек»."""
        guest_unit = self.randomization_unit in (RandomizationUnit.USER_ID, RandomizationUnit.DEVICE_ID)
        if self.exposure_surface == "pre_login" and self.randomization_unit != RandomizationUnit.DEVICE_ID:
            raise ValueError(
                "Экран до входа: гость ещё не известен, единица рандомизации должна быть device_id (ТЗ v4.1, IDN-1)."
            )
        if not self.targeting.is_empty:
            if not guest_unit:
                raise ValueError("Таргетинг по признакам приложения допустим только для единиц user_id и device_id.")
            if self.targeting_audience_share is None:
                raise ValueError(
                    "Заданы условия таргетинга: укажите targeting_audience_share — долю трафика, которая под них "
                    "подходит. Иначе мощность посчитается по всей базе (ТЗ v4.1, TGT-5)."
                )
        elif self.targeting_audience_share is not None:
            raise ValueError("targeting_audience_share задаётся только вместе с условиями таргетинга.")
        plan = self.ramp_plan
        if not plan or any(not (0.0 < x <= 100.0) for x in plan) or any(a >= b for a, b in zip(plan, plan[1:])):
            raise ValueError(f"ramp_plan должен строго возрастать в пределах (0; 100], получено: {plan}.")
        if self.owner is not None and self.owner == self.approved_by:
            raise ValueError("Паспорт не может утвердить его владелец: нужен второй человек (ТЗ v4, 2.2).")

    def _validate_v4_design(self) -> None:
        """Валидаторы ТЗ v4, раздел 9 (1–7, 9; 8 и 10 — в фазах 4 и 8)."""
        expected_domain = _DOMAIN_BY_UNIT[self.randomization_unit]
        if self.domain != expected_domain:
            raise ValueError(
                f"Домен '{self.domain.value}' не соответствует единице рандомизации "
                f"'{self.randomization_unit.value}' (ожидается '{expected_domain.value}')."
            )
        clustered = self.domain in (Domain.RESTAURANT, Domain.RESTAURANT_TIME)
        restaurant_estimands = {Estimand.PER_RESTAURANT_MEAN, Estimand.NETWORK_TOTAL}
        if clustered and self.estimand not in restaurant_estimands:
            raise ValueError("Для ресторанного домена estimand должен быть per_restaurant_mean или network_total.")
        if not clustered and self.estimand in restaurant_estimands:
            raise ValueError("Estimand per_restaurant_mean / network_total допустим только для ресторанного домена.")

        # (1)-(2) кластеры обязательны для ресторанов и для анализа по ресторан-дням.
        if clustered and self.cluster_level == ClusterLevel.NONE:
            raise ValueError(
                "Рандомизация по ресторанам требует кластерного инференса (cluster_level ≥ restaurant): "
                "анализ ресторан-дней как независимых наблюдений даёт 57–69% ложных срабатываний."
            )
        if self.analysis_unit == AnalysisUnit.STORE_DAY and self.cluster_level == ClusterLevel.NONE:
            raise ValueError("analysis_unit = store_day без cluster_level запрещён (ТЗ v4, раздел 9, валидатор 2).")
        if clustered and not (self.use_cuped and self.covariate_col):
            raise ValueError(
                "Для ресторанного домена CUPED по пре-периоду обязателен (ТЗ v4, 5.4): без него MDE ≈ 20% на 100 точках."
            )

        # (3)-(4) малое число кластеров и пары.
        if clustered:
            pool = self.n_clusters if self.n_clusters is not None else self.daily_eligible_units
            min_share = min(v.traffic_percentage for v in self.variants) / 100.0
            per_arm = int(pool * min_share)
            if per_arm < MIN_CLUSTERS_PER_ARM_ASYMPTOTIC and self.inference == InferenceMethod.ASYMPTOTIC:
                raise ValueError(
                    f"В меньшей руке ≈ {per_arm} кластеров (< {MIN_CLUSTERS_PER_ARM_ASYMPTOTIC}): "
                    "асимптотический инференс занижает ошибку. Укажите inference = randomization или wild_bootstrap."
                )
            if self.pairing == Pairing.PAIRS and per_arm < MIN_PAIRS:
                raise ValueError(
                    f"Пар ≈ {per_arm} (< {MIN_PAIRS}): двусторонний рандомизационный тест не может дать "
                    f"p < 0.05 (минимальный p = 2/2^k)."
                )

        # (5) квазиэксперимент.
        if self.experiment_type == ExperimentType.QUASI_EXPERIMENT:
            if not (self.why_not_randomized and len(self.why_not_randomized.strip()) >= 10):
                raise ValueError("Квазиэксперимент требует обоснования why_not_randomized (принцип 7).")
            if self.quasi_method is None:
                raise ValueError("Для квазиэксперимента укажите quasi_method (sdid, ascm, did, matching, its, ...).")
        elif self.quasi_method is not None:
            raise ValueError("quasi_method допустим только при experiment_type = quasi_experiment.")

        # (6)-(7) границы MVP.
        forbidden = {lv.strip().lower() for lv in self.levers} & FORBIDDEN_LEVERS_MVP
        if forbidden:
            raise ValueError("Ценовые тесты вне MVP до решения юристов (ТЗ v4, раздел 15, п. 2 ст. 426 ГК РФ).")
        if self.includes_franchise:
            raise ValueError("Рестораны франчайзи вне MVP (ТЗ v4, решение 7).")

    # ---------------- производные свойства ----------------
    @property
    def evidence_level(self) -> EvidenceLevel:
        """Уровень доказательности по дереву ТЗ v4, 6.1 (задаётся дизайном, а не вручную)."""
        if self.experiment_type == ExperimentType.QUASI_EXPERIMENT:
            return _QUASI_EVIDENCE[self.quasi_method]
        if self.experiment_type == ExperimentType.SWITCHBACK:
            return EvidenceLevel.A_MINUS
        return EvidenceLevel.A

    @property
    def auto_ship_allowed(self) -> bool:
        return self.evidence_level in AUTO_SHIP_EVIDENCE

    @property
    def target_daily_units(self) -> int:
        """Трафик единиц в день с учётом таргетинга — вход калькулятора мощности (ТЗ v4.1, TGT-5)."""
        return int(self.daily_eligible_units * (self.targeting_audience_share or 1.0))

    @property
    def design_warnings(self) -> List[str]:
        """Оговорки дизайна, которые не блокируют Паспорт, но обязаны попасть в форму и отчёт (TGT-3, IDN-1)."""
        out = [f"Таргетинг по признаку, на который может влиять тест, — {w}. Анализ только по первому показу."
               for w in self.targeting.treatment_sensitive]
        if self.randomization_unit == RandomizationUnit.DEVICE_ID:
            out.append(
                "Единица — устройство, а не гость: один гость на двух устройствах может увидеть оба варианта, "
                "эффект относится к устройствам и не склеивается с заказами после входа (ТЗ v4.1, IDN-1)."
            )
        return out

    # ---------------- неизменяемость и версии ----------------
    def model_copy(self, *, update: Optional[Dict[str, Any]] = None, deep: bool = False) -> "ExperimentPassport":
        """Копия с изменениями, которая заново проходит все валидаторы (обход через update запрещён)."""
        data = self.model_dump()
        if update:
            unknown = set(update) - set(type(self).model_fields)
            if unknown:
                raise ValueError(f"Неизвестные поля Паспорта: {sorted(unknown)}")
            data.update(update)
        return type(self).model_validate(data)

    def compute_hash(self) -> str:
        """
        SHA-256 от канонической JSON-сериализации Паспорта без полей
        passport_hash, status и created_at (они не относятся к идентичности
        зафиксированного дизайна эксперимента).
        """
        payload = self.model_dump(mode="json", exclude={"passport_hash", "status", "created_at"})
        canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def with_status(self, new_status: PassportStatus | str, actor: Optional[str] = None) -> "ExperimentPassport":
        """
        Переход статуса только по допустимым переходам ТЗ v4, 2.2.
        actor при утверждении записывается в approved_by; совпадение с владельцем отклоняется.
        """
        new_status = PassportStatus(new_status)
        if new_status not in ALLOWED_TRANSITIONS[self.status]:
            raise ValueError(f"Недопустимый переход статуса: {self.status.value} → {new_status.value}.")
        if new_status == PassportStatus.APPROVED:
            update: Dict[str, Any] = {"status": new_status}
            if actor is not None:
                update["approved_by"] = actor
            sealed = self.model_copy(update=update)
            return sealed.model_copy(update={"passport_hash": sealed.compute_hash()})
        if self.passport_hash is not None and self.passport_hash != self.compute_hash():
            raise ValueError("Хэш Паспорта не совпадает с содержимым: Паспорт изменён после утверждения.")
        return self.model_copy(update={"status": new_status})

    def locked(self) -> "ExperimentPassport":
        """Возвращает версию со статусом 'approved' и проставленным passport_hash (из draft или review)."""
        current = self.with_status(PassportStatus.REVIEW) if self.status == PassportStatus.DRAFT else self
        return current.with_status(PassportStatus.APPROVED)

    def new_version(self, **changes: Any) -> "ExperimentPassport":
        """Новая версия Паспорта с изменениями: version + 1, статус draft, хэш сброшен."""
        if not changes:
            raise ValueError("new_version без изменений не имеет смысла.")
        protected = {"version", "passport_hash", "status", "created_at", "experiment_id", "approved_by"}
        if protected & set(changes):
            raise ValueError(f"Поля {sorted(protected & set(changes))} нельзя менять через new_version.")
        return self.model_copy(update={
            **changes, "version": self.version + 1, "status": PassportStatus.DRAFT,
            "passport_hash": None, "approved_by": None, "created_at": datetime.now(timezone.utc),
        })
