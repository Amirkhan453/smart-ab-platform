from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set
from ab_platform.schemas.passport import (
    AnalysisUnit,
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


@dataclass
class ArborEdge:
    source_node: str
    target_node: str
    condition_description: str
    predicate: Callable[[Dict[str, Any]], bool]


@dataclass
class ArborNode:
    node_id: str
    question_prompt: str
    is_terminal: bool = False


@dataclass
class CopilotSessionState:
    current_node: str = "node_start_randomization_check"
    step_count: int = 0
    max_steps: int = 15
    collected_context: Dict[str, Any] = field(default_factory=dict)
    candidate_designs: Set[ExperimentType] = field(
        default_factory=lambda: {
            ExperimentType.CLASSIC_AB,
            ExperimentType.CLUSTER_AB,
            ExperimentType.SWITCHBACK,
            ExperimentType.GEO_EXPERIMENT,
            ExperimentType.QUASI_EXPERIMENT,
            ExperimentType.BANDIT,
        }
    )
    warnings: List[str] = field(default_factory=list)
    recommended_design: Optional[ExperimentType] = None
    recommended_randomization_unit: Optional[RandomizationUnit] = None

    @property
    def p_suff(self) -> float:
        """Сигнал достаточности информации: p_suff = 1 / |candidates|."""
        if not self.candidate_designs:
            return 0.0
        return round(1.0 / len(self.candidate_designs), 4)


class ArborDecisionCopilot:
    """
    Движок графа решений Arbor для ИИ-ассистента A/B-платформы:
    - Структурная валидация графа (отсутствие узлов-сирот, битых ссылок и циклов)
    - Локальная оценка переходов (Step 1: Evaluation) отдельно от генерации (Step 2: Generation)
    - Автоматический подбор дизайна теста, детекция SUTVA/каннибализации/лага и сборка Паспорта
    """

    def __init__(self) -> None:
        self.nodes: Dict[str, ArborNode] = {}
        self.edges: List[ArborEdge] = []
        self._build_default_graph()
        self.validate_graph_integrity()

    def _build_default_graph(self) -> None:
        self.nodes = {
            "node_start_randomization_check": ArborNode(
                node_id="node_start_randomization_check",
                question_prompt="Можем ли мы технически и юридически рандомизировать воздействие (сплитовать трафик/точки) или оцениваем уже случившийся исторический факт?",
            ),
            "node_environment_check": ArborNode(
                node_id="node_environment_check",
                question_prompt="Где проводится эксперимент: в цифровом продукте (Онлайн) или в физических точках/регионах (Оффлайн)?",
            ),
            "node_network_effects_check": ArborNode(
                node_id="node_network_effects_check",
                question_prompt="Есть ли сетевые эффекты (SUTVA violation): конкурируют ли пользователи групп A и B за общий пул курьеров/товаров/цен или общаются ли в соц. графе?",
            ),
            "node_interference_structure": ArborNode(
                node_id="node_interference_structure",
                question_prompt="Природа интерференции динамическая во времени в рамках одного города (такси/доставка/цены) или кластерная по группам юзеров/компаниям?",
            ),
            "node_traffic_and_goal_check": ArborNode(
                node_id="node_traffic_and_goal_check",
                question_prompt="Достаточно ли трафика для фиксации MDE за <=35 дней, или цель — быстро максимизировать конверсию на короткой акции?",
            ),
            "terminal_quasi": ArborNode(
                node_id="terminal_quasi",
                question_prompt="Рекомендован Квази-эксперимент (Propensity Score Matching / Difference-in-Differences / Synthetic Control).",
                is_terminal=True,
            ),
            "terminal_geo": ArborNode(
                node_id="terminal_geo",
                question_prompt="Рекомендован Оффлайн Гео-эксперимент / Matched Markets с подбором пар магазинов/городов и периодом охлаждения.",
                is_terminal=True,
            ),
            "terminal_switchback": ArborNode(
                node_id="terminal_switchback",
                question_prompt="Рекомендован Switchback-эксперимент (Регион x Время) с макро-взвешенным Power-Optimal CUPAC.",
                is_terminal=True,
            ),
            "terminal_cluster": ArborNode(
                node_id="terminal_cluster",
                question_prompt="Рекомендован Кластерный A/B-тест (рандомизация по кластерам/B2B-аккаунтам) с оценкой дисперсии через Дельта-метод или CR2.",
                is_terminal=True,
            ),
            "terminal_bandit": ArborNode(
                node_id="terminal_bandit",
                question_prompt="Рекомендован Многорукий Бандит (Thompson Sampling / Contextual Bandit) для минимизации упущенной выгоды (regret).",
                is_terminal=True,
            ),
            "terminal_classic_ab": ArborNode(
                node_id="terminal_classic_ab",
                question_prompt="Рекомендован Классический поюзерный A/B-тест с хэшированием MurmurHash3 и снижением дисперсии CUPED/Lin.",
                is_terminal=True,
            ),
        }

        self.edges = [
            ArborEdge(
                source_node="node_start_randomization_check",
                target_node="terminal_quasi",
                condition_description="Рандомизация невозможна (исторические логи)",
                predicate=lambda ctx: ctx.get("can_randomize") is False,
            ),
            ArborEdge(
                source_node="node_start_randomization_check",
                target_node="node_environment_check",
                condition_description="Рандомизация возможна",
                predicate=lambda ctx: ctx.get("can_randomize") is True,
            ),
            ArborEdge(
                source_node="node_environment_check",
                target_node="terminal_geo",
                condition_description="Оффлайн среда (розничные магазины, города, ТВ/наружная реклама)",
                predicate=lambda ctx: ctx.get("is_offline") is True,
            ),
            ArborEdge(
                source_node="node_environment_check",
                target_node="node_network_effects_check",
                condition_description="Онлайн среда (веб, мобильное приложение, бэкенд)",
                predicate=lambda ctx: ctx.get("is_offline") is False,
            ),
            ArborEdge(
                source_node="node_network_effects_check",
                target_node="node_interference_structure",
                condition_description="Обнаружены сетевые эффекты / конкуренция за общий ресурс",
                predicate=lambda ctx: ctx.get("has_network_effects") is True,
            ),
            ArborEdge(
                source_node="node_network_effects_check",
                target_node="node_traffic_and_goal_check",
                condition_description="Сетевых эффектов нет (наблюдения независимы, SUTVA выполнена)",
                predicate=lambda ctx: ctx.get("has_network_effects") is False,
            ),
            ArborEdge(
                source_node="node_interference_structure",
                target_node="terminal_switchback",
                condition_description="Рыночный баланс спроса/предложения меняется во времени (такси, доставка, прайсинг)",
                predicate=lambda ctx: ctx.get("interference_type") == "temporal_marketplace",
            ),
            ArborEdge(
                source_node="node_interference_structure",
                target_node="terminal_cluster",
                condition_description="Социальный граф, B2B-организации или изолированные кластеры юзеров",
                predicate=lambda ctx: ctx.get("interference_type") == "social_or_b2b_cluster",
            ),
            ArborEdge(
                source_node="node_traffic_and_goal_check",
                target_node="terminal_bandit",
                condition_description="Короткая кампания / мало трафика для мелкого MDE / цель — максимизация выгоды на лету",
                predicate=lambda ctx: ctx.get("prefer_regret_minimization") is True,
            ),
            ArborEdge(
                source_node="node_traffic_and_goal_check",
                target_node="terminal_classic_ab",
                condition_description="Нужно строгое статистическое доказательство эффекта (научный вывод)",
                predicate=lambda ctx: ctx.get("prefer_regret_minimization") is False,
            ),
        ]

    def validate_graph_integrity(self) -> bool:
        """
        Офлайн-валидация структуры дерева решений:
        1) Reference integrity (все ребра ведут в существующие узлы)
        2) Orphan detection (все нестартовые узлы достижимы из корня)
        3) Cycle detection (отсутствие бесконечных петель в графе через DFS)
        """
        root_id = "node_start_randomization_check"
        if root_id not in self.nodes:
            raise ValueError("Корневой узел отсутствует в графе Arbor.")

        adj: Dict[str, List[str]] = {nid: [] for nid in self.nodes}
        for edge in self.edges:
            if edge.source_node not in self.nodes or edge.target_node not in self.nodes:
                raise ValueError(f"Битая ссылка в ребре: {edge.source_node} -> {edge.target_node}")
            adj[edge.source_node].append(edge.target_node)

        # Поиск достижимых узлов (Orphan detection) и проверка на циклы
        visited: Set[str] = set()
        visiting: Set[str] = set()

        def dfs(node_id: str) -> None:
            visiting.add(node_id)
            visited.add(node_id)
            for neighbor in adj[node_id]:
                if neighbor in visiting:
                    raise ValueError(f"Обнаружен цикл в графе решений Arbor на узле '{neighbor}'.")
                if neighbor not in visited:
                    dfs(neighbor)
            visiting.remove(node_id)

        dfs(root_id)

        orphans = set(self.nodes.keys()) - visited
        if orphans:
            raise ValueError(f"Обнаружены недостижимые узлы-сироты (Orphan nodes): {orphans}")

        return True

    def advance_tree(self, state: CopilotSessionState, user_updates: Dict[str, Any]) -> CopilotSessionState:
        """
        Продвигает диалог по графу Arbor на основе новых ответов пользователя,
        обновляет кандидатов, считает сигнал достаточности p_suff и проверяет риски.
        """
        state.collected_context.update(user_updates)

        while not self.nodes[state.current_node].is_terminal:
            if state.step_count >= state.max_steps:
                raise RuntimeError("Превышен лимит шагов диалога (max_steps guard). Требуется вмешательство человека.")

            outgoing = [e for e in self.edges if e.source_node == state.current_node]
            matched_edge = None
            for edge in outgoing:
                if edge.predicate(state.collected_context):
                    matched_edge = edge
                    break

            if matched_edge is None:
                break

            state.current_node = matched_edge.target_node
            state.step_count += 1
            self._prune_candidates(state, matched_edge.target_node)

        self._evaluate_business_risks(state)
        return state

    def _prune_candidates(self, state: CopilotSessionState, reached_node: str) -> None:
        terminal_map = {
            "terminal_quasi": (ExperimentType.QUASI_EXPERIMENT, RandomizationUnit.USER_ID),
            "terminal_geo": (ExperimentType.GEO_EXPERIMENT, RandomizationUnit.STORE_ID),
            "terminal_switchback": (ExperimentType.SWITCHBACK, RandomizationUnit.CLUSTER_TIME_CELL),
            "terminal_cluster": (ExperimentType.CLUSTER_AB, RandomizationUnit.B2B_ACCOUNT_ID),
            "terminal_bandit": (ExperimentType.BANDIT, RandomizationUnit.USER_ID),
            "terminal_classic_ab": (ExperimentType.CLASSIC_AB, RandomizationUnit.USER_ID),
        }
        if reached_node in terminal_map:
            exp_type, rand_unit = terminal_map[reached_node]
            state.candidate_designs = {exp_type}
            state.recommended_design = exp_type
            state.recommended_randomization_unit = rand_unit
        elif reached_node == "node_environment_check":
            state.candidate_designs.discard(ExperimentType.QUASI_EXPERIMENT)
        elif reached_node == "node_network_effects_check":
            state.candidate_designs.discard(ExperimentType.GEO_EXPERIMENT)
        elif reached_node == "node_interference_structure":
            state.candidate_designs = {ExperimentType.SWITCHBACK, ExperimentType.CLUSTER_AB}
        elif reached_node == "node_traffic_and_goal_check":
            state.candidate_designs = {ExperimentType.CLASSIC_AB, ExperimentType.BANDIT}

    def _evaluate_business_risks(self, state: CopilotSessionState) -> None:
        ctx = state.collected_context
        warnings = []

        daily_traffic = ctx.get("daily_traffic", 10000)
        if daily_traffic * 30 < 200000 and ctx.get("mde_relative", 0.05) < 0.05:
            warnings.append(
                "Эвристика масштаба (Kohavi Rule): При аудитории <200 000 юзеров детектировать мелкие эффекты (<5%) "
                "в классическом A/B нерентабельно. Сфокусируйтесь на крупных изменениях (MDE 5-15%) или включите CUPED."
            )

        if ctx.get("has_cannibalization_risk"):
            warnings.append(
                "Риск каннибализации: Фича может перетянуть продажи из смежных товаров/категорий. "
                "Обязательно добавьте выручку смежных категорий и общую маржу в Guardrail-метрики."
            )

        if ctx.get("reaction_lag_days", 0) > 7:
            warnings.append(
                f"Длинный лаг созревания метрики ({ctx['reaction_lag_days']} дн.): "
                "Рекомендуется сформировать быстрый суррогатный критерий OEC (Overall Evaluation Criterion)."
            )

        state.warnings = warnings

    @staticmethod
    def compute_site_wide_impact(
        local_lift_rel: float, triggered_share_alpha: float
    ) -> dict:
        """
        Формула Site Wide Impact (SWI = Delta * alpha) из архитектуры LinkedIn XLNT.
        Переводит локальный эффект в группе триггера на масштаб всей компании.
        """
        if not (0.0 <= triggered_share_alpha <= 1.0):
            raise ValueError("Доля затронутого трафика alpha должна быть в диапазоне [0, 1].")
        swi_rel = local_lift_rel * triggered_share_alpha
        return {
            "local_lift_pct": round(local_lift_rel * 100.0, 3),
            "triggered_traffic_share_pct": round(triggered_share_alpha * 100.0, 2),
            "site_wide_impact_pct": round(swi_rel * 100.0, 3),
        }
