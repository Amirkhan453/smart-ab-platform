"""
Дерево выбора дизайна (ТЗ v4, 6.1): детерминированная функция, которую вызывает форма Паспорта (UI-0),
а в фазе 9 — ассистент (инструмент choose_design). Вход и выход — JSON.

Вопросы проходятся сверху вниз; первый ответ «да» задаёт метод. Ассистент не может выбрать метод вне результата.
"""
from __future__ import annotations

from typing import Any, Dict, List

# (ключ ответа, вопрос, метод, тип эксперимента, уровень доказательности, где делается)
TREE: List[Dict[str, str]] = [
    {"key": "randomize_guests", "question": "Можно ли случайно назначить гостей или сессии?",
     "method": "Гостевой или сессионный A/B (хэш)", "experiment_type": "classic_ab", "evidence_level": "A",
     "phase": "сейчас: эта форма"},
    {"key": "randomize_restaurants", "question": "Можно ли случайно назначить рестораны или зоны (≥ 6 пар)?",
     "method": "Кластерный RCT (пары и страты)", "experiment_type": "cluster_ab", "evidence_level": "A",
     "phase": "фаза 5 (рестораны)"},
    {"key": "switchable", "question": "Воздействие можно включать и выключать во времени без сильного последействия?",
     "method": "Switchback", "experiment_type": "switchback", "evidence_level": "A-",
     "phase": "после фазы 5"},
    {"key": "business_chosen_points",
     "question": "Тестовые точки выбраны бизнесом (1–30 точек), есть ≥ 8 недель истории и ≥ 19 чистых доноров?",
     "method": "SDID или ASCM; классический SC — для 1 точки или региона", "experiment_type": "quasi_experiment",
     "evidence_level": "B", "phase": "квазиэксперименты, только после калибровки CAU-0"},
    {"key": "staggered_rollout", "question": "Раскатка ступенчатая по многим ресторанам?",
     "method": "Staggered DiD (Callaway–Sant'Anna, Sun–Abraham), event study", "experiment_type": "quasi_experiment",
     "evidence_level": "B-", "phase": "квазиэксперименты, только после калибровки CAU-0"},
    {"key": "self_selected", "question": "Воздействие выбирает сам гость (начал пользоваться киоском или фичей)?",
     "method": "Matching, IPW или AIPW с диагностикой", "experiment_type": "quasi_experiment",
     "evidence_level": "C", "phase": "квазиэксперименты, только после калибровки CAU-0"},
    {"key": "whole_network", "question": "Изменение сразу на всю сеть, контроля нет?",
     "method": "ITS, CausalImpact или C-ARIMA", "experiment_type": "quasi_experiment", "evidence_level": "D",
     "phase": "квазиэксперименты, только после калибровки CAU-0"},
]
AUTO_SHIP_LEVELS = {"A", "A-"}


def choose_design(answers: Dict[str, Any]) -> Dict[str, Any]:
    """
    answers — {ключ вопроса: True/False}; неотвеченный вопрос считается «нет».
    Возвращает метод, уровень доказательности, правило дерева, которое сработало, и можно ли автоматический SHIP.
    """
    unknown = sorted(set(answers) - {n["key"] for n in TREE})
    if unknown:
        raise ValueError(f"Неизвестные вопросы дерева: {unknown}")
    for i, node in enumerate(TREE, start=1):
        if answers.get(node["key"]) is True:
            return {**node, "rule": f"6.1, вопрос {i}: {node['question']} — да",
                    "auto_ship_allowed": node["evidence_level"] in AUTO_SHIP_LEVELS}
    return {"key": None, "question": None, "method": None, "experiment_type": None, "evidence_level": None,
            "phase": None, "rule": "6.1: ни на один вопрос нет ответа «да» — причинный вывод невозможен",
            "auto_ship_allowed": False}
