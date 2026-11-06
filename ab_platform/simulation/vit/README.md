# Синтетический генератор «как ВИТ»

Данные сети ресторанов с **известной правдой** для проверки методов платформы: FPR на A/A, мощность, смещение. Спецификация — ТЗ v4, раздел 11; дизайн — «Фаза 0 · Дизайн-заметка: синтетический генератор». Параметры по умолчанию — допущения порядка величины из публичных данных, **не данные сети**.

```python
from ab_platform.simulation.vit import NetworkConfig, generate_network, inject, save_network

net = generate_network(NetworkConfig.preset("pessimistic", n_restaurants=1000, seed=42))
net.restaurants      # справочник: формат, регион, город, координаты, торговая зона, флаги, даты
net.daily            # ресторан × день: чеки, выручка, скидки, себестоимость, маржа, время выдачи, списания
net.channels()       # разбивка по каналам (зал, киоск, авто, доставка, приложение)
net.events           # праздники и федеральные промо
guests = net.guests(200_000, pre_days=28, test_days=28)
sessions = net.kiosk_sessions("2025-03-03", days=14)

ids = net.restaurants.restaurant_id[:100].tolist()
net2, truth = inject(net, ids, "2025-04-07", "2025-04-20", lift=0.03, spillover=0.2)
truth.per_restaurant_mean_rel, truth.network_total_rel, truth.spillover_abs
save_network(net2, "data/synthetic/run1")   # Parquet + config.yaml + truth.json
```

| Пресет | Что это |
| --- | --- |
| `reference` | Эталонная модель `tests/simulation/reference/sim_restaurants.py`, оптимистичный шум; на ней воспроизводятся таблицы ТЗ 19.2 |
| `reference_pessimistic` | То же, пессимистичный шум |
| `optimistic`, `pessimistic` | Полный режим: форматы, сезонность, праздники, промо, шоки регионов, открытия, закрытия, ремонты |

Сценарии: `inject` — эффект, неоднородность по формату, новизна, последействие, ступенчатый старт, переток в торговой зоне; взаимодействие тестов — вызов `inject` на пересечении; `break_data` — пропуски, дубли, закрытия; `apply_guest_effect` и `adopt_feature` — эффект и самоотбор у гостей; `apply_kiosk_ui` — интерференция через очередь в киоске.

Тесты: `tests/simulation/` (`pytest` — быстрые; `pytest -m slow` — полные симуляции ТЗ).
