# Эталонные симуляции

Независимые скрипты, по которым посчитаны таблицы ТЗ v4:

- `sim_restaurants.py` — ресторанный RCT, разделы 19.2 (A/A и MDE). Запуск: `python sim_restaurants.py optimistic|pessimistic`.
- `sim_kiosk.py` — сессионный A/B в киоске, раздел 7.4.

Это не тесты, pytest их не собирает. Генератор `ab_platform.simulation.vit` сверяется с цифрами из этих скриптов в `tests/simulation/test_vit_reference.py` (SYN-5).
