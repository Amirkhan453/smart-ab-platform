"""
Детерминированные независимые потоки случайных чисел.

У каждого компонента генератора свой поток: ``stream(seed, "panel")``,
``stream(seed, "restaurants")`` и т. д. Добавление нового компонента не меняет
данные уже существующих (SYN-1), потому что поток зависит только от seed и имени.
"""
from __future__ import annotations

import zlib

import numpy as np


def _name_key(name: str) -> int:
    return zlib.crc32(name.encode("utf-8"))


def stream(seed: int, *names: str) -> np.random.Generator:
    """Генератор для пары (seed, путь имён), например ``stream(42, "guests", "window_3")``."""
    if not names:
        raise ValueError("Нужно хотя бы одно имя потока.")
    ss = np.random.SeedSequence(entropy=int(seed), spawn_key=tuple(_name_key(n) for n in names))
    return np.random.Generator(np.random.PCG64(ss))
