"""Сохранение и загрузка сети: Parquet для таблиц, YAML для конфига, JSON для правды."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .config import NetworkConfig
from .network import Network
from .truth import Truth

_TABLES = ("restaurants", "events", "daily")


def save_network(net: Network, directory: str | Path) -> Path:
    out = Path(directory)
    out.mkdir(parents=True, exist_ok=True)
    for name in _TABLES:
        getattr(net, name).to_parquet(out / f"{name}.parquet", index=False)
    net.config.to_yaml(out / "config.yaml")
    (out / "truth.json").write_text(
        json.dumps([t.to_dict() for t in net.truths], ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    (out / "MANIFEST.json").write_text(
        json.dumps({"config_hash": net.config.config_hash(), "tables": list(_TABLES),
                    "rows": {n: len(getattr(net, n)) for n in _TABLES}}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return out


def _ns(df: pd.DataFrame) -> pd.DataFrame:
    """Parquet может вернуть даты в ms/us; приводим к ns, как у генератора."""
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            df[col] = df[col].astype("datetime64[ns]")
    return df


def load_network(directory: str | Path) -> Network:
    src = Path(directory)
    if not (src / "config.yaml").exists():
        raise FileNotFoundError(f"В {src} нет config.yaml — это не сохранённая сеть")
    cfg = NetworkConfig.from_yaml(src / "config.yaml")
    tables = {n: _ns(pd.read_parquet(src / f"{n}.parquet")) for n in _TABLES}
    truths = tuple(Truth.from_dict(d) for d in json.loads((src / "truth.json").read_text(encoding="utf-8")))
    return Network(config=cfg, truths=truths, **tables)
