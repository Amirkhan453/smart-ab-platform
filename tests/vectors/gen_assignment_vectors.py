"""
Генератор общих тест-векторов назначения для всех SDK (Python, JS, далее — мобильные).

Эталон не зависит от проверяемого кода:
- хэши считает C-референс tests/vectors/mm_reference.c (компилируется cc);
- правила ТЗ 4.1 записаны здесь заново, движок ab_platform.engine не импортируется.
  Ruleset сериализуется моделью Ruleset — ровно в том виде, в каком его отдаёт сервер.

Округление — как round() в Python: половина к чётному (round(12.5) = 12). Векторы содержат гостей
ровно на таких границах, иначе SDK с Math.round разошёлся бы незаметно.

Таргетинг (ТЗ v4.1, TGT-4) — третий ruleset: условия, тип единицы, параметры вариантов, «первый показ уже был».
Правила сравнения записаны в поле targeting_rules и в функции tgt_reason ниже.

Запуск: .venv/bin/python tests/vectors/gen_assignment_vectors.py  (перезаписывает assignment_vectors.json)
"""
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from ab_platform.schemas.ruleset import Ruleset  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE / "assignment_vectors.json"
N = 10000


def c_hasher():
    """Функция пакетного хэширования через C-референс."""
    tmp = tempfile.mkdtemp()
    exe = os.path.join(tmp, "mm_reference")
    subprocess.run(["cc", "-O2", "-o", exe, str(HERE / "mm_reference.c")], check=True)

    def hash_many(keys):
        keys = list(keys)
        assert all("\n" not in k for k in keys)
        out = subprocess.run([exe], input="".join(k + "\n" for k in keys).encode("utf-8"),
                             capture_output=True, check=True)
        assert out.stderr.decode().strip() == "verif=B0F57EE3", out.stderr
        vals = [int(x) for x in out.stdout.decode().split()]
        assert len(vals) == len(keys)
        return dict(zip(keys, vals))

    return hash_many


# --- правила ТЗ 4.1, записанные независимо от движка --------------------------------------------

def rnd(x):
    return int(round(x))  # половина к чётному


def pct_to_buckets(pct):
    return rnd(pct * 100)


def ramp_end(s, e, ramp_pct):
    return s + rnd(ramp_pct / 100.0 * (e - s))


def allocate(percentages):
    raw = [p * 100 for p in percentages]
    counts = [int(r) for r in raw]
    order = sorted(range(len(raw)), key=lambda i: raw[i] - counts[i], reverse=True)  # устойчивая сортировка
    for i in order[: N - sum(counts)]:
        counts[i] += 1
    return counts


DIGITS = "0123456789"


def ver(x):
    """Версия → 6 чисел (недостающие части — нули) или None."""
    if type(x) is not str:
        return None
    parts = x.split(".")
    if not 1 <= len(parts) <= 6 or any(not 1 <= len(p) <= 9 or any(ch not in DIGITS for ch in p) for p in parts):
        return None
    return [int(p) for p in parts] + [0] * (6 - len(parts))


def low(text):
    return "".join(chr(ord(ch) + 32) if "A" <= ch <= "Z" else ch for ch in text)


def tgt_reason(t, attrs):
    """Причина, по которой гость не подходит под условия t, или None. Порядок проверки фиксирован."""
    attrs = attrs or {}
    for name, allowed in (("platform", t["platforms"]), ("app_version", None), ("city", t["cities"]),
                          ("region", t["regions"]), ("is_new_guest", None), ("has_delivery", None)):
        x = attrs.get(name)
        if name == "app_version":
            lo, hi = t["app_version_min"], t["app_version_max"]
            if lo is None and hi is None:
                continue
            if ver(x) is None:
                return "targeting_missing:app_version"
            if (lo is not None and ver(x) < ver(lo)) or (hi is not None and ver(x) > ver(hi)):
                return "targeting:app_version"
        elif name in ("is_new_guest", "has_delivery"):
            if t[name] is None:
                continue
            if x is not True and x is not False:
                return f"targeting_missing:{name}"
            if x is not t[name]:
                return f"targeting:{name}"
        else:
            if allowed is None:
                continue
            if type(x) is not str:
                return f"targeting_missing:{name}"
            if low(x) not in allowed:
                return f"targeting:{name}"
    return None


def expected(rs, rule, uid, H, attrs=None, unit_type=None, enrolled=False):
    layer = next(l for l in rs["layers"] if l["layer_id"] == rule["layer_id"])
    control = next(v["variant_id"] for v in rule["variants"] if v["is_control"])
    if rule["kill_switch"]:
        return control, False, "kill_switch"
    if unit_type is not None and unit_type != rule["unit_type"]:
        return control, False, "unit_type_mismatch"
    hp = rs["holdout"]["pct"] if layer["marketing"] else 0.0
    if hp > 0 and H[f"{rs['holdout']['salt']}:{uid}"] % N < pct_to_buckets(hp):
        return control, False, "holdout"
    zl = H[f"{layer['salt']}:{uid}"] % N
    if not rule["slot_start"] <= zl < rule["slot_end"]:
        return control, False, "not_in_lane"
    if not enrolled and tgt_reason(rule["targeting"], attrs) is not None:
        return control, False, tgt_reason(rule["targeting"], attrs)
    if zl >= ramp_end(rule["slot_start"], rule["slot_end"], rule["ramp_pct"]):
        return control, False, "ramp"
    zv = H[f"{rule['salt']}:{uid}"] % N
    cum = 0
    for v, n in zip(rule["variants"], allocate([v["traffic_percentage"] for v in rule["variants"]])):
        cum += n
        if zv < cum:
            return v["variant_id"], True, "assigned"
    return rule["variants"][-1]["variant_id"], True, "assigned"


# --- наборы -------------------------------------------------------------------------------------

def var(spec, control):
    return [{"variant_id": v, "traffic_percentage": p, "is_control": v == control} for v, p in spec]


def rulesets():
    rs1 = Ruleset(
        version=7,
        holdout={"salt": "global_holdout_v1", "pct": 0.125},  # 12,5 бакета → 12 (половина к чётному)
        layers=[{"layer_id": "ui"}, {"layer_id": "algo", "salt": "algo_v2"},
                {"layer_id": "ops", "salt": "ops_layer", "marketing": False}],
        experiments=[
            {"experiment_id": "exp_menu", "layer_id": "ui", "slot_start": 0, "slot_end": 5000,
             "variants": var([("control", 50.0), ("treatment", 50.0)], "control")},
            # ramp 50% от ширины 1001 = 500,5 → 500: активны бакеты [5000, 5500)
            {"experiment_id": "exp_promo", "layer_id": "ui", "slot_start": 5000, "slot_end": 6001, "ramp_pct": 50.0,
             "variants": var([("A", 33.33), ("B", 33.33), ("C", 33.34)], "B")},
            {"experiment_id": "exp_rank", "layer_id": "algo", "slot_start": 0, "slot_end": 8000, "ramp_pct": 12.5,
             "variants": var([("v1", 10.0), ("v2", 20.0), ("v3", 70.0)], "v2")},
            {"experiment_id": "exp_killed", "layer_id": "algo", "slot_start": 8000, "slot_end": 10000,
             "kill_switch": True, "variants": var([("off", 50.0), ("on", 50.0)], "off")},
            {"experiment_id": "exp_ops", "flag_key": "ops_flag", "salt": "ops_salt_x", "layer_id": "ops",
             "variants": var([("c", 100 / 3), ("t1", 100 / 3), ("t2", 100 / 3)], "c")},
        ],
    )
    rs2 = Ruleset(
        version=1,
        holdout={"pct": 0.0},
        layers=[{"layer_id": "main"}],
        experiments=[
            {"experiment_id": "exp_zero", "layer_id": "main", "slot_start": 0, "slot_end": 9000, "ramp_pct": 0.0,
             "variants": var([("off", 50.0), ("on", 50.0)], "off")},
            {"experiment_id": "exp_tail", "layer_id": "main", "slot_start": 9000, "slot_end": 10000, "ramp_pct": 100.0,
             "variants": var([("x", 0.005), ("y", 99.995)], "y")},
        ],
    )
    # таргетинг, тип единицы, параметры вариантов (ТЗ v4.1)
    rs3 = Ruleset(
        version=3,
        layers=[{"layer_id": "app"}, {"layer_id": "pre"}],
        experiments=[
            {"experiment_id": "tg_full", "layer_id": "app", "slot_start": 0, "slot_end": 4000, "ramp_pct": 50.0,
             "targeting": {"platforms": ["ios", "android"], "app_version_min": "5.2", "app_version_max": "5.10",
                           "cities": ["msk", "spb", "kzn"], "regions": ["ru-mow", "ru-spe", "ru-ta"],
                           "is_new_guest": False, "has_delivery": True},
             "variants": [
                 {"variant_id": "control", "traffic_percentage": 50.0, "is_control": True, "params": {"title": "Меню"}},
                 {"variant_id": "treatment", "traffic_percentage": 50.0,
                  "params": {"title": "Новое меню", "order": [3, 1, 2], "discount": 0.1, "flags": {"big": True}}}]},
            {"experiment_id": "tg_ver_min", "layer_id": "app", "slot_start": 4000, "slot_end": 7000,
             "targeting": {"app_version_min": "5.10"},
             "variants": var([("control", 50.0), ("treatment", 50.0)], "control")},
            {"experiment_id": "tg_ver_max", "layer_id": "app", "slot_start": 7000, "slot_end": 10000,
             "targeting": {"app_version_max": "5.9.0", "is_new_guest": True},
             "variants": var([("control", 50.0), ("treatment", 50.0)], "control")},
            {"experiment_id": "dev_pre", "layer_id": "pre", "unit_type": "device_id",
             "targeting": {"platforms": ["web"], "has_delivery": False},
             "variants": [
                 {"variant_id": "old", "traffic_percentage": 50.0, "is_control": True},
                 {"variant_id": "new", "traffic_percentage": 50.0, "params": {"banner": "Войдите — получите кофе"}}]},
        ],
    )
    return [rs1, rs2, rs3]


OK_FULL = {"platform": "ios", "app_version": "5.2", "city": "msk", "region": "ru-mow", "is_new_guest": False,
           "has_delivery": True}


def targeting_scenarios():
    """Наборы признаков по флагам третьего ruleset: (признаки, тип единицы, был ли первый показ)."""
    def full(**change):
        return {**OK_FULL, **change}

    def without(name):
        return {k: v for k, v in OK_FULL.items() if k != name}

    tg_full = [
        OK_FULL, full(platform="IOS", city="MSK", region="RU-MOW"), full(platform="Android", city="kzn", region="ru-ta"),
        full(platform="web"), full(platform="ios "), full(platform=5), without("platform"),
        # str.lower() превратил бы знак Кельвина в «k», а «İ» — в «i» с точкой: совпадать не должно
        full(city="\u212aZN"), full(platform="İOS"), full(city="spb", region="ru-mow"), full(city="ekb"), full(city=None),
        full(region="ru-kda"), without("region"),
        full(app_version="5.1.9"), full(app_version="5.2.0"), full(app_version="5.9"), full(app_version="5.10"),
        full(app_version="5.10.0.0"), full(app_version="5.10.1"), full(app_version="5.11"), full(app_version="05.03"),
        full(app_version="6"), full(app_version="5.2-beta"), full(app_version="5.3\n"), full(app_version="٥.٣"),
        full(app_version=5.3), full(app_version="1.2.3.4.5.6.7"), full(app_version=""), without("app_version"),
        full(is_new_guest=True), full(is_new_guest=0), full(is_new_guest="false"), without("is_new_guest"),
        full(has_delivery=False), full(has_delivery=1), full(has_delivery=None),
        full(city="ekb", has_delivery=False), {}, None,
        # два невыполненных условия подряд: причина — по первому в порядке проверки
        full(platform="web", app_version="1.0"), full(app_version="1.0", city="ekb"), full(city="ekb", region="ru-kda"),
        full(region="ru-kda", is_new_guest=True), full(is_new_guest=True, has_delivery=False),
    ]
    out = {"tg_full": [(a, None, False) for a in tg_full] + [(full(city="ekb"), None, True), ({}, None, True),
                                                             (OK_FULL, "guest_id", False), (OK_FULL, "device_id", False)]}
    out["tg_ver_min"] = [({"app_version": v}, None, False)
                         for v in ("5.9", "5.9.99", "5.10", "5.10.0", "5.100", "6", "10.0", "5.1")] + [
        ({"app_version": "5.9"}, None, True), ({"platform": "ios"}, None, False), (None, None, False)]
    out["tg_ver_max"] = [({"app_version": v, "is_new_guest": True}, None, False)
                         for v in ("5.9", "5.9.0.0", "5.9.1", "5.10", "4", "0", "5.09")] + [
        ({"app_version": "5.9", "is_new_guest": False}, None, False), ({"app_version": "5.10"}, None, False),
        ({"app_version": "5.10", "is_new_guest": False}, None, True), ({"is_new_guest": True}, None, False)]
    web = {"platform": "web", "has_delivery": False}
    out["dev_pre"] = [(web, "device_id", False), (web, None, False), (web, "guest_id", False),
                      ({"platform": "ios", "has_delivery": False}, "device_id", False),
                      ({"platform": "ios", "has_delivery": False}, "guest_id", True),
                      ({"platform": "ios", "has_delivery": False}, "device_id", True), ({}, "device_id", False)]
    return out


def main(out: Path = OUT):
    H_many = c_hasher()
    rss = rulesets()
    dumps = [json.loads(r.to_json()) for r in rss]

    # гости ровно на границах округления (ищем перебором по C-хэшу)
    probes = [f"probe_{i}" for i in range(300_000)]
    hp = H_many([f"global_holdout_v1:{p}" for p in probes] + [f"layer:ui:{p}" for p in probes]
                + [f"exp_menu:{p}" for p in probes])

    def menu_guest(zv):
        # участник exp_menu (дорожка [0, 5000), вне holdout) с бакетом варианта ровно zv
        return next(p for p in probes if hp[f"exp_menu:{p}"] % N == zv and hp[f"layer:ui:{p}"] % N < 5000
                    and hp[f"global_holdout_v1:{p}"] % N >= 12)

    split_lo, split_hi = menu_guest(4999), menu_guest(5000)  # последний бакет control и первый treatment
    tie_holdout = next(p for p in probes if hp[f"global_holdout_v1:{p}"] % N == 12)
    tie_ramp = next(p for p in probes if hp[f"layer:ui:{p}"] % N == 5500 and hp[f"global_holdout_v1:{p}"] % N >= 12)
    hold_in = next(p for p in probes if hp[f"global_holdout_v1:{p}"] % N == 11)
    edge_ramp_in = next(p for p in probes if hp[f"layer:ui:{p}"] % N == 5499 and hp[f"global_holdout_v1:{p}"] % N >= 12)

    units = [f"guest_{i}" for i in range(250)] + [
        "гость_42", "😀", "東京タワー", "a:b:c", " пробел ", "x" * 300, "0", "Ω≈ç√", hold_in, tie_holdout, tie_ramp, edge_ramp_in, split_lo, split_hi]

    keys = set()
    for rs in dumps:
        for u in units if rs is not dumps[2] else [f"guest_{i}" for i in range(40)] + ["гость_42", "😀"]:
            keys.add(f"{rs['holdout']['salt']}:{u}")
            keys.update(f"{l['salt']}:{u}" for l in rs["layers"])
            keys.update(f"{e['salt']}:{u}" for e in rs["experiments"])
    hash_keys = ["", "a", "ab", "abc", "abcd", "abcde", "Привет", "😀", "global_holdout_v1:guest_1",
                 "layer:ui:гость_42", "exp_menu:" + "x" * 300]
    H = H_many(sorted(keys | set(hash_keys)))

    def params_of(rule, variant_id):
        return next(v["params"] for v in rule["variants"] if v["variant_id"] == variant_id)

    cases = []
    for i, rs in enumerate(dumps[:2]):
        for rule in rs["experiments"]:
            for u in units:
                v, inexp, reason = expected(rs, rule, u, H)
                cases.append({"rs": i, "flag": rule["flag_key"], "unit": u, "variant": v, "in": inexp,
                              "reason": reason})
    # третий ruleset: каждый гость × каждый набор признаков своего флага
    scenarios = targeting_scenarios()
    t_units = [f"guest_{i}" for i in range(40)] + ["гость_42", "😀"]
    for rule in dumps[2]["experiments"]:
        for attrs, unit_type, enrolled in scenarios[rule["flag_key"]]:
            for u in t_units:
                v, inexp, reason = expected(dumps[2], rule, u, H, attrs, unit_type, enrolled)
                case = {"rs": 2, "flag": rule["flag_key"], "unit": u, "attrs": attrs, "variant": v, "in": inexp,
                        "reason": reason, "params": params_of(rule, v)}
                if unit_type is not None:
                    case["unit_type"] = unit_type
                if enrolled:
                    case["enrolled"] = True
                cases.append(case)
    # обязательные граничные исходы — иначе векторы ничего не доказывают
    by = {(c["rs"], c["flag"], c["unit"]): c for c in cases}
    assert by[(0, "exp_menu", tie_holdout)]["reason"] != "holdout"  # 12 бакетов, бакет 12 — вне holdout
    assert by[(0, "exp_menu", hold_in)]["reason"] == "holdout" and by[(0, "ops_flag", hold_in)]["in"]
    assert by[(0, "exp_promo", tie_ramp)]["reason"] == "ramp"
    assert by[(0, "exp_promo", edge_ramp_in)]["in"]
    assert by[(0, "exp_menu", split_lo)]["variant"] == "control" and by[(0, "exp_menu", split_hi)]["variant"] == "treatment"
    reasons = {c["reason"] for c in cases}
    assert reasons == {"assigned", "holdout", "not_in_lane", "ramp", "kill_switch", "unit_type_mismatch"} | {
        f"{kind}:{a}" for kind in ("targeting", "targeting_missing")
        for a in ("platform", "app_version", "city", "region", "is_new_guest", "has_delivery")}, reasons
    t_cases = [c for c in cases if c["rs"] == 2]

    def some(flag, reason, **kw):
        return any(c["flag"] == flag and c["reason"] == reason and all(c.get(k) == v for k, v in kw.items())
                   for c in t_cases)

    # порядок проверок: holdout и дорожка — раньше таргетинга, таргетинг — раньше ramp-up
    bad = {**OK_FULL, "city": "ekb"}
    assert some("tg_full", "holdout", attrs=bad) and some("tg_full", "not_in_lane", attrs=bad)
    assert some("tg_full", "targeting:city", attrs=bad) and some("tg_full", "ramp", attrs=OK_FULL)
    ramp_out = {c["unit"] for c in t_cases if c["flag"] == "tg_full" and c["reason"] == "ramp" and c["attrs"] == OK_FULL}
    assert all(c["reason"] == "targeting:city" for c in t_cases
               if c["flag"] == "tg_full" and c["unit"] in ramp_out and c["attrs"] == bad and not c.get("enrolled"))
    for first, second, change in (("platform", "app_version", dict(platform="web", app_version="1.0")),
                                  ("app_version", "city", dict(app_version="1.0", city="ekb")),
                                  ("city", "region", dict(city="ekb", region="ru-kda")),
                                  ("region", "is_new_guest", dict(region="ru-kda", is_new_guest=True)),
                                  ("is_new_guest", "has_delivery", dict(is_new_guest=True, has_delivery=False))):
        assert some("tg_full", f"targeting:{first}", attrs={**OK_FULL, **change}), (first, second)
    # «первый показ уже был» снимает только таргетинг; оба варианта и параметры встречаются
    assert some("tg_full", "assigned", attrs=bad, enrolled=True) and some("tg_full", "ramp", attrs=bad, enrolled=True)
    assert some("dev_pre", "unit_type_mismatch", enrolled=True) and some("dev_pre", "assigned", enrolled=True)
    assert {c["variant"] for c in t_cases if c["flag"] == "tg_full" and c["in"]} == {"control", "treatment"}
    assert any(c["params"].get("discount") == 0.1 for c in t_cases) and any(c["params"] == {} for c in t_cases)
    # числовое, а не строковое сравнение версий
    assert some("tg_ver_min", "targeting:app_version", attrs={"app_version": "5.9"})
    assert some("tg_ver_min", "assigned", attrs={"app_version": "5.100"})
    assert some("tg_ver_max", "targeting:app_version", attrs={"app_version": "5.10", "is_new_guest": True})
    assert some("tg_ver_max", "assigned", attrs={"app_version": "5.09", "is_new_guest": True})
    assert {c["variant"] for c in cases if c["flag"] == "exp_tail" and c["in"]} == {"y"}  # 0,005% → 0 или 1 бакет

    data = {
        "description": "Общие векторы назначения гостевого домена (ТЗ v4 4.1) для всех SDK. "
                       "Сгенерированы tests/vectors/gen_assignment_vectors.py.",
        "source": "Хэши — C-референс mm_reference.c; назначения — независимая запись правил ТЗ в генераторе.",
        "rounding": "Как round() в Python: половина к чётному (12.5 → 12, 500.5 → 500).",
        "order": "kill_switch → тип единицы → holdout → слой и дорожка → таргетинг → ramp-up → вариант.",
        "targeting_rules": [
            "Все заданные условия должны выполняться; причина — по первому невыполненному в порядке: platform, "
            "app_version, city, region, is_new_guest, has_delivery.",
            "Нет признака или он не того типа → targeting_missing:<признак>; значение не подошло → targeting:<признак>.",
            "platform, city, region: строка; перед сравнением только A–Z переводятся в a–z; пробелы не обрезаются.",
            "app_version: от 1 до 6 частей через точку, в каждой от 1 до 9 цифр 0–9; сравнение по числам слева "
            "направо, недостающие части — нули (5.1 = 5.1.0, 5.10 > 5.9); границы включительные.",
            "is_new_guest, has_delivery: только true или false; 0, 1 и строки — как отсутствие признака.",
            "case.unit_type задан и не равен unit_type правила → unit_type_mismatch (проверяется до holdout).",
            "case.enrolled = true (первый показ уже был) → таргетинг не проверяется; остальные шаги — как обычно.",
            "case.params — параметры выданного варианта; если поля нет, параметры пустые.",
        ],
        "smhasher_verification": "B0F57EE3",
        "n_buckets": N,
        "hash": [{"key": k, "uint32": H[k], "bucket": H[k] % N} for k in hash_keys],
        "pct_to_buckets": [{"pct": p, "buckets": pct_to_buckets(p)}
                           for p in (0.0, 0.005, 0.015, 0.125, 0.375, 2.5, 5.0, 12.5, 33.33, 99.995)],
        "ramp_end": [{"slot_start": s, "slot_end": e, "ramp_pct": r, "end": ramp_end(s, e, r)}
                     for s, e, r in ((5000, 6001, 50.0), (0, 5, 50.0), (0, 3, 50.0), (0, 10000, 12.5),
                                     (100, 103, 50.0), (0, 8000, 12.5), (0, 10000, 0.0), (7, 10000, 100.0))],
        "allocate": [{"percentages": ps, "buckets": allocate(ps)}
                     for ps in ([50.0, 50.0], [33.33, 33.33, 33.34], [100 / 3] * 3, [10.0, 20.0, 70.0],
                                [0.005, 99.995], [12.345, 87.655],
                                # разные дробные остатки: проверяют порядок «наибольший остаток первым»
                                [10.006, 20.007, 69.987], [25.003, 25.002, 25.001, 24.994])],
        "rulesets": dumps,
        "cases": "__CASES__",
    }
    # один случай — одна строка: файл читается и диффается глазами
    body = "[\n" + ",\n".join("  " + json.dumps(c, ensure_ascii=False) for c in cases) + "\n ]"
    text = json.dumps(data, ensure_ascii=False, indent=1).replace('"__CASES__"', body)
    assert json.loads(text)["cases"] == cases
    out.write_text(text + "\n", encoding="utf-8")
    print(f"{out.name}: {len(cases)} случаев, причины {sorted(reasons)}; границы: {tie_holdout}, {tie_ramp}")


if __name__ == "__main__":
    main()
