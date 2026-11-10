"""
Фаза 3, П3: контрактные тесты хранилища (LNK-3, PAS-1, AUD-1). Один набор для всех реализаций:
память, SQLite; реализация на PostgreSQL (фаза 6) добавляется в STORAGES и обязана пройти его же.
"""
import sqlite3
import threading

import pytest

from ab_platform.schemas.passport import ExperimentPassport, PassportStatus
from ab_platform.schemas.ruleset import Layer, Ruleset
from ab_platform.storage import AuditIntegrityError, MemoryStorage, SqliteStorage, StorageError
from tests.test_phase1_passport_v4 import base

V2 = [{"variant_id": "c", "traffic_percentage": 50.0, "is_control": True},
      {"variant_id": "t", "traffic_percentage": 50.0, "params": {"text": "новое"}}]
STORAGES = ["memory", "sqlite"]


@pytest.fixture(params=STORAGES)
def store(request, tmp_path):
    if request.param == "memory":
        yield MemoryStorage()
    else:
        s = SqliteStorage(str(tmp_path / "ab.db"))
        yield s
        s.close()


def passport(exp="exp1", **kw):
    return ExperimentPassport(**base(experiment_id=exp, owner="ivanova", **kw))


def ruleset(version, n=1):
    return Ruleset(version=version, layers=[Layer(layer_id="L")], experiments=[
        dict(experiment_id=f"e{i}", layer_id="L", slot_start=i * 100, slot_end=(i + 1) * 100, variants=V2,
             targeting={"platforms": ["ios"]}) for i in range(n)])


# ---------------- Паспорта ----------------
def test_passport_roundtrip_keeps_content_and_hash(store):
    p = passport(targeting={"platforms": ["ios"], "app_version_min": "5.2"}, targeting_audience_share=0.4,
                 variants=V2, ramp_plan=[5, 50, 100])
    store.save_passport(p)
    assert store.get_passport("exp1") == p and store.get_passport("exp1", 1) == p
    approved = p.with_status("review").with_status("approved", actor="petrov")
    store.save_passport(p.with_status("review"))
    store.save_passport(approved)
    back = store.get_passport("exp1")
    assert back == approved and back.passport_hash == back.compute_hash() and back.approved_by == "petrov"
    assert store.get_passport("нет") is None and store.get_passport("exp1", 2) is None


def test_draft_is_editable_approved_is_not(store):
    p = passport()
    store.save_passport(p)
    edited = p.model_copy(update={"mde_relative": 0.07})
    store.save_passport(edited)  # черновик правится на месте
    assert store.get_passport("exp1").mde_relative == 0.07
    store.save_passport(edited.with_status("review"))
    approved = edited.locked()
    store.save_passport(approved)
    forged = approved.model_copy(update={"mde_relative": 0.5})                      # содержимое изменено, хэш старый
    rehashed = forged.model_copy(update={"passport_hash": forged.compute_hash()})   # содержимое и хэш изменены
    unsealed = approved.model_copy(update={"passport_hash": None, "status": "draft"})
    for bad, text in ((forged, "хэш не совпадает"), (rehashed, "неизменяем"), (unsealed, "неизменяем")):
        with pytest.raises(StorageError, match=text):
            store.save_passport(bad)
    assert store.get_passport("exp1") == approved
    running = approved.with_status("running")
    store.save_passport(running)  # статус утверждённой версии меняется
    assert store.get_passport("exp1").status == PassportStatus.RUNNING
    with pytest.raises(StorageError, match="недопустим"):
        store.save_passport(running.model_copy(update={"status": "approved"}))
    with pytest.raises(StorageError, match="недопустим"):
        store.save_passport(running.model_copy(update={"status": "archived"}))


def test_versions_go_in_order_and_listing_returns_latest(store):
    p1 = passport().locked()
    with pytest.raises(StorageError, match="ожидается версия 1"):
        store.save_passport(p1.new_version(mde_relative=0.05))
    store.save_passport(passport())
    store.save_passport(passport().with_status("review"))
    store.save_passport(p1)
    v2 = p1.new_version(mde_relative=0.05)
    with pytest.raises(StorageError, match="ожидается версия 2"):
        store.save_passport(v2.model_copy(update={"version": 3}))
    store.save_passport(v2)
    store.save_passport(passport("exp0"))
    assert store.get_passport("exp1").version == 2 and store.get_passport("exp1", 1) == p1
    assert [(p.experiment_id, p.version) for p in store.list_passports()] == [("exp0", 1), ("exp1", 2)]


# ---------------- ruleset ----------------
def test_ruleset_versions_are_append_only_and_sequential(store):
    assert store.latest_ruleset() is None and store.get_ruleset(1) is None and store.latest_ruleset_version() is None
    r5 = ruleset(5, n=3)  # первая версия может быть любой, дальше — строго +1
    store.save_ruleset(r5)
    for bad in (ruleset(5), ruleset(4), ruleset(7)):
        with pytest.raises(StorageError, match="ожидается версия 6"):
            store.save_ruleset(bad)
    store.save_ruleset(ruleset(6))
    latest = store.latest_ruleset()
    assert store.latest_ruleset_version() == 6
    assert latest.version == 6 and store.get_ruleset(5) == r5 and store.get_ruleset(5).etag == r5.etag
    assert store.get_ruleset(5).experiments[2].targeting.platforms == ["ios"]


# ---------------- аудит ----------------
def test_audit_chain_and_filter(store):
    a = store.append_audit("ivanova", "passport.create", "exp1", {"version": 1})
    b = store.append_audit("petrov", "passport.approve", "exp1", {"hash": "ab", "вложено": {"x": [1, 2]}})
    c = store.append_audit("admin", "auth.denied")
    assert (a.seq, b.seq, c.seq) == (1, 2, 3) and b.prev_hash == a.hash and c.prev_hash == b.hash
    assert a.prev_hash == "0" * 64 and len({a.hash, b.hash, c.hash}) == 3
    assert store.audit() == [a, b, c] and store.audit("exp1") == [a, b] and store.audit("нет") == []
    assert store.audit()[1].details == {"hash": "ab", "вложено": {"x": [1, 2]}} and c.details == {}
    assert store.verify_audit() == 3


def test_transaction_is_all_or_nothing(store):
    store.save_ruleset(ruleset(1))
    with pytest.raises(RuntimeError, match="сбой"):
        with store.transaction():
            store.save_passport(passport())
            store.save_ruleset(ruleset(2))
            store.append_audit("ivanova", "passport.create", "exp1")
            raise RuntimeError("сбой посреди команды")
    assert store.get_passport("exp1") is None and store.latest_ruleset().version == 1 and store.audit() == []
    with store.transaction():
        store.save_passport(passport())
        store.save_ruleset(ruleset(2))
        store.append_audit("ivanova", "passport.create", "exp1")
    assert store.get_passport("exp1") is not None and store.latest_ruleset().version == 2 and store.verify_audit() == 1
    with pytest.raises(StorageError, match="Вложенные"):
        with store.transaction():
            with store.transaction():
                pass
    # отклонённая запись внутри транзакции откатывает всё, что было до неё
    with pytest.raises(StorageError):
        with store.transaction():
            store.append_audit("x", "y")
            store.save_ruleset(ruleset(9))
    assert store.verify_audit() == 1


def test_concurrent_audit_appends_keep_chain(store):
    def work(i):
        for j in range(25):
            store.append_audit(f"user{i}", "act", "exp1", {"j": j})
    threads = [threading.Thread(target=work, args=(i,)) for i in range(8)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert store.verify_audit() == 200 and [r.seq for r in store.audit()] == list(range(1, 201))


# ---------------- только SQLite: файл и перезапуск ----------------
def test_sqlite_survives_restart(tmp_path):
    path = str(tmp_path / "ab.db")
    s = SqliteStorage(path)
    approved = passport(variants=V2).locked()
    s.save_passport(passport(variants=V2))
    s.save_passport(passport(variants=V2).with_status("review"))
    s.save_passport(approved)
    s.save_ruleset(ruleset(1))
    s.save_ruleset(ruleset(2, n=2))
    s.append_audit("ivanova", "passport.create", "exp1")
    s.append_audit("petrov", "passport.approve", "exp1")
    s.close()
    again = SqliteStorage(path)
    assert again.get_passport("exp1") == approved and again.latest_ruleset() == ruleset(2, n=2)
    assert again.get_ruleset(1).etag == ruleset(1).etag and again.verify_audit() == 2
    again.append_audit("admin", "after.restart")
    assert again.verify_audit() == 3 and again.audit()[-1].prev_hash == again.audit()[-2].hash
    again.close()


@pytest.mark.parametrize("sql, text", [
    ("UPDATE audit SET actor = 'mallory' WHERE seq = 2", "№2 изменена"),
    ("UPDATE audit SET details = '{\"version\": 9}' WHERE seq = 1", "№1 изменена"),
    ("DELETE FROM audit WHERE seq = 2", "удалена или вставлена"),
    ("DELETE FROM audit WHERE seq = 1", "удалена или вставлена"),
    ("UPDATE audit SET seq = 7 WHERE seq = 3", "удалена или вставлена"),
])
def test_sqlite_audit_tampering_is_detected(tmp_path, sql, text):
    path = str(tmp_path / "ab.db")
    s = SqliteStorage(path)
    for i in range(3):
        s.append_audit("ivanova", "act", "exp1", {"version": i})
    s.close()
    raw = sqlite3.connect(path)
    raw.execute(sql)
    raw.commit()
    raw.close()
    s = SqliteStorage(path)
    with pytest.raises(AuditIntegrityError, match=text):
        s.verify_audit()
    s.close()


def test_sqlite_two_connections_do_not_overwrite_each_other(tmp_path):
    """Два процесса с одним файлом: второй видит версию первого и не может записать её повторно."""
    path = str(tmp_path / "ab.db")
    a, b = SqliteStorage(path), SqliteStorage(path)
    a.save_ruleset(ruleset(1))
    with pytest.raises(StorageError, match="ожидается версия 2"):
        b.save_ruleset(ruleset(1))
    b.save_ruleset(ruleset(2))
    a.append_audit("a", "x")
    b.append_audit("b", "y")
    assert a.latest_ruleset().version == 2 and a.verify_audit() == 2 and b.audit()[1].prev_hash == a.audit()[0].hash
    a.close(), b.close()


def test_sqlite_rejects_unknown_schema_version(tmp_path):
    path = str(tmp_path / "ab.db")
    SqliteStorage(path).close()
    raw = sqlite3.connect(path)
    raw.execute("UPDATE meta SET value = '99' WHERE key = 'schema_version'")
    raw.commit()
    raw.close()
    with pytest.raises(StorageError, match="нужна миграция"):
        SqliteStorage(path)


def test_sqlite_concurrent_writers_from_two_connections(tmp_path):
    """Два соединения пишут одновременно: запись ждёт своей очереди, цепочка аудита остаётся целой."""
    path = str(tmp_path / "ab.db")
    stores, errors = [SqliteStorage(path), SqliteStorage(path)], []

    def work(s, name):
        try:
            for j in range(60):
                with s.transaction():
                    s.append_audit(name, "act", "exp1", {"j": j})
        except Exception as exc:  # noqa: BLE001 — ошибку потока надо донести до assert
            errors.append(exc)
    threads = [threading.Thread(target=work, args=(s, f"w{i}")) for i, s in enumerate(stores)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert errors == [] and stores[0].verify_audit() == 120
    [s.close() for s in stores]
