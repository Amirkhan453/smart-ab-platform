// Тесты JS SDK: общие векторы (совпадение с Python 100%) и поведение клиента. Запуск: node --test sdk/js/test/
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";

import {
  ExperimentClient, FALLBACK_INVALID_UNIT, FALLBACK_NO_RULESET, FALLBACK_STALE, FALLBACK_UNKNOWN_FLAG,
  SMHASHER_VERIFICATION_X86_32, allocateVariantBuckets, bucket, evaluate, murmurhash3_x86_32, parseRuleset,
  asciiLower, compareVersions, knownAttributes, parseVersion, pctToBuckets, rampEnd, roundHalfEven, smhasherSelfTest,
} from "../ab-sdk.mjs";

const V = JSON.parse(readFileSync(new URL("../../../tests/vectors/assignment_vectors.json", import.meta.url), "utf8"));
const MM = JSON.parse(readFileSync(new URL("../../../tests/vectors/murmur3_x86_32.json", import.meta.url), "utf8"));

// --- векторы ------------------------------------------------------------------------------------

test("smhasher и хэши C-референса", () => {
  const hex = (x) => x.toString(16).toUpperCase().padStart(8, "0");
  assert.equal(hex(smhasherSelfTest()), V.smhasher_verification);
  assert.equal(hex(SMHASHER_VERIFICATION_X86_32), V.smhasher_verification);
  for (const v of [...V.hash, ...MM.vectors]) {
    assert.equal(murmurhash3_x86_32(v.key), v.uint32, v.key);
    assert.equal(v.uint32 % 10000, v.bucket);
  }
});

test("округление и нарезка как в Python (половина к чётному)", () => {
  for (const c of V.pct_to_buckets) assert.equal(pctToBuckets(c.pct), c.buckets, `pct ${c.pct}`);
  for (const c of V.ramp_end) assert.equal(rampEnd(c.slot_start, c.slot_end, c.ramp_pct), c.end, JSON.stringify(c));
  for (const c of V.allocate) assert.deepEqual(allocateVariantBuckets(c.percentages), c.buckets, String(c.percentages));
  assert.deepEqual([0.5, 1.5, 2.5, 12.5, 500.5, 2.4999, 2.5001].map(roundHalfEven), [0, 2, 2, 12, 500, 2, 3]);
});

test("все назначения векторов совпадают (100%)", () => {
  const rss = V.rulesets.map((r) => parseRuleset(r));
  let n = 0;
  const clients = rss.map((rs) => new ExperimentClient({ initialRuleset: rs.raw }));
  const reasons = new Set();
  for (const c of V.cases) {
    const want = [c.variant, c.in, c.reason, c.params ?? {}];
    const opts = { attributes: c.attrs ?? null, unitType: c.unit_type ?? null };
    const r = evaluate(rss[c.rs], c.flag, c.unit, { ...opts, enrolled: c.enrolled ?? false });
    assert.deepEqual([r.variantId, r.inExperiment, r.reason, r.params], want, JSON.stringify(c));
    // клиент: «первый показ уже был» — его память о показах
    const client = clients[c.rs];
    const key = JSON.stringify([c.unit, r.experimentId]);
    if (c.enrolled) client._enrolled.set(key, true);
    const d = client.getVariant(c.flag, c.unit, { ...opts, logExposure: false });
    client._enrolled.delete(key);
    assert.deepEqual([d.variantId, d.inExperiment, d.reason, d.params], want, JSON.stringify(c));
    reasons.add(c.reason);
    n++;
  }
  assert.equal(n, V.cases.length);
  assert.ok(n > 4000);
  assert.equal([...reasons].filter((r) => r.startsWith("targeting")).length, 12);
  assert.ok(reasons.has("unit_type_mismatch"));
  assert.ok(clients.every((c) => c.pendingExposures().length === 0)); // logExposure: false показов не создаёт
});

test("bucket: некорректный ввод", () => {
  assert.throws(() => bucket("s", ""), TypeError);
  assert.throws(() => bucket("", "u"), TypeError);
  assert.throws(() => bucket("s", 42), TypeError);
  assert.throws(() => bucket("s", "\ud800x"), TypeError); // одиночный суррогат: Python тоже не кодирует
});

// --- клиент -------------------------------------------------------------------------------------

function variants(...ids) {
  return ids.map((v, i) => ({ variant_id: v, traffic_percentage: 100 / ids.length, is_control: i === 0 }));
}

function makeRuleset({ nLayers = 1, version = 1, ramp = 100, kill = false } = {}) {
  const layers = [];
  const experiments = [];
  for (let i = 0; i < nLayers; i++) {
    layers.push({ layer_id: `L${i}`, salt: `layer:L${i}`, marketing: true, kind: "platform" });
    experiments.push({
      experiment_id: `exp_${i}`, flag_key: `exp_${i}`, layer_id: `L${i}`, slot_start: 0, slot_end: 10000,
      ramp_pct: ramp, salt: `exp_${i}`, kill_switch: kill, variants: variants("A", "B"),
    });
  }
  const rs = { version, holdout: { salt: "global_holdout_v1", pct: 0 }, layers, experiments };
  rs.etag = `etag-v${version}-${kill}`;
  return rs;
}

class FakeServer {
  constructor(rs) {
    this.rs = rs;
    this.down = false;
    this.calls = 0;
    this.bodies = 0;
  }
  async fetch(inm) {
    this.calls++;
    if (this.down) throw new Error("сервер недоступен");
    if (inm === this.rs.etag) return { status: 304, body: null, etag: this.rs.etag };
    this.bodies++;
    return { status: 200, body: JSON.stringify(this.rs), etag: this.rs.etag };
  }
}

const IDS = Array.from({ length: 3000 }, (_, i) => `guest_${i}`);

test("решения клиента совпадают с evaluate; контроль правила, а не строка control", () => {
  const rs = makeRuleset({ ramp: 50 });
  const c = new ExperimentClient({ initialRuleset: rs });
  const parsed = parseRuleset(rs);
  for (const u of IDS) {
    const d = c.getVariant("exp_0", u);
    const e = evaluate(parsed, "exp_0", u);
    assert.deepEqual([d.variantId, d.inExperiment, d.reason], [e.variantId, e.inExperiment, e.reason]);
    assert.ok(["A", "B"].includes(d.variantId));
  }
  assert.equal(c.fallbackRate, 0);
});

test("fail-open: нет ruleset, неизвестный флаг, пустой ID", async () => {
  const srv = new FakeServer(makeRuleset());
  srv.down = true;
  const c = new ExperimentClient({ fetcher: srv });
  assert.equal(await c.refresh(), false);
  assert.match(c.lastRefreshError, /недоступен/);
  assert.deepEqual(pick(c.getVariant("exp_0", "g1", { defaultVariant: "A" })), ["A", false, FALLBACK_NO_RULESET]);
  srv.down = false;
  assert.equal(await c.refresh(), true);
  assert.equal(c.getVariant("нет_флага", "g1").reason, FALLBACK_UNKNOWN_FLAG);
  assert.equal(c.getVariant("exp_0", "").reason, FALLBACK_INVALID_UNIT);
  assert.equal(c.getVariant("exp_0", null).variantId, "control");
  const bad = c.getVariant("exp_0", "\ud800");
  assert.deepEqual([bad.variantId, bad.reason], ["A", "fallback:exception:TypeError"]);
  assert.equal(c.getVariant("exp_0", "g1").reason, "assigned");
  assert.ok(Math.abs(c.fallbackRate - 5 / 6) < 1e-12);
  assert.equal(c.metrics().ruleset_version, 1);
});

function pick(d) {
  return [d.variantId, d.inExperiment, d.reason];
}

test("ETag/304, битые ответы и откат версии не портят ruleset", async () => {
  const srv = new FakeServer(makeRuleset());
  const c = new ExperimentClient({ fetcher: srv });
  assert.equal(await c.refresh(), true);
  assert.equal(await c.refresh(), false);
  assert.equal(c.counters.not_modified, 1);
  assert.equal(srv.bodies, 1);
  const good = makeRuleset({ version: 5 });
  const broken = [
    { status: 200, body: "{не json" },
    { status: 500, body: null },
    { status: 200, body: JSON.stringify({ ...good, version: 0 }) },
    { status: 200, body: JSON.stringify({ ...good, experiments: [{ ...good.experiments[0], layer_id: "нет" }] }) },
    { status: 200, body: JSON.stringify(makeRuleset({ version: 2 })) }, // откат
  ];
  for (const resp of broken) {
    const c2 = new ExperimentClient({ fetcher: { fetch: async () => resp }, initialRuleset: good });
    const before = c2.ruleset;
    assert.equal(await c2.refresh(), false, JSON.stringify(resp).slice(0, 60));
    assert.equal(c2.ruleset, before);
  }
});

test("kill-switch доходит при следующем обновлении", async () => {
  const srv = new FakeServer(makeRuleset());
  const c = new ExperimentClient({ fetcher: srv });
  await c.refresh();
  assert.deepEqual(new Set(IDS.slice(0, 200).map((u) => c.getVariant("exp_0", u).variantId)), new Set(["A", "B"]));
  srv.rs = makeRuleset({ version: 2, kill: true });
  await c.refresh();
  const out = new Set(IDS.slice(0, 200).map((u) => pick(c.getVariant("exp_0", u)).join("|")));
  assert.deepEqual(out, new Set(["A|false|kill_switch"]));
});

test("устаревший ruleset → контроль; 304 подтверждает свежесть", async () => {
  let t = 1000;
  const srv = new FakeServer(makeRuleset());
  const c = new ExperimentClient({ fetcher: srv, clock: () => t, refreshIntervalMs: 30_000, maxStaleMs: 300_000 });
  await c.refresh();
  srv.down = true;
  t += 299_000;
  await c.refresh();
  assert.notEqual(c.getVariant("exp_0", "g1").reason, FALLBACK_STALE);
  t += 2_000;
  assert.deepEqual([c.getVariant("exp_0", "g1").variantId, c.getVariant("exp_0", "g1").reason], ["A", FALLBACK_STALE]);
  srv.down = false;
  await c.refresh();
  assert.notEqual(c.getVariant("exp_0", "g1").reason, FALLBACK_STALE);
});

test("проверка настроек", () => {
  assert.throws(() => new ExperimentClient({ refreshIntervalMs: 61_000 }), RangeError);
  assert.throws(() => new ExperimentClient({ refreshIntervalMs: 30_000, maxStaleMs: 10_000 }), RangeError);
});

test("exposure только для участников, один раз на (гость, эксперимент, сессию)", () => {
  const rs = makeRuleset({ ramp: 40, version: 3 });
  rs.holdout.pct = 5;
  const c = new ExperimentClient({ initialRuleset: rs });
  const exposed = new Set();
  for (const u of IDS) {
    let d;
    for (let k = 0; k < 3; k++) d = c.getVariant("exp_0", u, { sessionId: "s1" });
    if (d.inExperiment) exposed.add(u);
  }
  const ev = c.pendingExposures();
  assert.equal(ev.length, exposed.size);
  assert.deepEqual(new Set(ev.map((e) => e.unit_id)), exposed);
  assert.ok(ev.every((e) => e.ruleset_version === 3 && e.session_id === "s1" && /Z$/.test(e.event_time)));
  assert.equal(new Set(ev.map((e) => e.event_id)).size, ev.length);
  assert.ok(exposed.size > 0.3 * IDS.length && exposed.size < 0.5 * IDS.length);
  c.getVariant("exp_0", [...exposed][0], { sessionId: "s2" });
  assert.equal(c.pendingExposures().length, ev.length + 1);
});

test("пачки и повтор после сбоя приёма", async () => {
  const sent = [];
  let fail = true;
  const sink = async (batch) => {
    if (fail) throw new Error("приём недоступен");
    sent.push(batch);
  };
  const c = new ExperimentClient({ initialRuleset: makeRuleset(), sink, batchSize: 50 });
  for (const u of IDS.slice(0, 120)) c.getVariant("exp_0", u);
  await new Promise((r) => setImmediate(r));
  assert.equal(c.counters.sink_error, 2); // только на границах 50 и 100
  assert.equal(c.pendingExposures().length, 120);
  fail = false;
  assert.equal(await c.flush(), 120);
  assert.equal(sent.flat().length, 120);
  for (const u of IDS.slice(120, 230)) c.getVariant("exp_0", u);
  await new Promise((r) => setImmediate(r));
  assert.equal(sent.flat().length, 220);
  assert.equal(c.pendingExposures().length, 10);
});

test("буфер ограничен, потери считаются", () => {
  const c = new ExperimentClient({ initialRuleset: makeRuleset(), maxBuffer: 100, batchSize: 10_000 });
  for (const u of IDS.slice(0, 150)) c.getVariant("exp_0", u);
  assert.equal(c.pendingExposures().length, 100);
  assert.equal(c.counters.dropped, 50);
});

test("фоновое обновление доставляет kill-switch", async () => {
  const srv = new FakeServer(makeRuleset());
  const c = new ExperimentClient({ fetcher: srv, refreshIntervalMs: 50, maxStaleMs: 1000 });
  await c.refresh();
  c.start();
  try {
    srv.rs = makeRuleset({ version: 2, kill: true });
    const deadline = Date.now() + 2000;
    while (Date.now() < deadline && c.ruleset.version === 1) await new Promise((r) => setTimeout(r, 10));
    assert.equal(c.getVariant("exp_0", "g1").reason, "kill_switch");
  } finally {
    await c.close();
  }
});

test("ASG-5: p99 решения < 1 мс на 100 экспериментах", () => {
  const c = new ExperimentClient({ initialRuleset: makeRuleset({ nLayers: 100, ramp: 50 }) });
  const lat = [];
  for (let i = 0; i < 20000; i++) {
    const t = process.hrtime.bigint();
    c.getVariant(`exp_${i % 100}`, `g${i}`, { sessionId: "s" });
    lat.push(Number(process.hrtime.bigint() - t));
  }
  lat.sort((a, b) => a - b);
  const p99 = lat[Math.floor(0.99 * lat.length)];
  assert.ok(p99 < 1_000_000, `p99 = ${p99} нс`);
});

// --- таргетинг, параметры, решение и показ (ТЗ v4.1) -------------------------------------------

const IOS = { platform: "ios", app_version: "5.4" };

function targetedRuleset({ kill = false } = {}) {
  return {
    version: 1,
    holdout: { salt: "global_holdout_v1", pct: 0 },
    layers: [{ layer_id: "L", salt: "layer:L", marketing: true, kind: "platform" }],
    experiments: [{
      experiment_id: "exp", flag_key: "exp", layer_id: "L", slot_start: 0, slot_end: 10000, ramp_pct: 100, salt: "exp",
      kill_switch: kill, unit_type: "guest_id",
      targeting: { platforms: ["ios"], app_version_min: "5.2", app_version_max: "5.9", cities: null, regions: null,
        is_new_guest: null, has_delivery: null },
      variants: [{ variant_id: "c", traffic_percentage: 50, is_control: true, params: { n: 0 } },
        { variant_id: "t", traffic_percentage: 50, is_control: false, params: { n: 1, text: "новое" } }],
    }],
  };
}

test("версии и регистр: правила сравнения", () => {
  assert.deepEqual(parseVersion("5.1.0"), [5, 1]);
  assert.deepEqual(parseVersion("05.01"), [5, 1]);
  assert.deepEqual(parseVersion("0.0"), [0]);
  for (const bad of ["", "5.", ".5", "5..1", "5.1-beta", "v5", " 5.1", "5.1\n", "٥.١", "1.2.3.4.5.6.7", "1234567890", 5.1, null]) {
    assert.equal(parseVersion(bad), null, String(bad));
  }
  assert.equal(compareVersions(parseVersion("5.10"), parseVersion("5.9")), 1);
  assert.equal(compareVersions(parseVersion("5.1"), parseVersion("5.1.0")), 0);
  assert.equal(compareVersions(parseVersion("5"), parseVersion("5.0.1")), -1);
  assert.equal(asciiLower("IOS-Msk_1"), "ios-msk_1");
  assert.equal(asciiLower("\u212a"), "\u212a");
  assert.equal(asciiLower("İ"), "İ");
  assert.equal(asciiLower("МСК"), "МСК");
  assert.equal(knownAttributes(null), null);
  assert.deepEqual(knownAttributes({ platform: "ios", is_new_guest: false, has_delivery: 1, city: null, email: "a@b" }),
    { platform: "ios", is_new_guest: false });
  assert.throws(() => parseRuleset({ ...targetedRuleset(), experiments: [{ ...targetedRuleset().experiments[0],
    targeting: { app_version_min: "5.x" } }] }), /не версия/);
});

test("клиент: решение без показа, показ отдельно, снимок признаков", () => {
  const c = new ExperimentClient({ initialRuleset: targetedRuleset() });
  const d = c.getVariant("exp", "g1", { attributes: { ...IOS, phone: "+7900", city: 5 }, logExposure: false });
  assert.equal(d.inExperiment, true);
  assert.deepEqual(d.params, d.variantId === "t" ? { n: 1, text: "новое" } : { n: 0 });
  assert.equal(c.pendingExposures().length, 0);
  assert.equal(c.logExposure(d, "s1"), true);
  assert.equal(c.logExposure(d, "s1"), false);
  assert.equal(c.logExposure(d, "s2"), true);
  const [e1, e2] = c.pendingExposures();
  assert.deepEqual([e1.session_id, e2.session_id], ["s1", "s2"]);
  assert.deepEqual(e1.attributes, IOS);
  assert.equal(e1.unit_type, "guest_id");
  const fresh = new ExperimentClient({ initialRuleset: targetedRuleset() });
  const out = fresh.getVariant("exp", "g1", { attributes: { platform: "web" } });
  assert.deepEqual([out.variantId, out.inExperiment, out.reason, out.params], ["c", false, "targeting:platform", { n: 0 }]);
  assert.equal(fresh.getVariant("exp", "g1").reason, "targeting_missing:platform");
  assert.equal(fresh.getVariant("exp", "g1", { attributes: IOS, unitType: "device_id", logExposure: false }).reason,
    "unit_type_mismatch");
  assert.equal(fresh.logExposure(out), false);
  assert.equal(fresh.logExposure(null), false);
  assert.equal(fresh.pendingExposures().length, 0);
  assert.deepEqual(fresh.getVariant("нет", "g1").params, {});
});

test("клиент: после первого показа гость остаётся в тесте (TGT-3)", () => {
  const updated = { platform: "ios", app_version: "6.0" };
  const c = new ExperimentClient({ initialRuleset: targetedRuleset() });
  assert.equal(c.getVariant("exp", "g1", { attributes: updated }).reason, "targeting:app_version");
  const first = c.getVariant("exp", "g1", { attributes: IOS, sessionId: "s1" });
  const after = c.getVariant("exp", "g1", { attributes: updated, sessionId: "s2" });
  assert.equal(first.inExperiment && after.inExperiment, true);
  assert.equal(after.variantId, first.variantId);
  assert.equal(c.pendingExposures().length, 2);
  assert.equal(c.getVariant("exp", "g2", { attributes: updated }).reason, "targeting:app_version");
  // решение без показа не закрепляет; kill-switch сильнее закрепления
  const d = new ExperimentClient({ initialRuleset: targetedRuleset() });
  d.getVariant("exp", "g1", { attributes: IOS, logExposure: false });
  assert.equal(d.getVariant("exp", "g1", { attributes: updated }).reason, "targeting:app_version");
  c._ruleset = parseRuleset(targetedRuleset({ kill: true }));
  assert.equal(c.getVariant("exp", "g1", { attributes: updated }).reason, "kill_switch");
});
