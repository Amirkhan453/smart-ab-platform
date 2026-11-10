/**
 * JS SDK гостевого домена (ТЗ v4: 4.1, ASG-5, ASG-6, ASG-7). ES-модуль без зависимостей: браузер и Node ≥ 18.
 *
 * Повторяет эталонный Python SDK (ab_platform/sdk.py) и даёт те же решения: это проверяют общие векторы
 * tests/vectors/assignment_vectors.json.
 *
 * - Хэш: MurmurHash3 x86_32, seed 0, ключ "{salt}:{unit_id}" в UTF-8, бакет = hash mod 10000.
 * - Округление долей — половина к чётному, как round() в Python (12.5 → 12). Math.round здесь нельзя.
 * - Решение считается локально; ruleset обновляется по ETag не реже раза в 30 с.
 * - Fail-open: при любой ошибке — контрольный вариант с причиной fallback:*.
 * - Exposure — только для гостя в тесте, один раз на (гость, эксперимент, сессию), пачками.
 * - Таргетинг (ТЗ v4.1): условия проверяются после дорожки и до ramp-up, одинаково для всех вариантов.
 *   Правила сравнения — как в ab_platform/schemas/targeting.py: версии по числам, регистр только A–Z,
 *   булевы признаки строго true/false. После первого показа гость остаётся в тесте (в пределах клиента).
 * - Решение и показ можно разделить: getVariant(..., { logExposure: false }), затем logExposure(decision).
 */

export const N_BUCKETS = 10000;
export const SMHASHER_VERIFICATION_X86_32 = 0xb0f57ee3;

export const REASON_ASSIGNED = "assigned";
export const REASON_HOLDOUT = "holdout";
export const REASON_NOT_IN_LANE = "not_in_lane";
export const REASON_RAMP = "ramp";
export const REASON_KILL_SWITCH = "kill_switch";
export const REASON_UNIT_TYPE = "unit_type_mismatch";
export const TARGETING_ATTRIBUTES = ["platform", "app_version", "city", "region", "is_new_guest", "has_delivery"];
export const FALLBACK_NO_RULESET = "fallback:no_ruleset";
export const FALLBACK_STALE = "fallback:stale_ruleset";
export const FALLBACK_UNKNOWN_FLAG = "fallback:unknown_flag";
export const FALLBACK_INVALID_UNIT = "fallback:invalid_unit_id";

const ENCODER = new TextEncoder();

// --- хэш и бакеты -------------------------------------------------------------------------------

/** Канонический MurmurHash3 x86_32; результат — беззнаковое 32-битное число. Строка кодируется в UTF-8. */
export function murmurhash3_x86_32(data, seed = 0) {
  const bytes = typeof data === "string" ? ENCODER.encode(data) : data;
  const len = bytes.length;
  const nblocks = len >>> 2;
  const c1 = 0xcc9e2d51;
  const c2 = 0x1b873593;
  let h = seed | 0;
  for (let i = 0; i < nblocks; i++) {
    const j = i * 4;
    let k = bytes[j] | (bytes[j + 1] << 8) | (bytes[j + 2] << 16) | (bytes[j + 3] << 24);
    k = Math.imul(k, c1);
    k = (k << 15) | (k >>> 17);
    k = Math.imul(k, c2);
    h ^= k;
    h = (h << 13) | (h >>> 19);
    h = (Math.imul(h, 5) + 0xe6546b64) | 0;
  }
  const t = nblocks * 4;
  let k = 0;
  switch (len & 3) {
    case 3:
      k ^= bytes[t + 2] << 16;
    // fallthrough
    case 2:
      k ^= bytes[t + 1] << 8;
    // fallthrough
    case 1:
      k ^= bytes[t];
      k = Math.imul(k, c1);
      k = (k << 15) | (k >>> 17);
      k = Math.imul(k, c2);
      h ^= k;
  }
  h ^= len;
  h ^= h >>> 16;
  h = Math.imul(h, 0x85ebca6b);
  h ^= h >>> 13;
  h = Math.imul(h, 0xc2b2ae35);
  h ^= h >>> 16;
  return h >>> 0;
}

/** Проверочная сумма smhasher; эталон 0xB0F57EE3 — общий для всех языков. */
export function smhasherSelfTest() {
  const key = new Uint8Array(256);
  const hashes = new Uint8Array(1024);
  const view = new DataView(hashes.buffer);
  for (let i = 0; i < 256; i++) {
    key[i] = i;
    view.setUint32(i * 4, murmurhash3_x86_32(key.subarray(0, i), 256 - i), true);
  }
  return murmurhash3_x86_32(hashes, 0);
}

/** Бакет [0, 10000): murmur3("{salt}:{unitId}") mod 10000. Пустые и не-строки — ошибка. */
export function bucket(salt, unitId) {
  if (typeof unitId !== "string" || unitId.length === 0) throw new TypeError("unitId должен быть непустой строкой.");
  if (typeof salt !== "string" || salt.length === 0) throw new TypeError("salt должен быть непустой строкой.");
  const key = `${salt}:${unitId}`;
  // Python не кодирует одиночные суррогаты в UTF-8 (ошибка), TextEncoder молча заменил бы их на U+FFFD
  if (typeof key.isWellFormed === "function" && !key.isWellFormed()) {
    throw new TypeError("unitId содержит некорректный UTF-16 (одиночный суррогат).");
  }
  return murmurhash3_x86_32(key) % N_BUCKETS;
}

/** Округление половины к чётному — как round() в Python для float. */
export function roundHalfEven(x) {
  const f = Math.floor(x);
  const d = x - f; // точно для |x| < 2^52
  if (d > 0.5) return f + 1;
  if (d < 0.5) return f;
  return f % 2 === 0 ? f : f + 1;
}

/** Доля в процентах → число бакетов из 10 000. */
export function pctToBuckets(pct) {
  return roundHalfEven(pct * 100);
}

/** Конец активного поддиапазона дорожки при раскатке: s + round(ramp/100 · (e − s)). */
export function rampEnd(slotStart, slotEnd, rampPct) {
  return slotStart + roundHalfEven((rampPct / 100.0) * (slotEnd - slotStart));
}

/** Бакеты на варианты, сумма ровно 10 000 (наибольший остаток; при равных остатках — по порядку). */
export function allocateVariantBuckets(percentages) {
  const raw = percentages.map((p) => p * 100);
  const counts = raw.map((r) => Math.trunc(r));
  const order = raw.map((_, i) => i).sort((a, b) => raw[b] - counts[b] - (raw[a] - counts[a])); // sort устойчив
  const deficit = N_BUCKETS - counts.reduce((s, c) => s + c, 0);
  for (const i of order.slice(0, Math.max(0, deficit))) counts[i] += 1;
  return counts;
}

// --- таргетинг ----------------------------------------------------------------------------------

/** Нижний регистр только для A–Z: toLowerCase() зависит от Юникода и расходится между языками. */
export function asciiLower(text) {
  return text.replace(/[A-Z]/g, (c) => String.fromCharCode(c.charCodeAt(0) + 32));
}

const VERSION_RE = /^[0-9]{1,9}(\.[0-9]{1,9}){0,5}$/;

/** Версия → массив чисел без хвостовых нулей; null, если это не версия. */
export function parseVersion(text) {
  if (typeof text !== "string" || !VERSION_RE.test(text)) return null;
  const parts = text.split(".").map((p) => Number.parseInt(p, 10));
  while (parts.length > 1 && parts[parts.length - 1] === 0) parts.pop();
  return parts;
}

/** Сравнение версий: −1, 0, 1. Короткая версия, совпадающая с началом длинной, меньше. */
export function compareVersions(a, b) {
  for (let i = 0; i < Math.min(a.length, b.length); i++) {
    if (a[i] !== b[i]) return a[i] < b[i] ? -1 : 1;
  }
  return a.length === b.length ? 0 : a.length < b.length ? -1 : 1;
}

function compileTargeting(t, experimentId) {
  t = t ?? {};
  const list = (v) => (v == null ? null : new Set(v));
  const version = (v, name) => {
    if (v == null) return null;
    const parsed = parseVersion(v);
    if (!parsed) throw new Error(`ruleset: у '${experimentId}' ${name} — не версия`);
    return parsed;
  };
  const c = {
    platform: list(t.platforms),
    city: list(t.cities),
    region: list(t.regions),
    versionMin: version(t.app_version_min, "app_version_min"),
    versionMax: version(t.app_version_max, "app_version_max"),
    is_new_guest: t.is_new_guest ?? null,
    has_delivery: t.has_delivery ?? null,
  };
  const used = {
    platform: c.platform !== null,
    app_version: c.versionMin !== null || c.versionMax !== null,
    city: c.city !== null,
    region: c.region !== null,
    is_new_guest: c.is_new_guest !== null,
    has_delivery: c.has_delivery !== null,
  };
  c.required = TARGETING_ATTRIBUTES.filter((a) => used[a]);
  return c;
}

/** null — гость подходит; иначе причина по первому невыполненному условию (порядок TARGETING_ATTRIBUTES). */
export function targetingMismatch(compiled, attributes) {
  const attrs = attributes ?? {};
  for (const name of compiled.required) {
    const value = Object.prototype.hasOwnProperty.call(attrs, name) ? attrs[name] : undefined;
    if (name === "platform" || name === "city" || name === "region") {
      if (typeof value !== "string") return `targeting_missing:${name}`;
      if (!compiled[name].has(asciiLower(value))) return `targeting:${name}`;
    } else if (name === "app_version") {
      const v = parseVersion(value);
      if (!v) return `targeting_missing:${name}`;
      if ((compiled.versionMin && compareVersions(v, compiled.versionMin) < 0) ||
          (compiled.versionMax && compareVersions(v, compiled.versionMax) > 0)) {
        return `targeting:${name}`;
      }
    } else {
      if (typeof value !== "boolean") return `targeting_missing:${name}`;
      if (value !== compiled[name]) return `targeting:${name}`;
    }
  }
  return null;
}

/** Снимок признаков для показа: только признаки таргетинга и только строки и булевы. */
export function knownAttributes(attributes) {
  if (!attributes) return null;
  const out = {};
  for (const name of TARGETING_ATTRIBUTES) {
    const v = Object.prototype.hasOwnProperty.call(attributes, name) ? attributes[name] : undefined;
    if (typeof v === "string" || typeof v === "boolean") out[name] = v;
  }
  return Object.keys(out).length ? out : null;
}

// --- ruleset ------------------------------------------------------------------------------------

/**
 * Разбор ruleset из JSON сервера. Сервер уже проверил его целиком; здесь — проверки, без которых
 * решение было бы неверным. Ошибка → исключение (SDK останется на прежнем ruleset).
 */
export function parseRuleset(json) {
  const rs = typeof json === "string" ? JSON.parse(json) : json;
  if (!rs || !Number.isInteger(rs.version) || rs.version < 1) throw new Error("ruleset: version должен быть целым ≥ 1");
  if (!rs.holdout || typeof rs.holdout.salt !== "string" || typeof rs.holdout.pct !== "number") {
    throw new Error("ruleset: некорректный holdout");
  }
  if (!Array.isArray(rs.layers) || !Array.isArray(rs.experiments)) throw new Error("ruleset: нет layers/experiments");
  const layers = new Map(rs.layers.map((l) => [l.layer_id, l]));
  const flags = new Map();
  for (const e of rs.experiments) {
    if (!layers.has(e.layer_id)) throw new Error(`ruleset: '${e.experiment_id}' ссылается на неизвестный слой`);
    if (!Array.isArray(e.variants) || e.variants.length < 2) throw new Error(`ruleset: у '${e.experiment_id}' < 2 вариантов`);
    if (e.variants.filter((v) => v.is_control).length !== 1) throw new Error(`ruleset: у '${e.experiment_id}' не один контроль`);
    if (!(Number.isInteger(e.slot_start) && Number.isInteger(e.slot_end) && 0 <= e.slot_start && e.slot_start < e.slot_end && e.slot_end <= N_BUCKETS)) {
      throw new Error(`ruleset: некорректная дорожка '${e.experiment_id}'`);
    }
    if (flags.has(e.flag_key)) throw new Error(`ruleset: flag_key '${e.flag_key}' повторяется`);
    const control = e.variants.find((v) => v.is_control).variant_id;
    const alloc = allocateVariantBuckets(e.variants.map((v) => v.traffic_percentage));
    const params = new Map(e.variants.map((v) => [v.variant_id, v.params ?? {}]));
    flags.set(e.flag_key, { rule: e, control, alloc, params, targeting: compileTargeting(e.targeting, e.experiment_id),
      unitType: e.unit_type ?? "guest_id" });
  }
  return { version: rs.version, etag: rs.etag ?? null, holdout: rs.holdout, raw: rs, layers, flags };
}

/**
 * Вариант гостя по правилу (одна точка вычисления, как evaluate в Python).
 * Порядок: kill-switch → тип единицы → holdout → слой и дорожка → таргетинг → ramp-up → вариант.
 * unitType — тип переданного идентификатора (null — не сверять); enrolled — первый показ уже был.
 */
export function evaluate(ruleset, flagKey, unitId, { attributes = null, unitType = null, enrolled = false } = {}) {
  const entry = ruleset.flags.get(flagKey);
  if (!entry) throw new Error(`неизвестный флаг '${flagKey}'`);
  const { rule, control, alloc, params } = entry;
  const out = (reason) => ({ variantId: control, inExperiment: false, reason, experimentId: rule.experiment_id,
    params: params.get(control) });
  if (rule.kill_switch) return out(REASON_KILL_SWITCH);
  if (unitType != null && unitType !== entry.unitType) return out(REASON_UNIT_TYPE);
  const layer = ruleset.layers.get(rule.layer_id);
  const holdoutPct = layer.marketing ? ruleset.holdout.pct : 0;
  if (holdoutPct > 0 && bucket(ruleset.holdout.salt, unitId) < pctToBuckets(holdoutPct)) return out(REASON_HOLDOUT);
  const zl = bucket(layer.salt, unitId);
  if (!(rule.slot_start <= zl && zl < rule.slot_end)) return out(REASON_NOT_IN_LANE);
  if (!enrolled) {
    const mismatch = targetingMismatch(entry.targeting, attributes);
    if (mismatch) return out(mismatch);
  }
  if (zl >= rampEnd(rule.slot_start, rule.slot_end, rule.ramp_pct)) return out(REASON_RAMP);
  const z = bucket(rule.salt, unitId);
  let cum = 0;
  let chosen = rule.variants[rule.variants.length - 1].variant_id;
  for (let i = 0; i < alloc.length; i++) {
    cum += alloc[i];
    if (z < cum) {
      chosen = rule.variants[i].variant_id;
      break;
    }
  }
  return { variantId: chosen, inExperiment: true, reason: REASON_ASSIGNED, experimentId: rule.experiment_id,
    params: params.get(chosen) };
}

// --- HTTP-транспорт -----------------------------------------------------------------------------

/** Источник ruleset: GET {baseUrl}/ruleset с If-None-Match. */
export function createHttpFetcher(baseUrl, { timeoutMs = 5000, fetchImpl = globalThis.fetch } = {}) {
  const url = `${baseUrl.replace(/\/+$/, "")}/ruleset`;
  return {
    async fetch(ifNoneMatch) {
      const headers = ifNoneMatch ? { "If-None-Match": `"${ifNoneMatch}"` } : {};
      const r = await fetchImpl(url, { headers, signal: AbortSignal.timeout(timeoutMs) });
      const etag = r.headers.get("etag")?.replace(/^W\//, "").replace(/"/g, "") ?? null;
      return { status: r.status, body: r.status === 200 ? await r.text() : null, etag };
    },
  };
}

/** Отправка exposure: POST {baseUrl}/exposures. Не-2xx → исключение (SDK повторит пачку). */
export function createHttpSink(baseUrl, { timeoutMs = 5000, fetchImpl = globalThis.fetch } = {}) {
  const url = `${baseUrl.replace(/\/+$/, "")}/exposures`;
  const sink = async (batch) => {
    const r = await fetchImpl(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(batch),
      signal: AbortSignal.timeout(timeoutMs),
    });
    if (!r.ok) throw new Error(`приём exposure: статус ${r.status}`);
    const res = await r.json();
    sink.rejected.push(...(res.rejected ?? [])); // повтором не исправить — храним для диагностики
  };
  sink.rejected = [];
  return sink;
}

// --- клиент -------------------------------------------------------------------------------------

function uuid4() {
  if (globalThis.crypto?.randomUUID) return globalThis.crypto.randomUUID();
  const b = new Uint8Array(16);
  for (let i = 0; i < 16; i++) b[i] = Math.floor(Math.random() * 256);
  b[6] = (b[6] & 0x0f) | 0x40;
  b[8] = (b[8] & 0x3f) | 0x80;
  const h = [...b].map((x) => x.toString(16).padStart(2, "0")).join("");
  return `${h.slice(0, 8)}-${h.slice(8, 12)}-${h.slice(12, 16)}-${h.slice(16, 20)}-${h.slice(20)}`;
}

export class ExperimentClient {
  constructor({
    fetcher = null,
    sink = null,
    initialRuleset = null,
    refreshIntervalMs = 30_000,
    maxStaleMs = 300_000,
    batchSize = 100,
    maxBuffer = 100_000,
    dedupCapacity = 1_000_000,
    clock = () => performance.now(),
  } = {}) {
    if (!(refreshIntervalMs > 0 && refreshIntervalMs <= 30_000)) {
      throw new RangeError("refreshIntervalMs должен быть в (0, 30000] — иначе kill-switch не уложится в 60 с.");
    }
    if (maxStaleMs < refreshIntervalMs) throw new RangeError("maxStaleMs не может быть меньше refreshIntervalMs.");
    this._fetcher = fetcher;
    this._sink = sink;
    this._clock = clock;
    this.refreshIntervalMs = refreshIntervalMs;
    this.maxStaleMs = maxStaleMs;
    this._batchSize = batchSize;
    this._maxBuffer = maxBuffer;
    this._dedupCapacity = dedupCapacity;
    this._ruleset = initialRuleset == null ? null : parseRuleset(initialRuleset);
    this._fetchedAt = this._ruleset ? clock() : null;
    this._buffer = [];
    this._seen = new Map();
    this._enrolled = new Map(); // "гость, эксперимент" с первым показом
    this._timer = null;
    this._tick = null;
    this.decisions = {};
    this.counters = { refresh_ok: 0, not_modified: 0, refresh_error: 0, sent: 0, sink_error: 0, dropped: 0 };
    this.lastRefreshError = null;
  }

  get ruleset() {
    return this._ruleset;
  }

  rulesetAgeMs() {
    return this._fetchedAt == null ? null : this._clock() - this._fetchedAt;
  }

  /** Один запрос к источнику. true — ruleset заменён. Не бросает: при ошибке остаётся прежний ruleset. */
  async refresh() {
    if (!this._fetcher) return false;
    const current = this._ruleset;
    try {
      const { status, body } = await this._fetcher.fetch(current ? current.etag : null);
      if (status === 304 && current) {
        this._fetchedAt = this._clock();
        this.counters.not_modified += 1;
        return false;
      }
      if (status !== 200 || body == null) throw new Error(`статус ${status}`);
      const next = parseRuleset(body);
      if (current && next.version < current.version) {
        throw new Error(`версия ${next.version} старше текущей ${current.version}`);
      }
      this._ruleset = next;
      this._fetchedAt = this._clock();
      this.counters.refresh_ok += 1;
      return true;
    } catch (err) {
      this.counters.refresh_error += 1;
      this.lastRefreshError = `${err?.name ?? "Error"}: ${err?.message ?? err}`;
      return false;
    }
  }

  /** Фоновое обновление ruleset и отправка exposure каждые refreshIntervalMs. */
  start() {
    if (this._timer) return;
    this._timer = setInterval(() => {
      if (this._tick) return; // предыдущий цикл ещё идёт
      this._tick = (async () => {
        await this.refresh();
        await this.flush();
      })().finally(() => {
        this._tick = null;
      });
    }, this.refreshIntervalMs);
    this._timer.unref?.(); // не держать процесс Node живым
  }

  async close() {
    if (this._timer) clearInterval(this._timer);
    this._timer = null;
    if (this._tick) await this._tick;
    await this.flush();
  }

  _fallback(flagKey, unitId, reason, defaultVariant, entry = null) {
    this._count(reason);
    return {
      flagKey,
      unitId: String(unitId),
      variantId: entry ? entry.control : defaultVariant,
      inExperiment: false,
      reason,
      experimentId: entry ? entry.rule.experiment_id : null,
      rulesetVersion: this._ruleset ? this._ruleset.version : null,
      params: entry ? entry.params.get(entry.control) : {},
      unitType: entry ? entry.unitType : null,
      attributes: null,
    };
  }

  /** Вариант гостя. Никогда не бросает. defaultVariant — если ruleset или флаг неизвестны. */
  getVariant(flagKey, unitId, { sessionId = null, defaultVariant = "control", attributes = null, unitType = null,
    logExposure = true } = {}) {
    try {
      const rs = this._ruleset;
      if (typeof unitId !== "string" || unitId.length === 0) {
        return this._fallback(flagKey, unitId, FALLBACK_INVALID_UNIT, defaultVariant);
      }
      if (!rs) return this._fallback(flagKey, unitId, FALLBACK_NO_RULESET, defaultVariant);
      const entry = rs.flags.get(flagKey);
      if (!entry) return this._fallback(flagKey, unitId, FALLBACK_UNKNOWN_FLAG, defaultVariant);
      const age = this.rulesetAgeMs();
      if (this._fetcher && age != null && age > this.maxStaleMs) {
        return this._fallback(flagKey, unitId, FALLBACK_STALE, defaultVariant, entry);
      }
      let res;
      try {
        const enrolled = this._enrolled.has(JSON.stringify([unitId, entry.rule.experiment_id]));
        res = evaluate(rs, flagKey, unitId, { attributes, unitType, enrolled });
      } catch (err) {
        return this._fallback(flagKey, unitId, `fallback:exception:${err?.name ?? "Error"}`, defaultVariant, entry);
      }
      this._count(res.reason);
      const decision = { flagKey, unitId, ...res, rulesetVersion: rs.version, unitType: entry.unitType,
        attributes: knownAttributes(attributes) };
      if (logExposure) this.logExposure(decision, sessionId);
      return decision;
    } catch (err) {
      // последняя линия fail-open: даже сбой учёта не должен уронить приложение
      return { flagKey, unitId: String(unitId), variantId: defaultVariant, inExperiment: false,
        reason: `fallback:exception:${err?.name ?? "Error"}`, experimentId: null, rulesetVersion: null, params: {},
        unitType: null, attributes: null };
    }
  }

  /** Показ по решению: только для гостя в тесте. false — показ не нужен или уже учтён. */
  logExposure(decision, sessionId = null) {
    if (!decision || !decision.inExperiment || decision.experimentId == null || decision.rulesetVersion == null) {
      return false;
    }
    return this._logExposure(decision.unitId, decision.experimentId, decision.variantId, decision.rulesetVersion,
      sessionId, decision.unitType ?? null, decision.attributes ?? null);
  }

  _count(reason) {
    this.decisions[reason] = (this.decisions[reason] ?? 0) + 1;
  }

  get fallbackRate() {
    let total = 0;
    let fb = 0;
    for (const [r, n] of Object.entries(this.decisions)) {
      total += n;
      if (r.startsWith("fallback:")) fb += n;
    }
    return total ? fb / total : 0;
  }

  metrics() {
    const rs = this._ruleset;
    return {
      decisions: { ...this.decisions },
      fallback_rate: this.fallbackRate,
      ruleset_version: rs ? rs.version : null,
      ruleset_etag: rs ? rs.etag : null,
      ruleset_age_ms: this.rulesetAgeMs(),
      buffered_exposures: this._buffer.length,
      ...this.counters,
    };
  }

  _logExposure(unitId, experimentId, variantId, version, sessionId, unitType = null, attributes = null) {
    const enrolledKey = JSON.stringify([unitId, experimentId]);
    this._enrolled.delete(enrolledKey);
    this._enrolled.set(enrolledKey, true);
    if (this._enrolled.size > this._dedupCapacity) this._enrolled.delete(this._enrolled.keys().next().value);
    const key = JSON.stringify([unitId, experimentId, sessionId]);
    if (this._seen.has(key)) {
      this._seen.delete(key);
      this._seen.set(key, true); // в конец очереди вытеснения
      return false;
    }
    this._seen.set(key, true);
    if (this._seen.size > this._dedupCapacity) this._seen.delete(this._seen.keys().next().value);
    if (this._buffer.length >= this._maxBuffer) {
      this._buffer.shift();
      this.counters.dropped += 1;
    }
    this._buffer.push({
      event_id: uuid4(),
      event_time: new Date().toISOString(),
      unit_id: unitId,
      experiment_id: experimentId,
      variant_id: variantId,
      ruleset_version: version,
      session_id: sessionId,
      unit_type: unitType,
      attributes,
    });
    // отправка на границе пачки: после сбоя приёма не долбим его на каждом событии
    if (this._buffer.length % this._batchSize === 0) void this.flush();
    return true;
  }

  /** Отправить буфер. При ошибке события возвращаются в буфер (повтор идемпотентен по event_id). */
  async flush() {
    if (!this._sink || this._buffer.length === 0) return 0;
    const batch = this._buffer;
    this._buffer = [];
    try {
      await this._sink(batch);
      this.counters.sent += batch.length;
      return batch.length;
    } catch {
      this.counters.sink_error += 1;
      const merged = batch.concat(this._buffer);
      this.counters.dropped += Math.max(0, merged.length - this._maxBuffer);
      this._buffer = merged.slice(-this._maxBuffer);
      return 0;
    }
  }

  pendingExposures() {
    return [...this._buffer];
  }
}
