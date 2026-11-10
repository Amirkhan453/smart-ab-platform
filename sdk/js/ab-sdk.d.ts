// Типы JS SDK гостевого домена (ab-sdk.mjs).

export const N_BUCKETS: 10000;
export const SMHASHER_VERIFICATION_X86_32: number;

export const REASON_ASSIGNED: "assigned";
export const REASON_HOLDOUT: "holdout";
export const REASON_NOT_IN_LANE: "not_in_lane";
export const REASON_RAMP: "ramp";
export const REASON_KILL_SWITCH: "kill_switch";
export const REASON_UNIT_TYPE: "unit_type_mismatch";
export const TARGETING_ATTRIBUTES: readonly ["platform", "app_version", "city", "region", "is_new_guest", "has_delivery"];
export const FALLBACK_NO_RULESET: "fallback:no_ruleset";
export const FALLBACK_STALE: "fallback:stale_ruleset";
export const FALLBACK_UNKNOWN_FLAG: "fallback:unknown_flag";
export const FALLBACK_INVALID_UNIT: "fallback:invalid_unit_id";

export type Reason =
  | "assigned"
  | "holdout"
  | "not_in_lane"
  | "ramp"
  | "kill_switch"
  | "unit_type_mismatch"
  | `targeting:${TargetingAttribute}`
  | `targeting_missing:${TargetingAttribute}`
  | "fallback:no_ruleset"
  | "fallback:stale_ruleset"
  | "fallback:unknown_flag"
  | "fallback:invalid_unit_id"
  | `fallback:exception:${string}`;

export type TargetingAttribute = "platform" | "app_version" | "city" | "region" | "is_new_guest" | "has_delivery";
export type UnitType = "guest_id" | "device_id";

/** Признаки гостя для таргетинга. Булевы — строго true/false; версия — числа через точку. */
export interface Attributes {
  platform?: string;
  app_version?: string;
  city?: string;
  region?: string;
  is_new_guest?: boolean;
  has_delivery?: boolean;
}

/** Условия таргетинга в ruleset: все заданные должны выполняться; null — условие не задано. */
export interface TargetingJson {
  platforms: string[] | null;
  app_version_min: string | null;
  app_version_max: string | null;
  cities: string[] | null;
  regions: string[] | null;
  is_new_guest: boolean | null;
  has_delivery: boolean | null;
}

export function asciiLower(text: string): string;
export function parseVersion(text: unknown): number[] | null;
export function compareVersions(a: number[], b: number[]): -1 | 0 | 1;
export function knownAttributes(attributes: Attributes | null | undefined): Attributes | null;

export function murmurhash3_x86_32(data: string | Uint8Array, seed?: number): number;
export function smhasherSelfTest(): number;
export function bucket(salt: string, unitId: string): number;
export function roundHalfEven(x: number): number;
export function pctToBuckets(pct: number): number;
export function rampEnd(slotStart: number, slotEnd: number, rampPct: number): number;
export function allocateVariantBuckets(percentages: number[]): number[];

/** Ruleset в формате сервера (GET /ruleset). */
export interface RulesetJson {
  version: number;
  etag?: string;
  holdout: { salt: string; pct: number };
  layers: Array<{ layer_id: string; salt: string; marketing: boolean; kind: "platform" | "external_flags" }>;
  experiments: Array<{
    experiment_id: string;
    flag_key: string;
    layer_id: string;
    slot_start: number;
    slot_end: number;
    ramp_pct: number;
    salt: string;
    kill_switch: boolean;
    variants: Array<{ variant_id: string; traffic_percentage: number; is_control: boolean; params?: Record<string, unknown> }>;
    unit_type?: UnitType;
    targeting?: TargetingJson;
    passport_id?: string | null;
    passport_version?: number | null;
    passport_hash?: string | null;
  }>;
}

export interface ParsedRuleset {
  readonly version: number;
  readonly etag: string | null;
  readonly holdout: { salt: string; pct: number };
  readonly raw: RulesetJson;
  readonly layers: ReadonlyMap<string, RulesetJson["layers"][number]>;
  readonly flags: ReadonlyMap<string, {
    rule: RulesetJson["experiments"][number];
    control: string;
    alloc: number[];
    params: ReadonlyMap<string, Record<string, unknown>>;
    unitType: UnitType;
  }>;
}

export function parseRuleset(json: string | RulesetJson): ParsedRuleset;

export interface Evaluation {
  variantId: string;
  inExperiment: boolean;
  reason: Reason;
  experimentId: string;
  /** Параметры выданного варианта (для контроля вне теста — параметры контроля). */
  params: Record<string, unknown>;
}

export interface EvaluateOptions {
  attributes?: Attributes | null;
  /** Тип переданного идентификатора; не совпал с правилом — контроль с причиной unit_type_mismatch. */
  unitType?: UnitType | null;
  /** У гостя уже был первый показ в этом тесте: таргетинг не перепроверяется. */
  enrolled?: boolean;
}

export function evaluate(ruleset: ParsedRuleset, flagKey: string, unitId: string, options?: EvaluateOptions): Evaluation;

export interface Decision {
  flagKey: string;
  unitId: string;
  variantId: string;
  inExperiment: boolean;
  reason: Reason;
  experimentId: string | null;
  rulesetVersion: number | null;
  params: Record<string, unknown>;
  unitType: UnitType | null;
  attributes: Attributes | null;
}

export interface ExposureEvent {
  event_id: string;
  event_time: string;
  unit_id: string;
  experiment_id: string;
  variant_id: string;
  ruleset_version: number;
  session_id: string | null;
  unit_type: UnitType | null;
  attributes: Attributes | null;
}

export interface RulesetFetcher {
  fetch(ifNoneMatch: string | null): Promise<{ status: number; body: string | null; etag: string | null }>;
}

export type ExposureSink = (batch: ExposureEvent[]) => Promise<void> | void;

export interface HttpOptions {
  timeoutMs?: number;
  fetchImpl?: typeof fetch;
}

export function createHttpFetcher(baseUrl: string, options?: HttpOptions): RulesetFetcher;
export function createHttpSink(baseUrl: string, options?: HttpOptions): ExposureSink & { rejected: Array<{ index: number; error: string }> };

export interface ClientOptions {
  fetcher?: RulesetFetcher | null;
  sink?: ExposureSink | null;
  initialRuleset?: string | RulesetJson | null;
  /** (0, 30000] мс. */
  refreshIntervalMs?: number;
  /** Старше — отдаётся контроль (fallback:stale_ruleset). */
  maxStaleMs?: number;
  batchSize?: number;
  maxBuffer?: number;
  dedupCapacity?: number;
  clock?: () => number;
}

export class ExperimentClient {
  constructor(options?: ClientOptions);
  readonly ruleset: ParsedRuleset | null;
  readonly refreshIntervalMs: number;
  readonly maxStaleMs: number;
  readonly decisions: Record<string, number>;
  readonly counters: {
    refresh_ok: number;
    not_modified: number;
    refresh_error: number;
    sent: number;
    sink_error: number;
    dropped: number;
  };
  readonly lastRefreshError: string | null;
  readonly fallbackRate: number;
  rulesetAgeMs(): number | null;
  refresh(): Promise<boolean>;
  start(): void;
  close(): Promise<void>;
  getVariant(
    flagKey: string,
    unitId: string,
    options?: {
      sessionId?: string | null;
      defaultVariant?: string;
      attributes?: Attributes | null;
      unitType?: UnitType | null;
      /** false — только решение; показ отправляется позже через logExposure(). */
      logExposure?: boolean;
    },
  ): Decision;
  /** Показ по решению. false — гость вне теста или показ в этой сессии уже учтён. */
  logExposure(decision: Decision, sessionId?: string | null): boolean;
  metrics(): Record<string, unknown>;
  flush(): Promise<number>;
  pendingExposures(): ExposureEvent[];
}
