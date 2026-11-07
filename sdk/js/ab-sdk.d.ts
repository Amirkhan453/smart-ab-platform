// Типы JS SDK гостевого домена (ab-sdk.mjs).

export const N_BUCKETS: 10000;
export const SMHASHER_VERIFICATION_X86_32: number;

export const REASON_ASSIGNED: "assigned";
export const REASON_HOLDOUT: "holdout";
export const REASON_NOT_IN_LANE: "not_in_lane";
export const REASON_RAMP: "ramp";
export const REASON_KILL_SWITCH: "kill_switch";
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
  | "fallback:no_ruleset"
  | "fallback:stale_ruleset"
  | "fallback:unknown_flag"
  | "fallback:invalid_unit_id"
  | `fallback:exception:${string}`;

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
    variants: Array<{ variant_id: string; traffic_percentage: number; is_control: boolean }>;
  }>;
}

export interface ParsedRuleset {
  readonly version: number;
  readonly etag: string | null;
  readonly holdout: { salt: string; pct: number };
  readonly raw: RulesetJson;
  readonly layers: ReadonlyMap<string, RulesetJson["layers"][number]>;
  readonly flags: ReadonlyMap<string, { rule: RulesetJson["experiments"][number]; control: string; alloc: number[] }>;
}

export function parseRuleset(json: string | RulesetJson): ParsedRuleset;

export interface Evaluation {
  variantId: string;
  inExperiment: boolean;
  reason: Reason;
  experimentId: string;
}

export function evaluate(ruleset: ParsedRuleset, flagKey: string, unitId: string): Evaluation;

export interface Decision {
  flagKey: string;
  unitId: string;
  variantId: string;
  inExperiment: boolean;
  reason: Reason;
  experimentId: string | null;
  rulesetVersion: number | null;
}

export interface ExposureEvent {
  event_id: string;
  event_time: string;
  unit_id: string;
  experiment_id: string;
  variant_id: string;
  ruleset_version: number;
  session_id: string | null;
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
  getVariant(flagKey: string, unitId: string, options?: { sessionId?: string | null; defaultVariant?: string }): Decision;
  metrics(): Record<string, unknown>;
  flush(): Promise<number>;
  pendingExposures(): ExposureEvent[];
}
