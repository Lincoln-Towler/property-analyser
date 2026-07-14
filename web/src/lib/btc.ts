// Data layer for the BTC signal page. The btc_* tables are RLS-locked
// (deny-all for anon), so reads happen server-side with the service-role
// key — the same one the snapshot cron uses. Rendered output is cached by
// ISR like every other page; the key never reaches the browser.

import 'server-only';
import { cache } from 'react';
import { createClient } from '@supabase/supabase-js';

export interface BtcKeyLevel {
  level_usd: number;
  label: string;
  signal: string;
  base_score: number;
  emoji: string | null;
  source: string | null;
  notes: string | null;
}

export interface BtcContextFlag {
  flag_key: string;
  flag_label: string;
  category: string | null;
  score_modifier: number;
  current_value: string | null;
  is_active: boolean;
  updated_at: string | null;
}

export interface BtcLogEntry {
  logged_at: string;
  price_usd: number;
  signal: string;
  intensity: string | null;
  score: number;
  base_score: number | null;
  modifier_total: number | null;
  current_band: string | null;
  resistance_level: number | null;
  support_level: number | null;
  pct_to_resistance: number | null;
  pct_to_support: number | null;
}

export interface BtcDailyPoint {
  date: string;
  price: number;
  score: number;
}

export interface BtcData {
  levels: BtcKeyLevel[];
  flags: BtcContextFlag[];
  latest: BtcLogEntry | null;
  daily: BtcDailyPoint[];
  fetchedAt: string;
}

const num = (v: unknown): number => Number(v);
const numOrNull = (v: unknown): number | null => (v == null ? null : Number(v));

/** Returns null when Supabase env vars are absent (e.g. CI builds). */
export const getBtcData = cache(async (): Promise<BtcData | null> => {
  const url = process.env.NEXT_PUBLIC_SUPABASE_URL;
  const serviceKey = process.env.SUPABASE_SERVICE_ROLE_KEY;
  if (!url || !serviceKey) return null;

  const supabase = createClient(url, serviceKey, { auth: { persistSession: false } });

  const [levels, flags, log] = await Promise.all([
    supabase
      .from('btc_key_levels')
      .select('level_usd, label, signal, base_score, emoji, source, notes')
      .order('level_usd', { ascending: false }),
    supabase
      .from('btc_context_flags')
      .select('flag_key, flag_label, category, score_modifier, current_value, is_active, updated_at')
      .order('score_modifier', { ascending: true }),
    supabase
      .from('btc_signal_log')
      .select(
        'logged_at, price_usd, signal, intensity, score, base_score, modifier_total, current_band, resistance_level, support_level, pct_to_resistance, pct_to_support',
      )
      .order('logged_at', { ascending: true }),
  ]);

  if (levels.error) throw new Error(`btc_key_levels query failed: ${levels.error.message}`);
  if (flags.error) throw new Error(`btc_context_flags query failed: ${flags.error.message}`);
  if (log.error) throw new Error(`btc_signal_log query failed: ${log.error.message}`);

  const entries: BtcLogEntry[] = (log.data ?? []).map((r) => ({
    logged_at: r.logged_at,
    price_usd: num(r.price_usd),
    signal: r.signal,
    intensity: r.intensity,
    score: num(r.score),
    base_score: numOrNull(r.base_score),
    modifier_total: numOrNull(r.modifier_total),
    current_band: r.current_band,
    resistance_level: numOrNull(r.resistance_level),
    support_level: numOrNull(r.support_level),
    pct_to_resistance: numOrNull(r.pct_to_resistance),
    pct_to_support: numOrNull(r.pct_to_support),
  }));

  // One point per day (the last log entry that day) — early testing wrote
  // several probe prices with the same timestamp, which would zig-zag a chart.
  const byDay = new Map<string, BtcDailyPoint>();
  for (const e of entries) {
    const date = e.logged_at.slice(0, 10);
    byDay.set(date, { date, price: e.price_usd, score: e.score });
  }

  return {
    levels: (levels.data ?? []).map((r) => ({
      level_usd: num(r.level_usd),
      label: r.label,
      signal: r.signal,
      base_score: num(r.base_score),
      emoji: r.emoji,
      source: r.source,
      notes: r.notes,
    })),
    flags: (flags.data ?? []).map((r) => ({
      flag_key: r.flag_key,
      flag_label: r.flag_label,
      category: r.category,
      score_modifier: num(r.score_modifier),
      current_value: r.current_value,
      is_active: r.is_active,
      updated_at: r.updated_at,
    })),
    latest: entries.length ? entries[entries.length - 1] : null,
    daily: [...byDay.values()],
    fetchedAt: new Date().toISOString(),
  };
});
