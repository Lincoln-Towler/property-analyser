// Pure view-model helpers for the BTC page (no server-only import, so they
// are unit-testable). Signal thresholds mirror get_btc_signal() in Supabase.

import type { BtcContextFlag, BtcKeyLevel } from './btc';

/** Same bands as get_btc_signal(): score → signal text. */
export function btcSignalForScore(score: number): string {
  if (score >= 9) return 'MAX BUY';
  if (score >= 7.5) return 'STRONG BUY';
  if (score >= 6) return 'BUY';
  if (score >= 5) return 'LIGHT BUY';
  if (score >= 4) return 'HOLD';
  if (score >= 2.5) return 'CAUTION / REDUCE';
  return 'SELL / TAKE PROFIT';
}

/**
 * The best score any price could produce with today's flags: the highest
 * band base plus the net modifier, clamped 1–10 like get_btc_signal().
 * When this is still a sell, price no longer matters — the flags decide.
 */
export function scoreCeiling(levels: BtcKeyLevel[], modifierSum: number): { score: number; signal: string } | null {
  if (!levels.length) return null;
  const best = Math.max(...levels.map((l) => l.base_score));
  const score = Math.max(1, Math.min(10, Math.round((best + modifierSum) * 10) / 10));
  return { score, signal: btcSignalForScore(score) };
}

export interface FlagGroup {
  category: string;
  flags: BtcContextFlag[];
  net: number;
  bearishCount: number;
  bullishCount: number;
  bearishSum: number;
  bullishSum: number;
}

const CATEGORY_ORDER = ['MACRO', 'ON_CHAIN', 'CYCLE'];

/** Active flags grouped by category, most negative modifier first. */
export function groupFlags(flags: BtcContextFlag[]): FlagGroup[] {
  const groups = new Map<string, BtcContextFlag[]>();
  for (const f of flags.filter((x) => x.is_active)) {
    const key = f.category ?? 'OTHER';
    groups.set(key, [...(groups.get(key) ?? []), f]);
  }
  const rank = (c: string) => (CATEGORY_ORDER.includes(c) ? CATEGORY_ORDER.indexOf(c) : CATEGORY_ORDER.length);
  return [...groups.entries()]
    .sort(([a], [b]) => rank(a) - rank(b) || a.localeCompare(b))
    .map(([category, list]) => {
      const sorted = [...list].sort((a, b) => a.score_modifier - b.score_modifier);
      const bearish = sorted.filter((f) => f.score_modifier < 0);
      const bullish = sorted.filter((f) => f.score_modifier > 0);
      const sum = (xs: BtcContextFlag[]) => Math.round(xs.reduce((s, f) => s + f.score_modifier, 0) * 10) / 10;
      return {
        category,
        flags: sorted,
        net: sum(sorted),
        bearishCount: bearish.length,
        bullishCount: bullish.length,
        bearishSum: sum(bearish),
        bullishSum: sum(bullish),
      };
    });
}

export type LadderRow =
  | { kind: 'level'; level: BtcKeyLevel; distancePct: number | null; isCurrentBand: boolean }
  | { kind: 'price'; price: number };

/** Key levels high → low with the current price slotted in where it sits. */
export function buildLadder(levels: BtcKeyLevel[], price: number | null, currentBand: string | null): LadderRow[] {
  const sorted = [...levels].sort((a, b) => b.level_usd - a.level_usd);
  const rows: LadderRow[] = [];
  let placed = price === null;
  for (const level of sorted) {
    if (!placed && price !== null && price >= level.level_usd) {
      rows.push({ kind: 'price', price });
      placed = true;
    }
    rows.push({
      kind: 'level',
      level,
      distancePct: price ? Math.round(((level.level_usd - price) / price) * 1000) / 10 : null,
      isCurrentBand: currentBand !== null && level.label === currentBand,
    });
  }
  if (!placed && price !== null) rows.push({ kind: 'price', price });
  return rows;
}

export const CATEGORY_LABEL: Record<string, string> = {
  MACRO: 'Macro',
  ON_CHAIN: 'On-chain',
  CYCLE: 'Cycle',
  OTHER: 'Other',
};

/** Tailwind classes for a key-level signal chip, sell → max buy. */
export function levelSignalClass(signal: string): string {
  switch (signal) {
    case 'SELL':
      return 'bg-red-500/15 text-red-300 ring-red-500/30';
    case 'HOLD':
      return 'bg-amber-500/15 text-amber-300 ring-amber-500/30';
    case 'LIGHT_BUY':
      return 'bg-yellow-500/15 text-yellow-200 ring-yellow-500/30';
    case 'BUY':
      return 'bg-lime-500/15 text-lime-300 ring-lime-500/30';
    case 'STRONG_BUY':
      return 'bg-green-500/15 text-green-300 ring-green-500/30';
    default:
      return 'bg-emerald-500/20 text-emerald-300 ring-emerald-500/40';
  }
}
