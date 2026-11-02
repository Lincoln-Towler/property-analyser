// Visual tokens shared by server pages and client charts. Recharts needs raw
// colour values, so the palette lives here rather than only in Tailwind.

export const COLORS = {
  accent: '#38bdf8', // sky-400 — the one accent: links, active nav, primary series
  accentSoft: 'rgba(56, 189, 248, 0.12)',
  grid: '#1e293b', // slate-800
  axis: '#94a3b8', // slate-400
  tooltipBg: '#0f172a',
  tooltipBorder: '#334155',
  good: '#10b981', // emerald-500
  goodBand: 'rgba(16, 185, 129, 0.10)',
  warn: '#f59e0b', // amber-500
  bad: '#f87171', // red-400
  secondSeries: '#f472b6', // pink-400
  btc: '#f59e0b',
} as const;

export const TOOLTIP_STYLE = {
  contentStyle: {
    background: COLORS.tooltipBg,
    border: `1px solid ${COLORS.tooltipBorder}`,
    borderRadius: 8,
    fontSize: 12,
  },
  labelStyle: { color: '#cbd5e1' },
} as const;

export const AXIS_TICK = { fill: COLORS.axis, fontSize: 11 } as const;

/** Market-score zones. Thresholds mirror signalForScore in the engine
 *  (pinned by tests/ui.test.ts) — keep the two in step. */
export const SIGNAL_BANDS: Array<{ from: number; to: number; label: string; color: string }> = [
  { from: 0, to: 30, label: 'Wait', color: '#ef4444' },
  { from: 30, to: 40, label: 'Caution', color: '#f97316' },
  { from: 40, to: 50, label: 'Hold', color: '#eab308' },
  { from: 50, to: 60, label: 'Moderate Buy', color: '#a3e635' },
  { from: 60, to: 75, label: 'Buy', color: '#22c55e' },
  { from: 75, to: 100, label: 'Strong Buy', color: '#15803d' },
];

export function signalBand(score: number) {
  return SIGNAL_BANDS.find((b) => score < b.to) ?? SIGNAL_BANDS[SIGNAL_BANDS.length - 1];
}

/** Location scores start at a neutral 50, so green begins higher (70). */
export const LOCATION_GOOD = 70;

/** Tailwind classes for a 0–100 score: green from `good`, amber from 40. */
export function scoreTone(score: number, good = 60): { text: string; bar: string } {
  if (score >= good) return { text: 'text-emerald-400', bar: 'bg-emerald-500' };
  if (score >= 40) return { text: 'text-amber-400', bar: 'bg-amber-500' };
  return { text: 'text-red-400', bar: 'bg-red-500' };
}

export type StatusKind = 'success' | 'warning' | 'danger';

export const STATUS_META: Record<StatusKind, { label: string; dot: string; text: string; tile: string; stroke: string }> = {
  success: {
    label: 'On target',
    dot: 'bg-emerald-500',
    text: 'text-emerald-400',
    tile: 'border-emerald-800/80 bg-emerald-950/30',
    stroke: COLORS.good,
  },
  warning: {
    label: 'Watch',
    dot: 'bg-amber-500',
    text: 'text-amber-400',
    tile: 'border-amber-800/80 bg-amber-950/30',
    stroke: COLORS.warn,
  },
  danger: {
    label: 'Warning',
    dot: 'bg-red-500',
    text: 'text-red-400',
    tile: 'border-red-800/80 bg-red-950/30',
    stroke: COLORS.bad,
  },
};
