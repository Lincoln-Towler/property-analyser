// Visual tokens shared by server pages and client charts. Recharts needs raw
// colour values, so these mirror the @theme block in app/globals.css
// (tests/design.test.ts keeps the two in step). See web/DESIGN.md for the rules.

export const COLORS = {
  accent: '#7aa6f0', // the one accent: links and the main chart series
  primary: '#f2f5f7',
  surface: '#171a1e',
  tile: '#24272a',
  grid: 'rgba(255, 255, 255, 0.07)',
  axis: 'rgba(255, 255, 255, 0.48)',
  secondary: 'rgba(255, 255, 255, 0.62)',
  good: '#47c280',
  warn: '#e0a847',
  bad: '#e96660',
} as const;

export const TOOLTIP_STYLE = {
  contentStyle: {
    background: COLORS.tile,
    border: 'none',
    borderRadius: 8,
    fontSize: 12,
  },
  labelStyle: { color: COLORS.secondary },
  itemStyle: { color: COLORS.primary },
  cursor: { stroke: 'rgba(255, 255, 255, 0.16)' },
} as const;

export const AXIS_TICK = { fill: COLORS.axis, fontSize: 11 } as const;

/** Market-score zones. Thresholds mirror signalForScore in the engine
 *  (pinned by tests/ui.test.ts) — keep the two in step. */
export const SIGNAL_BANDS: Array<{ from: number; to: number; label: string; color: string; fill: string }> = [
  { from: 0, to: 30, label: 'Wait', color: '#e96660', fill: 'bg-danger' },
  { from: 30, to: 40, label: 'Caution', color: '#e08850', fill: 'bg-reduce' },
  { from: 40, to: 50, label: 'Hold', color: '#e0a847', fill: 'bg-caution' },
  { from: 50, to: 60, label: 'Moderate Buy', color: '#c8bc55', fill: 'bg-light-buy' },
  { from: 60, to: 75, label: 'Buy', color: '#8fc060', fill: 'bg-buy' },
  { from: 75, to: 100, label: 'Strong Buy', color: '#47c280', fill: 'bg-go' },
];

export function signalBand(score: number) {
  return SIGNAL_BANDS.find((b) => score < b.to) ?? SIGNAL_BANDS[SIGNAL_BANDS.length - 1];
}

type Neighbour = { label: string; distance: number } | null;

/** How far the score sits above the band below it and below the band above it,
 *  for the meter's caption. Null at either end of the scale. */
export function bandDistances(score: number): { below: Neighbour; above: Neighbour } {
  const i = SIGNAL_BANDS.indexOf(signalBand(score));
  const band = SIGNAL_BANDS[i];
  return {
    below: i > 0 ? { label: SIGNAL_BANDS[i - 1].label, distance: score - band.from } : null,
    above: i < SIGNAL_BANDS.length - 1 ? { label: SIGNAL_BANDS[i + 1].label, distance: band.to - score } : null,
  };
}

/** Location scores start at a neutral 50, so green begins higher (70). */
export const LOCATION_GOOD = 70;

/** Tailwind classes for a 0–100 score: green from `good`, amber from 40. */
export function scoreTone(score: number, good = 60): { text: string; bar: string } {
  if (score >= good) return { text: 'text-go', bar: 'bg-go' };
  if (score >= 40) return { text: 'text-caution', bar: 'bg-caution' };
  return { text: 'text-danger', bar: 'bg-danger' };
}

export type StatusKind = 'success' | 'warning' | 'danger';

export const STATUS_META: Record<StatusKind, { label: string; text: string; fill: string }> = {
  success: { label: 'On target', text: 'text-go', fill: 'bg-go' },
  warning: { label: 'Watch', text: 'text-caution', fill: 'bg-caution' },
  danger: { label: 'Warning', text: 'text-danger', fill: 'bg-danger' },
};
