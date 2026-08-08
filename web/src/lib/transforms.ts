// Data-hygiene transforms applied between the database and the scoring
// engine / charts. The engine itself stays untouched (parity-tested); these
// run on the input series.
//
// 1. Plausibility filter — mirror of the DB-side is_plausible_value()
//    (migration 0005). The DB guard stops new garbage at insert time; this
//    catches anything already present or arriving via a path without the
//    trigger. Motivating incident: the n8n interest_rate scraper broke in
//    July 2026 and wrote literal 0 every week, dragging the chart to zero.
//
// 2. Building-approvals annualisation — the feed delivers MONTHLY ABS
//    dwelling approvals (~10-18k), but INDICATORS_CONFIG thresholds are
//    ANNUAL (optimal >240k, deficit <180k). Each point becomes the mean of
//    the trailing 12 months of monthly readings x 12, so thresholds, the
//    chart target line, and the score all operate in the units they were
//    written for.

import type { SeriesMap, DataPoint } from './scoring/engine';

export const PLAUSIBLE_RANGES: Record<string, [number, number]> = {
  interest_rate: [0.01, 25],
  household_debt_gdp: [20, 300],
  rental_vacancy_rate: [0.05, 25],
  building_approvals: [1000, 500000],
  mortgage_stress_rate: [0.5, 90],
  unemployment_rate: [0.5, 30],
  auction_clearance_rate: [10, 100],
  mortgage_arrears_rate: [0.01, 20],
};

export interface DroppedPoint {
  indicator: string;
  date: string;
  value: number;
}

export interface SeriesAdjustments {
  dropped: DroppedPoint[];
  /** Indicators whose values were annualised from monthly readings. */
  annualized: string[];
}

function monthsBeforeISO(dateISO: string, months: number): string {
  const d = new Date(dateISO + 'T00:00:00Z');
  d.setUTCMonth(d.getUTCMonth() - months);
  return d.toISOString().slice(0, 10);
}

/** Trailing-12-month mean x 12 for each point of a monthly series. */
export function annualizeMonthly(points: DataPoint[]): DataPoint[] {
  const sorted = [...points].sort((a, b) => (a.date < b.date ? -1 : 1));
  return sorted.map((p) => {
    const windowStart = monthsBeforeISO(p.date, 12);
    const window = sorted.filter((q) => q.date > windowStart && q.date <= p.date);
    const mean = window.reduce((a, q) => a + q.value, 0) / window.length;
    return { ...p, value: Math.round(mean * 12) };
  });
}

/** Heuristic: are these building-approvals readings monthly-scale? The two
 *  unit regimes are far apart (monthly ~8-25k vs annual ~150-260k), so a
 *  max under 60k means monthly data. */
export function looksMonthly(points: DataPoint[]): boolean {
  if (!points.length) return false;
  return Math.max(...points.map((p) => p.value)) < 60000;
}

export function prepareSeries(raw: SeriesMap): { series: SeriesMap; adjustments: SeriesAdjustments } {
  const dropped: DroppedPoint[] = [];
  const annualized: string[] = [];
  const series: SeriesMap = {};

  for (const [name, points] of Object.entries(raw)) {
    const range = PLAUSIBLE_RANGES[name];
    let kept = points;
    if (range) {
      kept = points.filter((p) => {
        const ok = p.value >= range[0] && p.value <= range[1];
        if (!ok) dropped.push({ indicator: name, date: p.date, value: p.value });
        return ok;
      });
    }
    if (name === 'building_approvals' && looksMonthly(kept)) {
      kept = annualizeMonthly(kept);
      annualized.push(name);
    }
    series[name] = kept;
  }

  return { series, adjustments: { dropped, annualized } };
}
