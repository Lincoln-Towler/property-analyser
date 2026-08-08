// Per-location investment scoring, ported from calculate_location_score()
// in property_analysis_dashboard.py (inside show_location_analysis).
//
// PARITY NOTE: the Python guards read `if growth and growth > 10`, so a
// value of exactly 0 is falsy and skips its whole branch. That is preserved
// here — most visibly for vacancy_rate, where a 0.00 reading (which in this
// dataset means "no data", not "zero vacancy") earns no bonus at all.

import type { PropertyPoint } from './engine';

export const LOCATION_METRICS = [
  'median_price',
  'annual_growth',
  'rental_yield',
  'vacancy_rate',
  'days_on_market',
  'sales_volume',
] as const;

export type LocationMetric = (typeof LOCATION_METRICS)[number];

export interface MetricReading {
  value: number;
  date: string;
  source?: string | null;
}

export type LocationMetrics = Partial<Record<string, MetricReading>>;

/** Latest reading per metric for one location. */
export function latestMetrics(rows: PropertyPoint[], location: string): LocationMetrics {
  const out: LocationMetrics = {};
  for (const row of rows) {
    if (row.location !== location) continue;
    const current = out[row.metric_name];
    if (!current || row.date > current.date) {
      out[row.metric_name] = { value: row.value, date: row.date };
    }
  }
  return out;
}

/** Python truthiness: null/undefined/0 all skip the branch. */
function truthy(v: number | undefined): v is number {
  return v !== undefined && v !== 0 && !Number.isNaN(v);
}

export function locationScore(metrics: LocationMetrics): number {
  let score = 50;

  const growth = metrics.annual_growth?.value;
  if (truthy(growth) && growth > 10) score += 15;
  else if (truthy(growth) && growth > 5) score += 10;
  else if (truthy(growth) && growth > 0) score += 5;
  else if (truthy(growth) && growth < 0) score -= 10;

  const rentalYield = metrics.rental_yield?.value;
  if (truthy(rentalYield) && rentalYield > 4) score += 10;
  else if (truthy(rentalYield) && rentalYield > 3) score += 5;

  const vacancy = metrics.vacancy_rate?.value;
  if (truthy(vacancy) && vacancy < 1) score += 15;
  else if (truthy(vacancy) && vacancy < 2) score += 10;
  else if (truthy(vacancy) && vacancy > 3) score -= 10;

  const daysOnMarket = metrics.days_on_market?.value;
  if (truthy(daysOnMarket) && daysOnMarket < 30) score += 10;
  else if (truthy(daysOnMarket) && daysOnMarket > 60) score -= 5;

  return Math.max(0, Math.min(100, score));
}

export interface ScoredLocation {
  location: string;
  score: number;
  metrics: LocationMetrics;
  /** Newest reading date across all metrics — the location's freshness. */
  latestDate: string | null;
}

export function scoreAllLocations(rows: PropertyPoint[]): ScoredLocation[] {
  const locations = [...new Set(rows.map((r) => r.location))].sort();
  return locations
    .map((location) => {
      const metrics = latestMetrics(rows, location);
      const dates = Object.values(metrics)
        .map((m) => m?.date)
        .filter((d): d is string => Boolean(d))
        .sort();
      return {
        location,
        score: locationScore(metrics),
        metrics,
        latestDate: dates.length ? dates[dates.length - 1] : null,
      };
    })
    .sort((a, b) => b.score - a.score || a.location.localeCompare(b.location));
}

export const METRIC_LABELS: Record<string, string> = {
  median_price: 'Median Price',
  annual_growth: '12-Month Growth',
  rental_yield: 'Rental Yield',
  vacancy_rate: 'Vacancy Rate',
  days_on_market: 'Days on Market',
  sales_volume: 'Sales Volume',
};

export function formatMetric(metric: string, value: number | undefined): string {
  if (value === undefined) return 'No data';
  switch (metric) {
    case 'median_price':
      return `$${(value / 1000).toFixed(0)}K`;
    case 'annual_growth':
      return `${value >= 0 ? '+' : ''}${value.toFixed(1)}%`;
    case 'rental_yield':
    case 'vacancy_rate':
      return `${value.toFixed(1)}%`;
    case 'days_on_market':
      return value.toFixed(0);
    case 'sales_volume':
      return value.toFixed(0);
    default:
      return String(value);
  }
}

/** Higher is better? Used to colour the better value in a head-to-head.
 *  A metric absent from this map is NOT comparable across locations and
 *  must never get a winner highlight — sales_volume is the case in point:
 *  the readings mix geographic scope and period (Wodonga 446 and Sale 379
 *  against Sydney 13 and Tamworth 4), so "higher" is meaningless. */
export const METRIC_HIGHER_IS_BETTER: Record<string, boolean> = {
  median_price: false, // cheaper entry is "better" for a buyer
  annual_growth: true,
  rental_yield: true,
  vacancy_rate: false,
  days_on_market: false,
};

export function isComparableMetric(metric: string): boolean {
  return metric in METRIC_HIGHER_IS_BETTER;
}
