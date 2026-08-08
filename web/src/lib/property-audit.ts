// Read-only quality checks over property_data. Nothing here alters or
// excludes a row — every reading stays in the database and in the charts.
//
// This is the outcome of the Aug 2026 outlier audit: an adversarial review
// rejected every proposed deletion. The values that look extreme are
// defensible once you account for the fact that "locations" here mix
// capitals, suburbs and regional towns at different geographic scopes —
// e.g. Sydney's Feb snapshot carries sales_volume 13 and days_on_market
// 106, which is a sub-market sample, not a city-wide figure. So the job is
// to make the oddities visible, not to remove them.

import type { PropertyPoint } from './scoring/engine';

export interface PropertyFlag {
  location: string;
  metric: string;
  date: string;
  value: number;
  note: string;
}

export interface DuplicateSnapshot {
  location: string;
  metric: string;
  value: number;
  dates: string[];
}

export interface PropertyAudit {
  flags: PropertyFlag[];
  duplicates: DuplicateSnapshot[];
  /** metrics that have no reading in the newest snapshot date */
  missingFromLatest: string[];
  latestDate: string | null;
  locations: number;
}

/** Bounds chosen to catch "look at this", not to define truth. Deliberately
 *  wider than any single market's normal range. */
const ATTENTION_BOUNDS: Record<string, { min: number; max: number; unit: string }> = {
  vacancy_rate: { min: 0.05, max: 5, unit: '%' },
  annual_growth: { min: -25, max: 25, unit: '%' },
  rental_yield: { min: 1, max: 8, unit: '%' },
  days_on_market: { min: 5, max: 120, unit: ' days' },
};

const DUPLICATE_WINDOW_DAYS = 21;

function daysBetween(a: string, b: string): number {
  return Math.abs(
    (new Date(a + 'T00:00:00Z').getTime() - new Date(b + 'T00:00:00Z').getTime()) / 86400000,
  );
}

export function auditPropertyData(rows: PropertyPoint[]): PropertyAudit {
  const flags: PropertyFlag[] = [];

  for (const row of rows) {
    const bounds = ATTENTION_BOUNDS[row.metric_name];
    if (!bounds) continue;
    if (row.value < bounds.min) {
      flags.push({
        location: row.location,
        metric: row.metric_name,
        date: row.date,
        value: row.value,
        note:
          row.value === 0
            ? 'exactly 0 — may be a genuine thin-market reading or a failed scrape; check the source'
            : `below ${bounds.min}${bounds.unit}`,
      });
    } else if (row.value > bounds.max) {
      flags.push({
        location: row.location,
        metric: row.metric_name,
        date: row.date,
        value: row.value,
        note: `above ${bounds.max}${bounds.unit} — plausible for a narrow sub-market sample, not for a whole city`,
      });
    }
  }

  // Same location+metric+value recorded twice within a few weeks = re-import
  const byKey = new Map<string, PropertyPoint[]>();
  for (const row of rows) {
    const key = `${row.location}|${row.metric_name}|${row.value}`;
    const list = byKey.get(key) ?? [];
    list.push(row);
    byKey.set(key, list);
  }
  const duplicates: DuplicateSnapshot[] = [];
  for (const [key, list] of byKey) {
    if (list.length < 2) continue;
    const dates = [...new Set(list.map((r) => r.date))].sort();
    if (dates.length < 2) continue;
    if (daysBetween(dates[0], dates[dates.length - 1]) > DUPLICATE_WINDOW_DAYS) continue;
    const [location, metric] = key.split('|');
    duplicates.push({ location, metric, value: list[0].value, dates });
  }
  duplicates.sort((a, b) => a.location.localeCompare(b.location) || a.metric.localeCompare(b.metric));

  const allDates = [...new Set(rows.map((r) => r.date))].sort();
  const latestDate = allDates.length ? allDates[allDates.length - 1] : null;
  const metricsAll = [...new Set(rows.map((r) => r.metric_name))].sort();
  const metricsLatest = new Set(rows.filter((r) => r.date === latestDate).map((r) => r.metric_name));
  const missingFromLatest = metricsAll.filter((m) => !metricsLatest.has(m));

  flags.sort(
    (a, b) => a.location.localeCompare(b.location) || a.metric.localeCompare(b.metric),
  );

  return {
    flags,
    duplicates,
    missingFromLatest,
    latestDate,
    locations: new Set(rows.map((r) => r.location)).size,
  };
}
