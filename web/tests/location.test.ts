import { describe, it, expect } from 'vitest';
import { locationScore, latestMetrics, scoreAllLocations } from '../src/lib/scoring/location';
import type { PropertyPoint } from '../src/lib/scoring/engine';

const m = (
  annual_growth?: number,
  rental_yield?: number,
  vacancy_rate?: number,
  days_on_market?: number,
) => {
  const out: Record<string, { value: number; date: string }> = {};
  if (annual_growth !== undefined) out.annual_growth = { value: annual_growth, date: '2026-08-01' };
  if (rental_yield !== undefined) out.rental_yield = { value: rental_yield, date: '2026-08-01' };
  if (vacancy_rate !== undefined) out.vacancy_rate = { value: vacancy_rate, date: '2026-08-01' };
  if (days_on_market !== undefined)
    out.days_on_market = { value: days_on_market, date: '2026-08-01' };
  return out;
};

describe('locationScore', () => {
  it('scores real August 2026 readings the same as the Streamlit formula', () => {
    // Adelaide: +15 growth>10, +5 yield>3, +10 vacancy<2, +10 dom<30
    expect(locationScore(m(10.5, 3.5, 1.2, 24))).toBe(90);
    // Tamworth: +15, +10 yield>4, +15 vacancy<1, +10 -> clamped at 100
    expect(locationScore(m(21.5, 4.3, 0.8, 22))).toBe(100);
    // Sydney: +5 growth>0, yield 2.8 no bonus, +10 vacancy<2, dom 42 no change
    expect(locationScore(m(2.3, 2.8, 1.5, 42))).toBe(65);
    // Sale VIC: +10 growth>5, +10 yield>4, vacancy 2.5 no bonus, -5 dom>60
    expect(locationScore(m(8.3, 4.33, 2.5, 75))).toBe(65);
  });

  it('penalises negative growth and high vacancy', () => {
    // -10 growth<0, -10 vacancy>3
    expect(locationScore(m(-6.07, 2.91, 3.15, 40))).toBe(30);
  });

  it('clamps to 0..100', () => {
    expect(locationScore(m(50, 6, 0.1, 10))).toBe(100);
    // 50 -10 (growth<0) -10 (vacancy>3) -5 (dom>60), yield 1 earns nothing
    expect(locationScore(m(-40, 1, 9, 200))).toBe(25);
  });

  it('treats a 0.00 vacancy reading as a real value, like the divergence path', () => {
    // Deliberate divergence from the Python, whose `if vacancy and ...`
    // idiom accidentally skipped zero. 0 and 0.5 are both under 1, so both
    // earn +15 — and this now matches calculateRegionalDivergence, which
    // has always used an explicit presence check.
    expect(locationScore(m(9.57, 3.69, 0, 56))).toBe(80);
    expect(locationScore(m(9.57, 3.69, 0.5, 56))).toBe(80);
  });

  it('still skips a metric that is absent rather than zero', () => {
    // no vacancy_rate key at all -> no bonus, no penalty
    expect(locationScore(m(9.57, 3.69, undefined, 56))).toBe(65);
  });

  it('returns the neutral base when no metrics are present', () => {
    expect(locationScore({})).toBe(50);
  });
});

describe('latestMetrics', () => {
  const rows: PropertyPoint[] = [
    { location: 'Perth WA', metric_name: 'median_price', date: '2026-02-16', value: 1319750 },
    { location: 'Perth WA', metric_name: 'median_price', date: '2026-08-01', value: 1050000 },
    { location: 'Perth WA', metric_name: 'vacancy_rate', date: '2026-08-01', value: 1.5 },
    { location: 'Sydney NSW', metric_name: 'median_price', date: '2026-08-01', value: 1270000 },
  ];

  it('takes the newest reading per metric for one location only', () => {
    const perth = latestMetrics(rows, 'Perth WA');
    expect(perth.median_price).toEqual({ value: 1050000, date: '2026-08-01' });
    expect(perth.vacancy_rate?.value).toBe(1.5);
    expect(Object.keys(perth)).toHaveLength(2);
  });

  it('ranks locations by score and reports freshness', () => {
    const scored = scoreAllLocations(rows);
    expect(scored.map((s) => s.location)).toContain('Perth WA');
    expect(scored.find((s) => s.location === 'Perth WA')?.latestDate).toBe('2026-08-01');
  });
});
