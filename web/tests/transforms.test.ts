import { describe, it, expect } from 'vitest';
import { prepareSeries, annualizeMonthly, looksMonthly } from '../src/lib/transforms';

describe('plausibility filter', () => {
  it('drops the broken-scraper zero pattern that hit interest_rate in July 2026', () => {
    const { series, adjustments } = prepareSeries({
      interest_rate: [
        { date: '2026-05-03', value: 4.1 },
        { date: '2026-07-16', value: 0 },
        { date: '2026-07-17', value: 4.35 },
        { date: '2026-07-19', value: 0 },
      ],
    });
    expect(series.interest_rate.map((p) => p.value)).toEqual([4.1, 4.35]);
    expect(adjustments.dropped).toEqual([
      { indicator: 'interest_rate', date: '2026-07-16', value: 0 },
      { indicator: 'interest_rate', date: '2026-07-19', value: 0 },
    ]);
  });

  it('keeps legitimate zero/negative values for unranged indicators', () => {
    const { series, adjustments } = prepareSeries({
      credit_growth: [{ date: '2026-06-01', value: 0 }],
      wage_growth: [{ date: '2026-06-01', value: -0.5 }],
    });
    expect(series.credit_growth).toHaveLength(1);
    expect(series.wage_growth).toHaveLength(1);
    expect(adjustments.dropped).toHaveLength(0);
  });
});

describe('building approvals annualisation', () => {
  const monthly = [
    { date: '2025-12-01', value: 14883 },
    { date: '2026-01-01', value: 10468 },
    { date: '2026-02-01', value: 18327 },
    { date: '2026-03-01', value: 17780 },
    { date: '2026-04-01', value: 17349 },
    { date: '2026-05-01', value: 17019 },
    { date: '2026-06-30', value: 18328 },
  ];

  it('detects monthly-scale data', () => {
    expect(looksMonthly(monthly)).toBe(true);
    expect(looksMonthly([{ date: '2026-01-01', value: 172000 }])).toBe(false);
  });

  it('annualises from the trailing 12-month window and drops partial windows', () => {
    const out = annualizeMonthly(monthly);
    // 7 readings, minimum window 6 -> only the 6th and 7th points survive.
    // Without this guard the earliest point annualised a single month
    // (14883*12 = 178596) and the widening window drew a fake uptrend.
    expect(out).toHaveLength(2);
    expect(out.map((p) => p.date)).toEqual(['2026-05-01', '2026-06-30']);
    // last point: mean of all 7 readings * 12
    const mean = monthly.reduce((a, p) => a + p.value, 0) / monthly.length;
    expect(out[out.length - 1].value).toBe(Math.round(mean * 12));
    // annualised values land in the annual-threshold regime
    expect(out[out.length - 1].value).toBeGreaterThan(150000);
    expect(out[out.length - 1].value).toBeLessThan(260000);
  });

  it('emits nothing when there are too few readings to annualise honestly', () => {
    expect(annualizeMonthly(monthly.slice(0, 3))).toEqual([]);
  });

  it('applies via prepareSeries only when the series looks monthly', () => {
    const annualScale = { building_approvals: [{ date: '2026-01-01', value: 172000 }] };
    expect(prepareSeries(annualScale).adjustments.annualized).toEqual([]);
    const monthlyScale = { building_approvals: monthly };
    const prepared = prepareSeries(monthlyScale);
    expect(prepared.adjustments.annualized).toEqual(['building_approvals']);
    // first surviving point is the 6th reading (first full-enough window)
    expect(prepared.series.building_approvals[0].date).toBe('2026-05-01');
  });
});
