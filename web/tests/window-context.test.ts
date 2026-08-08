import { describe, it, expect } from 'vitest';
import {
  getWindowContext,
  windowContradictsTrend,
  getIndicatorTrend,
  buildAudit,
} from '../src/lib/scoring/engine';

// The real auction_clearance_rate series as of Aug 2026 — the case that
// motivated this module.
const CLEARANCE = [
  { date: '2026-05-16', value: 54.7 },
  { date: '2026-06-06', value: 50.5 },
  { date: '2026-06-13', value: 49.8 },
  { date: '2026-06-20', value: 48.5 },
  { date: '2026-07-11', value: 47.2 },
  { date: '2026-07-19', value: 45.3 },
];
const NOW = new Date('2026-08-08T12:00:00Z');

describe('getWindowContext', () => {
  it('exposes the collapse that the two-point trend rule calls "stable"', () => {
    const [trend, changePct] = getIndicatorTrend(CLEARANCE, NOW, 3);
    // the scored view: last two points only, -4% -> under the 5% gate
    expect(trend).toBe('stable');
    expect(changePct).toBeCloseTo(-4.03, 1);

    // the honest view over the same window
    const ctx = getWindowContext(CLEARANCE, NOW, 3)!;
    expect(ctx.pointsInWindow).toBe(6);
    expect(ctx.changePct).toBeCloseTo(-17.18, 1);
    expect(ctx.direction).toBe('falling');
    expect(ctx.monotoneSteps).toBe(5);
    expect(ctx.totalSteps).toBe(5);
    expect(ctx.spanDays).toBe(64);
  });

  it('flags the contradiction between window and scored trend', () => {
    const ctx = getWindowContext(CLEARANCE, NOW, 3);
    expect(windowContradictsTrend(ctx, 'stable')).toBe(true);
  });

  it('stays quiet when the window agrees with the trend', () => {
    const flat = [
      { date: '2026-06-01', value: 4.5 },
      { date: '2026-07-01', value: 4.52 },
      { date: '2026-08-01', value: 4.51 },
    ];
    const ctx = getWindowContext(flat, NOW, 3);
    expect(windowContradictsTrend(ctx, 'stable')).toBe(false);
  });

  it('returns null when there are too few points to add anything', () => {
    expect(getWindowContext(CLEARANCE.slice(-2), NOW, 3)).toBeNull();
    expect(getWindowContext([], NOW, 3)).toBeNull();
  });

  it('never affects the scored trend (parity guard)', () => {
    // getWindowContext is pure and read-only; calling it must not change
    // what getIndicatorTrend returns for the same input.
    const before = getIndicatorTrend(CLEARANCE, NOW, 3);
    getWindowContext(CLEARANCE, NOW, 3);
    expect(getIndicatorTrend(CLEARANCE, NOW, 3)).toEqual(before);
  });
});

describe('buildAudit weight-at-risk', () => {
  it('reports the weight carried by a frozen single-point indicator', () => {
    const audit = buildAudit(
      {
        household_debt_gdp: [{ date: '2025-06-30', value: 113.7 }],
        auction_clearance_rate: CLEARANCE,
      },
      NOW,
    );
    const debt = audit.find((a) => a.indicator === 'household_debt_gdp')!;
    expect(debt.weight).toBe(25);
    expect(debt.weight_pct).toBeCloseTo((25 / 135) * 100, 5); // 18.5%
    expect(debt.can_trend).toBe(false);
    expect(debt.can_measure_volatility).toBe(false);
    expect(debt.stale).toBe(true);

    // a densely sampled series is not frozen
    const clearance = audit.find((a) => a.indicator === 'auction_clearance_rate')!;
    expect(clearance.can_trend).toBe(true);
    expect(clearance.can_measure_volatility).toBe(true);
    expect(clearance.stale).toBe(false);
  });
});

describe('cadence vs engine windows', () => {
  it('shows a complete quarterly backfill still cannot trend or measure volatility', () => {
    // 3 full years of quarterly readings, ending this quarter.
    const quarterly = [
      '2023-09-30','2023-12-31','2024-03-31','2024-06-30','2024-09-30','2024-12-31',
      '2025-03-31','2025-06-30','2025-09-30','2025-12-31','2026-03-31','2026-06-30',
    ].map((date, i) => ({ date, value: 110 + i * 0.4 }));

    const audit = buildAudit({ household_debt_gdp: quarterly }, NOW);
    const debt = audit.find((a) => a.indicator === 'household_debt_gdp')!;

    expect(debt.point_count).toBe(12);
    expect(debt.stale).toBe(false);
    // ...and yet:
    expect(debt.can_trend).toBe(false);
    expect(debt.can_measure_volatility).toBe(false);
    // because the cadence itself makes those windows unreachable
    expect(debt.cadence_blocks_trend).toBe(true);
    expect(debt.cadence_blocks_volatility).toBe(true);
  });

  it('does not blame cadence for a weekly series that simply lacks data', () => {
    const audit = buildAudit({ auction_clearance_rate: CLEARANCE }, NOW);
    const clearance = audit.find((a) => a.indicator === 'auction_clearance_rate')!;
    expect(clearance.cadence_blocks_trend).toBe(false);
    expect(clearance.cadence_blocks_volatility).toBe(false);
  });
});
