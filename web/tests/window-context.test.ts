import { describe, it, expect } from 'vitest';
import {
  getWindowContext,
  windowContradictsTrend,
  getIndicatorTrend,
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
