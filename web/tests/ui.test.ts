import { describe, it, expect } from 'vitest';
import {
  fmtDate,
  fmtDateTimeUtc,
  fmtIndicator,
  fmtMonthYear,
  fmtPriceCompact,
  niceScale,
  stripLeadingEmoji,
  timeTicks,
} from '../src/lib/format';
import { SIGNAL_BANDS, signalBand } from '../src/lib/theme';
import { signalForScore } from '../src/lib/scoring/engine';
import { INDICATORS_CONFIG } from '../src/lib/scoring/config';
import { indicatorTargetText, statusPhrase, targetBand, warningLine } from '../src/lib/scoring/ui-helpers';
import { groupByPublisher, sourcePublisher } from '../src/lib/sources';
import {
  BTC_MODIFIER_CAP,
  btcSignalForScore,
  buildLadder,
  capModifiers,
  changeName,
  effectiveScore,
  groupFlags,
  levelSignalClass,
  scoreCeiling,
  stanceClass,
} from '../src/lib/btc-view';
import { andersonState } from '../src/lib/scoring/anderson';
import { METRIC_LABELS } from '../src/lib/scoring/location';
import type { BtcContextFlag, BtcFlagChange, BtcKeyLevel } from '../src/lib/btc';

describe('format', () => {
  it('formats ISO dates without time-zone drift', () => {
    expect(fmtDate('2026-11-01')).toBe('1 Nov 2026');
    expect(fmtDate('2026-08-01T19:31:08.966504+00:00')).toBe('1 Aug 2026');
    expect(fmtDate(null)).toBe('—');
    expect(fmtMonthYear('2026-03-24')).toBe("Mar '26");
    expect(fmtDateTimeUtc('2026-11-02T02:53:11.000Z')).toBe('2 Nov 2026, 02:53 UTC');
  });

  it('formats property prices compactly', () => {
    expect(fmtPriceCompact(1150000)).toBe('$1.15M');
    expect(fmtPriceCompact(1752000)).toBe('$1.75M');
    expect(fmtPriceCompact(1000000)).toBe('$1M');
    expect(fmtPriceCompact(1050000)).toBe('$1.05M');
    expect(fmtPriceCompact(955000)).toBe('$955K');
  });

  it('formats indicator readings with units', () => {
    expect(fmtIndicator(202729)).toBe('202,729');
    expect(fmtIndicator(4.6, '%')).toBe('4.6%');
    expect(fmtIndicator(1.3900000001, '%')).toBe('1.39%');
    expect(fmtIndicator(null)).toBe('—');
  });

  it('strips the engine emoji from signal strings', () => {
    expect(stripLeadingEmoji('🟡 Hold')).toBe('Hold');
    expect(stripLeadingEmoji('⚠️ Late Cycle (Year 18/18.6) - Exercise Caution')).toBe(
      'Late Cycle (Year 18/18.6) - Exercise Caution',
    );
    expect(stripLeadingEmoji('Hold')).toBe('Hold');
  });

  it('builds a round y-axis that keeps a target edge in view', () => {
    // household debt: readings 113.4–114, target "below 100"
    const debt = niceScale([113.4, 113.7, 114, 100]);
    expect(debt.domain[0]).toBeLessThanOrEqual(100);
    expect(debt.domain[1]).toBeGreaterThanOrEqual(114);
    expect(debt.ticks).toEqual([100, 105, 110, 115]);
    const vacancy = niceScale([1.04, 1.39, 2]);
    expect(vacancy.ticks[0]).toBeLessThanOrEqual(1.04);
    expect(vacancy.ticks[vacancy.ticks.length - 1]).toBeGreaterThanOrEqual(2);
  });

  it('puts time ticks on month boundaries, within range, capped in count', () => {
    const min = Date.UTC(2026, 1, 21); // 21 Feb 2026
    const max = Date.UTC(2026, 10, 1); // 1 Nov 2026
    const ticks = timeTicks(min, max, 5);
    expect(ticks.length).toBeGreaterThan(1);
    expect(ticks.length).toBeLessThanOrEqual(5);
    for (const t of ticks) {
      expect(t).toBeGreaterThanOrEqual(min);
      expect(t).toBeLessThanOrEqual(max);
      expect(new Date(t).getUTCDate()).toBe(1);
    }
  });
});

describe('signal bands', () => {
  it('match the engine thresholds exactly', () => {
    for (const band of SIGNAL_BANDS) {
      for (const score of [band.from, (band.from + band.to) / 2, band.to - 0.01]) {
        expect(stripLeadingEmoji(signalForScore(score))).toBe(band.label);
        expect(signalBand(score).label).toBe(band.label);
      }
    }
  });
});

describe('chart thresholds', () => {
  it('labels the mortgage stress line with the value it is drawn at', () => {
    const cfg = INDICATORS_CONFIG.mortgage_stress_rate;
    expect(warningLine(cfg)).toEqual({ value: 30, label: 'warning >30%' });
    expect(targetBand(cfg)).toEqual([null, 25]);
  });

  it('derives bands and lines for direct indicators', () => {
    expect(targetBand(INDICATORS_CONFIG.auction_clearance_rate)).toEqual([65, null]);
    expect(warningLine(INDICATORS_CONFIG.auction_clearance_rate)).toEqual({ value: 55, label: 'weak <55%' });
    expect(targetBand(INDICATORS_CONFIG.credit_growth)).toEqual([0.3, 0.8]);
    expect(warningLine(INDICATORS_CONFIG.building_approvals)?.label).toBe('deficit <180,000');
    expect(targetBand(INDICATORS_CONFIG.interest_rate)).toEqual([2.5, 4]);
  });

  it('shades a band for every indicator that states a target', () => {
    for (const cfg of Object.values(INDICATORS_CONFIG)) {
      expect(targetBand(cfg) !== null).toBe(indicatorTargetText(cfg) !== null);
    }
  });

  it('turns a classifier note into a status phrase', () => {
    expect(statusPhrase('above warning (114.00%, >110%)')).toBe('Above warning');
    expect(statusPhrase('in optimal zone (1.39%)')).toBe('In optimal zone');
  });
});

describe('source publishers', () => {
  it('rolls free-text labels up to the publisher', () => {
    expect(sourcePublisher('history')).toBe('n8n feed');
    expect(sourcePublisher('Roy Morgan Mortgage Stress Jun 2026')).toBe('Roy Morgan');
    expect(sourcePublisher('Cotality final, combined capitals wk ending 9 Aug 2026')).toBe('Cotality');
    expect(sourcePublisher('BIS via Trading Economics Q3 2025')).toBe('BIS');
    expect(sourcePublisher('ABS Labour Force Jul 2026 (SA)')).toBe('ABS');
    expect(sourcePublisher('Mayfield Analytics Oct 2026')).toBe('Mayfield Analytics');
  });

  it('sums counts per publisher', () => {
    expect(
      groupByPublisher({
        'Roy Morgan Mortgage Stress Jun 2026': 1,
        'Roy Morgan Mortgage Stress Jul 2026': 1,
        RBA: 1,
        history: 3,
      }),
    ).toEqual([
      ['n8n feed', 3],
      ['Roy Morgan', 2],
      ['RBA', 1],
    ]);
  });
});

describe('btc view', () => {
  const level = (level_usd: number, base_score: number, label = `L${level_usd}`): BtcKeyLevel => ({
    level_usd,
    label,
    signal: 'BUY',
    base_score,
    emoji: null,
    source: null,
    notes: null,
  });
  const flag = (flag_key: string, category: string, score_modifier: number, is_active = true): BtcContextFlag => ({
    flag_key,
    flag_label: flag_key,
    category,
    score_modifier,
    current_value: null,
    is_active,
    updated_at: '2026-09-18',
  });

  it('uses get_btc_signal() thresholds', () => {
    expect(btcSignalForScore(9)).toBe('MAX BUY');
    expect(btcSignalForScore(4)).toBe('HOLD');
    expect(btcSignalForScore(2.49)).toBe('SELL / TAKE PROFIT');
  });

  it('caps the flags at ±2 like get_btc_signal()', () => {
    expect(BTC_MODIFIER_CAP).toBe(2);
    expect(capModifiers(-6.7)).toBe(-2);
    expect(capModifiers(2.4)).toBe(2);
    expect(capModifiers(-1.3)).toBe(-1.3);
    // $77.7K sits in the $75K band (base 3.5): 3.5 - 2 = 1.5, not 3.5 - 6.7
    expect(effectiveScore(3.5, -6.7)).toBe(1.5);
    expect(effectiveScore(2, -6.7)).toBe(1); // clamped at 1
    expect(effectiveScore(9.8, 3)).toBe(10); // clamped at 10
  });

  it('computes the score ceiling with the capped flag total', () => {
    // Before the cap, -7.5 left even the best band at 2.3 (SELL at every price)
    expect(scoreCeiling([level(31750, 9.8), level(83000, 2)], -7.5)).toEqual({ score: 7.8, signal: 'STRONG BUY' });
    expect(scoreCeiling([level(31750, 9.8)], 0)).toEqual({ score: 9.8, signal: 'MAX BUY' });
  });

  it('colours chips for both stored labels and signal text', () => {
    expect(levelSignalClass('CAUTION / REDUCE')).toContain('orange');
    expect(levelSignalClass('SELL / TAKE PROFIT')).toBe(levelSignalClass('SELL'));
    expect(levelSignalClass('LIGHT BUY')).toBe(levelSignalClass('LIGHT_BUY'));
  });

  it('slots the current price into the ladder', () => {
    const rows = buildLadder([level(75000, 3.5, 'Caution'), level(83000, 2), level(69000, 5.5)], 77705, 'Caution');
    expect(rows.map((r) => (r.kind === 'price' ? 'price' : r.level.level_usd))).toEqual([83000, 'price', 75000, 69000]);
    const caution = rows.find((r) => r.kind === 'level' && r.level.level_usd === 75000);
    expect(caution && caution.kind === 'level' && caution.isCurrentBand).toBe(true);
    expect(caution && caution.kind === 'level' && caution.distancePct).toBe(-3.5);
  });

  it('names logged flag changes even when the row has no label', () => {
    const change = (flag_key: string, flag_label: string | null): BtcFlagChange => ({
      action: 'retire', flag_key, flag_label, score_modifier: null, reason: null, status: 'applied', apply_note: null, created_at: '2026-11-02',
    });
    const flags = [{ ...flag('known_flag', 'MACRO', -0.2), flag_label: 'A known flag' }];
    expect(changeName(change('x', 'Own label'), flags)).toBe('Own label');
    expect(changeName(change('known_flag', null), flags)).toBe('A known flag');
    expect(changeName(change('gone_flag', null), flags)).toBe('gone flag');
  });

  it('colours stance chips, and anything unexpected stays neutral', () => {
    expect(stanceClass('bearish')).toContain('red');
    expect(stanceClass('bullish')).toContain('emerald');
    expect(stanceClass('mixed')).toBe(stanceClass('neutral'));
  });

  it('groups active flags by category with signed sums', () => {
    const groups = groupFlags([
      flag('a', 'MACRO', -0.5),
      flag('b', 'MACRO', 0.3),
      flag('c', 'CYCLE', -0.2),
      flag('d', 'MACRO', -0.4, false),
    ]);
    expect(groups.map((g) => g.category)).toEqual(['MACRO', 'CYCLE']);
    expect(groups[0]).toMatchObject({ net: -0.2, bearishSum: -0.5, bullishSum: 0.3, bearishCount: 1, bullishCount: 1 });
  });
});

describe('anderson advice', () => {
  it('does not say "do not buy" inside the predicted best-buying window', () => {
    const recs = andersonState(new Date(Date.UTC(2026, 10, 2))).recommendations.join(' ');
    expect(recs).not.toMatch(/DO NOT BUY/);
    expect(recs).toMatch(/BEST BUYING WINDOW/);
  });

  it('still says "do not buy" before the window opens', () => {
    const recs = andersonState(new Date(Date.UTC(2024, 5, 1))).recommendations.join(' ');
    expect(recs).toMatch(/DO NOT BUY/);
    expect(recs).not.toMatch(/🟢 \*\*HOLD CASH/);
  });
});

describe('location labels', () => {
  it('calls the median_price metric what it now holds: an asking price', () => {
    expect(METRIC_LABELS.median_price).toBe('Asking Price (3-bed)');
  });
});
