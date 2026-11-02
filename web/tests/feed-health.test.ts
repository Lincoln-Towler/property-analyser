import { describe, it, expect } from 'vitest';
import { buildFeedHealth } from '../src/lib/feed-health';
import type { SeriesMap } from '../src/lib/scoring/engine';

const NOW = new Date('2026-11-02T12:00:00Z');

// Everything on schedule: dated the way the feeds date them today
const pt = (date: string) => ({ date, value: 1, source: 'history' });
const series: SeriesMap = {
  interest_rate: [pt('2026-11-02')],
  auction_clearance_rate: [pt('2026-10-31')],
  rental_vacancy_rate: [pt('2026-09-01')],
  building_approvals: [pt('2026-09-01')],
  unemployment_rate: [pt('2026-09-01')],
  credit_growth: [pt('2026-09-01')],
  mortgage_stress_rate: [pt('2026-09-01')],
  wage_growth: [pt('2026-06-01')],
  household_debt_gdp: [pt('2026-06-01')],
};
const propertyData = [{ date: '2026-10-23', location: 'Perth WA', metric_name: 'median_price', value: 1 }];
const scoreHistory = [{ score_date: '2026-11-01' }];
const log = { logged_at: '2026-11-01T19:31:08+00:00' } as NonNullable<Parameters<typeof buildFeedHealth>[0]['btc']>['latest'];
const video = { video_id: 'a', title: 'A', published_at: '2026-11-01T10:00:00+00:00', url: null, summary: null, stance: null, key_levels: null, status: 'analysed', attempts: 1 };
const btc = { latest: log, videos: [video], changes: [] };
const base = { series, propertyData, scoreHistory, btc, now: NOW };

describe('buildFeedHealth', () => {
  it('is healthy when every feed is on schedule, allowing for publication lag', () => {
    const h = buildFeedHealth(base);
    expect(h.jobs.map((j) => j.ok)).toEqual([true, true, true, true, true]);
    expect(h.late).toEqual([]);
    expect(h.healthy).toBe(true);
  });

  it('flags a workflow that stopped running', () => {
    const h = buildFeedHealth({ ...base, series: { ...series, interest_rate: [pt('2026-10-12')] } });
    expect(h.jobs.find((j) => j.name === 'Economic indicators')?.ok).toBe(false);
    expect(h.healthy).toBe(false);
  });

  it('flags the BTC signal after three days without a reading', () => {
    const stale = { ...log!, logged_at: '2026-10-29T19:31:08+00:00' };
    const h = buildFeedHealth({ ...base, btc: { ...btc, latest: stale } });
    expect(h.jobs.find((j) => j.name === 'BTC signal')?.ok).toBe(false);
  });

  it('a late automatic feed is a fault; a late manual indicator is only a reminder', () => {
    const feed = buildFeedHealth({ ...base, series: { ...series, credit_growth: [pt('2026-07-01')] } });
    expect(feed.late.map((l) => l.indicator)).toEqual(['credit_growth']);
    expect(feed.late[0].manual).toBe(false);
    expect(feed.healthy).toBe(false);

    const manual = buildFeedHealth({ ...base, series: { ...series, household_debt_gdp: [pt('2025-12-01')] } });
    expect(manual.late.map((l) => l.indicator)).toEqual(['household_debt_gdp']);
    expect(manual.late[0].manual).toBe(true);
    expect(manual.healthy).toBe(true);
  });

  it('counts an indicator with no data at all as late', () => {
    const { wage_growth: _gone, ...rest } = series;
    const h = buildFeedHealth({ ...base, series: rest });
    expect(h.late.find((l) => l.indicator === 'wage_growth')?.ageDays).toBeNull();
  });

  it('reports failed videos and flag changes that did not apply', () => {
    const h = buildFeedHealth({
      ...base,
      btc: {
        ...btc,
        videos: [video, { ...video, video_id: 'b', title: 'B', status: 'failed', attempts: 3 }],
        changes: [{ action: 'update', flag_key: 'x', flag_label: null, score_modifier: null, reason: null, status: 'failed', apply_note: 'No flag called x', created_at: '2026-11-02' }],
      },
    });
    expect(h.failedVideos).toEqual([{ title: 'B', attempts: 3 }]);
    expect(h.failedChanges).toEqual([{ flag_key: 'x', note: 'No flag called x' }]);
    expect(h.healthy).toBe(false);
  });

  it('still works when the BTC data is unavailable', () => {
    const h = buildFeedHealth({ ...base, btc: null });
    expect(h.jobs).toHaveLength(3);
    expect(h.healthy).toBe(true);
  });
});
