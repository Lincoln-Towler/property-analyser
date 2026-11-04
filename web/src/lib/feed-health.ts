// Pure logic behind the "Automation health" panel on /audit. Nothing checks the
// feeds by hand any more, and they once died unnoticed for months, so the site
// has to say when one stops. Display only: nothing here touches the score.

import { INDICATORS_CONFIG, type Frequency } from './scoring/config';
import type { PropertyPoint, SeriesMap } from './scoring/engine';
import type { BtcData } from './btc';

/** How old a reading can get before its feed counts as late. Monthly and
 *  quarterly values are dated the first day of the period and published weeks
 *  later, so these limits include that lag. */
export const MAX_AGE_DAYS: Record<Frequency, number> = { weekly: 14, monthly: 75, quarterly: 200 };

/** Scored indicators with no feed. They're entered by hand, so a late one
 *  needs a manual update rather than a workflow fix. */
export const MANUAL_INDICATORS = new Set(['household_debt_gdp', 'mortgage_stress_rate']);

export interface JobHealth {
  name: string;
  schedule: string;
  latest: string | null;
  ageDays: number | null;
  ok: boolean;
}

export interface LateIndicator {
  indicator: string;
  displayName: string;
  latest: string | null;
  ageDays: number | null;
  manual: boolean;
}

export interface FeedHealth {
  jobs: JobHealth[];
  late: LateIndicator[];
  failedVideos: { title: string; attempts: number }[];
  failedChanges: { flag_key: string; note: string | null }[];
  healthy: boolean;
}

function ageInDays(iso: string | null | undefined, now: Date): number | null {
  if (!iso) return null;
  const then = Date.parse(iso);
  if (Number.isNaN(then)) return null;
  return Math.floor((now.getTime() - then) / 86_400_000);
}

function newest(dates: (string | null | undefined)[]): string | null {
  const known = dates.filter((d): d is string => Boolean(d)).sort();
  return known.length ? known[known.length - 1] : null;
}

function job(name: string, schedule: string, latest: string | null, limitDays: number, now: Date): JobHealth {
  const ageDays = ageInDays(latest, now);
  return { name, schedule, latest, ageDays, ok: ageDays !== null && ageDays <= limitDays };
}

export function buildFeedHealth(input: {
  series: SeriesMap;
  propertyData: PropertyPoint[];
  scoreHistory: { score_date: string }[];
  btc: Pick<BtcData, 'latest' | 'videos' | 'changes'> | null;
  now: Date;
}): FeedHealth {
  const { series, propertyData, scoreHistory, btc, now } = input;
  const latestOf = (name: string) => newest((series[name] ?? []).map((p) => p.date));

  // The cash rate is saved with the date of each run, so it shows the workflow itself ran
  const jobs: JobHealth[] = [
    job('Economic indicators', 'Daily 7:00am', latestOf('interest_rate'), 9, now),
    job('Location data (SQM)', 'Mondays 7:15am', newest(propertyData.filter((p) => p.metric_name === 'median_price').map((p) => p.date)), 21, now),
    job('Daily score snapshot', 'Daily', newest(scoreHistory.map((s) => s.score_date)), 2, now),
  ];
  if (btc) {
    jobs.push(job('Cowen videos to BTC flags', 'Daily 8:30am', newest(btc.videos.map((v) => v.published_at)), 10, now));
    jobs.push(job('BTC signal', 'Daily', btc.latest?.logged_at ?? null, 2, now));
  }

  const late: LateIndicator[] = [];
  for (const [indicator, config] of Object.entries(INDICATORS_CONFIG)) {
    if (!config.expected_frequency) continue;
    const latest = latestOf(indicator);
    const ageDays = ageInDays(latest, now);
    if (ageDays === null || ageDays > MAX_AGE_DAYS[config.expected_frequency]) {
      late.push({ indicator, displayName: config.display_name, latest, ageDays, manual: MANUAL_INDICATORS.has(indicator) });
    }
  }

  const failedVideos = (btc?.videos ?? [])
    .filter((v) => v.status === 'failed')
    .map((v) => ({ title: v.title, attempts: v.attempts }));
  const failedChanges = (btc?.changes ?? [])
    .filter((c) => c.status === 'failed')
    .map((c) => ({ flag_key: c.flag_key, note: c.apply_note }));

  return {
    jobs,
    late,
    failedVideos,
    failedChanges,
    healthy:
      jobs.every((j) => j.ok) &&
      late.every((l) => l.manual) &&
      failedVideos.length === 0 &&
      failedChanges.length === 0,
  };
}
