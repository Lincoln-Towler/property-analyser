// The data layer. One query per table per revalidation window — replacing
// the old app's ~40 sequential round-trips per render. Pages call getSiteData()
// (deduped per request via React.cache) and derive everything in memory.

import 'server-only';
import { cache } from 'react';
import { createClient } from '@supabase/supabase-js';
import type { SeriesMap, PropertyPoint } from './scoring/engine';
import { prepareSeries, type SeriesAdjustments } from './transforms';

export interface ScoreHistoryRow {
  score_date: string;
  final_score: number;
  signal: string;
}

export interface SiteData {
  series: SeriesMap;
  propertyData: PropertyPoint[];
  scoreHistory: ScoreHistoryRow[];
  adjustments: SeriesAdjustments;
  fetchedAt: string;
  /** Query failures for the optional tables. An empty result and a denied
   *  query look identical downstream, so record the difference: "no rows"
   *  and "you cannot read this table" need very different fixes. */
  errors: { propertyData?: string; scoreHistory?: string };
}

function monthsAgoISO(months: number): string {
  const d = new Date();
  d.setUTCMonth(d.getUTCMonth() - months);
  return d.toISOString().slice(0, 10);
}

/** Returns null when Supabase env vars are absent (e.g. CI builds). */
export const getSiteData = cache(async (): Promise<SiteData | null> => {
  const url = process.env.NEXT_PUBLIC_SUPABASE_URL;
  const anonKey = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY;
  if (!url || !anonKey) return null;

  const supabase = createClient(url, anonKey, { auth: { persistSession: false } });

  const [indicators, property, history] = await Promise.all([
    supabase
      .from('economic_indicators_combined')
      .select('date, indicator_name, value, source')
      .gte('date', monthsAgoISO(24))
      .order('date', { ascending: true }),
    supabase
      .from('property_data')
      .select('date, location, metric_name, value')
      .gte('date', monthsAgoISO(12))
      .order('date', { ascending: true }),
    supabase
      .from('market_score_history')
      .select('score_date, final_score, signal')
      .order('score_date', { ascending: false })
      .limit(180),
  ]);

  if (indicators.error) throw new Error(`indicators query failed: ${indicators.error.message}`);

  // property_data and market_score_history are optional — a missing table
  // must not take the whole site down. But record WHY they are empty: a
  // denied read (RLS) and a genuinely empty table are indistinguishable
  // once the rows are gone, and they need opposite fixes.
  const errors: SiteData['errors'] = {};
  if (property.error) errors.propertyData = property.error.message;
  if (history.error) errors.scoreHistory = history.error.message;

  const rawSeries: SeriesMap = {};
  for (const row of indicators.data ?? []) {
    (rawSeries[row.indicator_name] ??= []).push({
      date: row.date,
      value: Number(row.value),
      source: row.source,
    });
  }
  // Drop implausible values and annualise monthly building approvals before
  // anything downstream (engine, charts) sees the data.
  const { series, adjustments } = prepareSeries(rawSeries);

  const propertyData: PropertyPoint[] = (property.data ?? []).map((row) => ({
    date: row.date,
    location: row.location,
    metric_name: row.metric_name,
    value: Number(row.value),
  }));

  const scoreHistory: ScoreHistoryRow[] = (history.data ?? [])
    .map((r) => ({ ...r, final_score: Number(r.final_score) }))
    .reverse();

  return {
    series,
    propertyData,
    scoreHistory,
    adjustments,
    errors,
    fetchedAt: new Date().toISOString(),
  };
});
