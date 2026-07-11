// Daily score snapshot: computes today's v3 score and upserts one row into
// market_score_history. Triggered by Vercel Cron (see vercel.json) with
// `Authorization: Bearer ${CRON_SECRET}` — Vercel sets this automatically
// for cron invocations when the CRON_SECRET env var exists.

import { NextRequest, NextResponse } from 'next/server';
import { createClient } from '@supabase/supabase-js';
import { calculateMarketScoreV3, buildAudit } from '@/lib/scoring/engine';
import type { SeriesMap, PropertyPoint } from '@/lib/scoring/engine';
import { generateAutoCommentary } from '@/lib/scoring/commentary';

export const dynamic = 'force-dynamic';

export async function GET(request: NextRequest) {
  const secret = process.env.CRON_SECRET;
  if (!secret || request.headers.get('authorization') !== `Bearer ${secret}`) {
    return NextResponse.json({ error: 'unauthorized' }, { status: 401 });
  }

  const url = process.env.NEXT_PUBLIC_SUPABASE_URL;
  const serviceKey = process.env.SUPABASE_SERVICE_ROLE_KEY;
  if (!url || !serviceKey) {
    return NextResponse.json({ error: 'supabase env vars missing' }, { status: 500 });
  }

  const supabase = createClient(url, serviceKey, { auth: { persistSession: false } });
  const now = new Date();
  const cutoff = new Date(now);
  cutoff.setUTCMonth(cutoff.getUTCMonth() - 24);

  const [indicators, property] = await Promise.all([
    supabase
      .from('economic_indicators_combined')
      .select('date, indicator_name, value, source')
      .gte('date', cutoff.toISOString().slice(0, 10)),
    supabase.from('property_data').select('date, location, metric_name, value'),
  ]);
  if (indicators.error) {
    return NextResponse.json({ error: indicators.error.message }, { status: 500 });
  }

  const series: SeriesMap = {};
  for (const row of indicators.data ?? []) {
    (series[row.indicator_name] ??= []).push({
      date: row.date,
      value: Number(row.value),
      source: row.source,
    });
  }
  const propertyData: PropertyPoint[] = (property.data ?? []).map((r) => ({
    date: r.date,
    location: r.location,
    metric_name: r.metric_name,
    value: Number(r.value),
  }));

  const v3 = calculateMarketScoreV3(series, propertyData, now);
  const commentary = generateAutoCommentary(series, v3.score, v3.signal, now);
  const audit = buildAudit(series, now);

  const { error } = await supabase.from('market_score_history').upsert(
    {
      score_date: now.toISOString().slice(0, 10),
      final_score: v3.score,
      signal: v3.signal,
      base_score: v3.breakdown.base_score,
      sub_scores: v3.breakdown.sub_scores,
      breakdown: v3.breakdown,
      confidence: v3.breakdown.confidence_interval,
      audit,
      commentary_md: commentary,
      computed_at: now.toISOString(),
    },
    { onConflict: 'score_date' },
  );
  if (error) {
    return NextResponse.json({ error: error.message }, { status: 500 });
  }

  return NextResponse.json({
    ok: true,
    score_date: now.toISOString().slice(0, 10),
    final_score: v3.score,
    signal: v3.signal,
    stale_indicators: audit.filter((a) => a.point_count > 0 && a.stale).map((a) => a.indicator),
  });
}
