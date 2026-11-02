import Link from 'next/link';
import { getSiteData } from '@/lib/data';
import { calculateMarketScoreV3, buildAudit } from '@/lib/scoring/engine';
import type { IndicatorScore } from '@/lib/scoring/engine';
import { buildCommentaryParts, classifyIndicator, type CommentaryFactor } from '@/lib/scoring/commentary';
import { INDICATORS_CONFIG } from '@/lib/scoring/config';
import { indicatorStatus, statusPhrase } from '@/lib/scoring/ui-helpers';
import { STATUS_META, signalBand, scoreTone } from '@/lib/theme';
import { fmtDate, fmtIndicator, fmtMonthYear, stripLeadingEmoji } from '@/lib/format';
import { ScoreGauge } from '@/components/ScoreGauge';
import { ScoreHistoryChart } from '@/components/IndicatorChart';
import { Sparkline } from '@/components/Sparkline';
import { SetupNotice } from '@/components/SetupNotice';
import { Card, DataStamp, SectionTitle, StatusPill } from '@/components/ui';

export const revalidate = 3600;

const KEY_TILES = ['household_debt_gdp', 'mortgage_stress_rate', 'rental_vacancy_rate', 'auction_clearance_rate'];

const PILLARS: Array<[string, 'affordability' | 'supply_demand' | 'financial_stress' | 'momentum', string]> = [
  ['Affordability', 'affordability', 'cash rate, debt, mortgage stress'],
  ['Supply / Demand', 'supply_demand', 'vacancy, approvals, clearance'],
  ['Financial Health', 'financial_stress', 'unemployment, stress, credit'],
  ['Momentum', 'momentum', 'direction of every indicator'],
];

function FactorColumn({ title, tone, items }: { title: string; tone: string; items: CommentaryFactor[] }) {
  return (
    <div>
      <h3 className={`mb-2 text-xs font-semibold uppercase tracking-wider ${tone}`}>
        {title} <span className="text-slate-500">({items.length})</span>
      </h3>
      {items.length ? (
        <ul className="space-y-2">
          {items.map((f) => (
            <li key={f.key} className="rounded-lg bg-slate-800/50 px-3 py-2 text-sm">
              <span className="font-medium text-slate-100">{f.display}</span>
              <span className="block text-xs text-slate-400">{f.note}</span>
            </li>
          ))}
        </ul>
      ) : (
        <p className="text-sm text-slate-500">None right now.</p>
      )}
    </div>
  );
}

export default async function Dashboard() {
  const data = await getSiteData();
  if (!data) return <SetupNotice />;

  const now = new Date();
  const v3 = calculateMarketScoreV3(data.series, data.propertyData, now);
  const parts = buildCommentaryParts(data.series, v3.score, now);
  const ci = v3.breakdown.confidence_interval;
  const sub = v3.breakdown.sub_scores;
  const staleCount = buildAudit(data.series, now).filter((a) => a.point_count > 0 && a.stale).length;
  const band = signalBand(v3.score);
  const scores = (v3.breakdown.indicator_scores ?? {}) as Record<string, IndicatorScore>;
  const yearAgo = new Date(now);
  yearAgo.setUTCFullYear(yearAgo.getUTCFullYear() - 1);
  const yearAgoIso = yearAgo.toISOString().slice(0, 10);
  const latestDate: Record<string, string | undefined> = Object.fromEntries(
    KEY_TILES.map((k) => [k, (data.series[k] ?? []).map((p) => p.date).sort().slice(-1)[0]]),
  );

  return (
    <div className="space-y-8">
      <section className="grid gap-6 md:grid-cols-2">
        <Card className="flex flex-col">
          <SectionTitle>Market score</SectionTitle>
          <div className="flex flex-1 flex-col items-center justify-center gap-3">
            <ScoreGauge score={v3.score} range={ci ? [ci.lower, ci.upper] : null} />
            <span
              className="rounded-full px-4 py-1 text-lg font-semibold"
              style={{ background: `${band.color}22`, color: band.color, boxShadow: `inset 0 0 0 1px ${band.color}55` }}
            >
              {stripLeadingEmoji(v3.signal)}
            </span>
            {ci?.level && <p className="text-xs text-slate-500">{ci.level} confidence</p>}
          </div>
          {v3.breakdown.cycle_warning ? (
            <p className="mt-4 rounded-lg border border-amber-800/60 bg-amber-950/30 px-3 py-2 text-sm text-amber-200">
              {stripLeadingEmoji(String(v3.breakdown.cycle_warning))}
            </p>
          ) : null}
          {(v3.breakdown.risk_warnings as string[] | undefined)?.map((w) => (
            <p key={w} className="mt-2 rounded-lg bg-slate-800/80 px-3 py-2 text-sm">
              {w}
            </p>
          ))}
        </Card>

        <Card>
          <SectionTitle right={<span className="text-xs text-slate-500">0–100 · higher is more favourable</span>}>
            Four pillars
          </SectionTitle>
          <div className="space-y-4">
            {sub &&
              PILLARS.map(([label, key, hint]) => {
                const value = sub[key];
                const tone = scoreTone(value);
                return (
                  <div key={key}>
                    <div className="mb-1 flex items-baseline justify-between gap-2 text-sm">
                      <span>
                        {label} <span className="hidden text-xs text-slate-500 sm:inline">· {hint}</span>
                      </span>
                      <span className={`font-semibold tabular-nums ${tone.text}`}>{value.toFixed(0)}</span>
                    </div>
                    <div className="h-2 rounded-full bg-slate-800">
                      <div className={`h-2 rounded-full ${tone.bar}`} style={{ width: `${Math.max(0, Math.min(100, value))}%` }} />
                    </div>
                  </div>
                );
              })}
          </div>
          {data.scoreHistory.length > 1 && (
            <div className="mt-6">
              <h3 className="mb-1 text-xs font-semibold uppercase tracking-wider text-slate-500">
                Score history · {data.scoreHistory.length} days
              </h3>
              <ScoreHistoryChart
                points={data.scoreHistory.map((r) => ({ date: r.score_date, score: r.final_score }))}
              />
            </div>
          )}
        </Card>
      </section>

      <section>
        <SectionTitle>Key indicators</SectionTitle>
        <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
          {KEY_TILES.map((key) => {
            const cfg = INDICATORS_CONFIG[key];
            const entry = scores[key];
            if (!entry) {
              return (
                <div key={key} className="rounded-xl border border-slate-800 bg-slate-900/60 p-4">
                  <p className="text-xs text-slate-400">{cfg.display_name}</p>
                  <p className="mt-1 text-sm text-slate-500">No data</p>
                </div>
              );
            }
            const status = indicatorStatus(entry.value, cfg);
            const meta = STATUS_META[status];
            const [, note] = classifyIndicator(entry.value, cfg);
            const recent = (data.series[key] ?? [])
              .filter((p) => p.date >= yearAgoIso)
              .sort((a, b) => (a.date < b.date ? -1 : 1));
            const first = recent[0];
            return (
              <Link
                key={key}
                href="/indicators"
                className={`block rounded-xl border p-4 transition hover:brightness-125 ${meta.tile}`}
              >
                <div className="flex items-start justify-between gap-2">
                  <p className="text-xs text-slate-300">{cfg.display_name}</p>
                  <StatusPill status={status} />
                </div>
                <p className="mt-1 text-3xl font-semibold tabular-nums tracking-tight">
                  {fmtIndicator(entry.value, cfg.unit)}
                </p>
                <p className={`text-xs ${meta.text}`}>{statusPhrase(note)}</p>
                {first && recent.length > 1 ? (
                  <>
                    <div className="mt-2">
                      <Sparkline points={recent} stroke={meta.stroke} />
                    </div>
                    <p className="mt-1 text-xs text-slate-400">
                      {fmtMonthYear(first.date)}: {fmtIndicator(first.value, cfg.unit)} → now{' '}
                      {fmtIndicator(entry.value, cfg.unit)}
                    </p>
                  </>
                ) : (
                  <p className="mt-3 text-xs text-slate-400">
                    {recent.length === 1 ? 'Only one reading in the past year' : 'No readings in the past year'}
                    {latestDate[key] ? ` — latest ${fmtDate(latestDate[key])}` : ''}
                  </p>
                )}
              </Link>
            );
          })}
        </div>
      </section>

      <Card>
        <SectionTitle
          right={
            staleCount > 0 ? (
              <Link href="/audit" className="text-xs text-amber-400 hover:underline">
                {staleCount} stale indicator{staleCount > 1 ? 's' : ''} — see Data Audit →
              </Link>
            ) : undefined
          }
        >
          Market commentary
        </SectionTitle>
        <div className="mb-5 rounded-lg border border-sky-800/70 bg-sky-950/30 px-4 py-3">
          <p className="text-xs font-semibold uppercase tracking-wider text-sky-300">Recommendation</p>
          <p className="mt-1 text-base text-slate-100">{parts.recommendation}</p>
          <p className="mt-1 text-xs text-slate-400">
            Cycle position: Year {parts.cyclePosition.toFixed(1)} of 18.6 ({parts.phase})
          </p>
        </div>
        <div className="grid gap-5 md:grid-cols-3">
          <FactorColumn title="Bearish" tone="text-red-400" items={parts.bearish} />
          <FactorColumn title="Bullish" tone="text-emerald-400" items={parts.bullish} />
          <FactorColumn title="Neutral / watch" tone="text-slate-300" items={parts.neutral} />
        </div>
        <p className="mt-5 text-xs italic text-slate-500">Auto-generated from the latest indicator data on {fmtDate(now.toISOString())}.</p>
      </Card>

      <DataStamp fetchedAt={data.fetchedAt} />
    </div>
  );
}
