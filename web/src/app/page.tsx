import { getSiteData } from '@/lib/data';
import { calculateMarketScoreV3, buildAudit } from '@/lib/scoring/engine';
import type { IndicatorScore } from '@/lib/scoring/engine';
import { generateAutoCommentary } from '@/lib/scoring/commentary';
import { INDICATORS_CONFIG } from '@/lib/scoring/config';
import { indicatorStatus } from '@/lib/scoring/ui-helpers';
import { ScoreGauge } from '@/components/ScoreGauge';
import { Markdown } from '@/components/Markdown';
import { ScoreSparkline } from '@/components/IndicatorChart';
import { SetupNotice } from '@/components/SetupNotice';

export const revalidate = 3600;

const KEY_TILES: Array<[string, (v: number) => string]> = [
  ['household_debt_gdp', (v) => `${v.toFixed(1)}%`],
  ['mortgage_stress_rate', (v) => `${v.toFixed(0)}%`],
  ['rental_vacancy_rate', (v) => `${v.toFixed(1)}%`],
  ['auction_clearance_rate', (v) => `${v.toFixed(0)}%`],
];

const STATUS_STYLES: Record<string, string> = {
  success: 'border-emerald-700 bg-emerald-950/40',
  warning: 'border-amber-700 bg-amber-950/40',
  danger: 'border-red-700 bg-red-950/40',
};

export default async function Dashboard() {
  const data = await getSiteData();
  if (!data) return <SetupNotice />;

  const now = new Date();
  const v3 = calculateMarketScoreV3(data.series, data.propertyData, now);
  const commentary = generateAutoCommentary(data.series, v3.score, v3.signal, now);
  const ci = v3.breakdown.confidence_interval;
  const sub = v3.breakdown.sub_scores;
  const staleCount = buildAudit(data.series, now).filter((a) => a.point_count > 0 && a.stale).length;

  return (
    <div className="space-y-8">
      <section className="grid gap-6 md:grid-cols-2">
        <div className="rounded-xl border border-slate-800 bg-slate-900/60 p-6">
          <h1 className="text-sm font-medium uppercase tracking-wide text-slate-400">Market Score</h1>
          <div className="mt-2 flex items-center gap-4">
            <ScoreGauge score={v3.score} label={`${ci?.lower}–${ci?.upper} (${ci?.level} confidence)`} />
          </div>
          <p className="mt-3 text-xl font-semibold">{v3.signal}</p>
          <p className="mt-1 text-sm text-slate-400">{String(v3.breakdown.cycle_warning ?? '')}</p>
          {(v3.breakdown.risk_warnings as string[] | undefined)?.map((w) => (
            <p key={w} className="mt-2 rounded-md bg-slate-800/80 px-3 py-2 text-sm">{w}</p>
          ))}
        </div>

        <div className="rounded-xl border border-slate-800 bg-slate-900/60 p-6">
          <h2 className="text-sm font-medium uppercase tracking-wide text-slate-400">Four Pillars</h2>
          <div className="mt-4 space-y-4">
            {sub &&
              (
                [
                  ['Affordability', sub.affordability],
                  ['Supply / Demand', sub.supply_demand],
                  ['Financial Health', sub.financial_stress],
                  ['Momentum', sub.momentum],
                ] as Array<[string, number]>
              ).map(([label, value]) => (
                <div key={label}>
                  <div className="mb-1 flex justify-between text-sm">
                    <span>{label}</span>
                    <span className="tabular-nums text-slate-300">{value.toFixed(0)}</span>
                  </div>
                  <div className="h-2 rounded-full bg-slate-800">
                    <div
                      className="h-2 rounded-full bg-sky-500"
                      style={{ width: `${Math.max(0, Math.min(100, value))}%` }}
                    />
                  </div>
                </div>
              ))}
          </div>
          {data.scoreHistory.length > 1 && (
            <div className="mt-6">
              <h3 className="mb-1 text-xs uppercase tracking-wide text-slate-500">Score history</h3>
              <ScoreSparkline
                points={data.scoreHistory.map((r) => ({ date: r.score_date, score: r.final_score }))}
              />
            </div>
          )}
        </div>
      </section>

      <section>
        <h2 className="mb-3 text-sm font-medium uppercase tracking-wide text-slate-400">
          Key Indicators
        </h2>
        <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
          {KEY_TILES.map(([key, fmt]) => {
            const cfg = INDICATORS_CONFIG[key];
            const scores = (v3.breakdown.indicator_scores ?? {}) as Record<string, IndicatorScore>;
            const entry = scores[key];
            if (!entry) {
              return (
                <div key={key} className="rounded-lg border border-slate-800 bg-slate-900/60 p-4">
                  <p className="text-xs text-slate-400">{cfg.display_name}</p>
                  <p className="mt-1 text-sm text-slate-500">No data</p>
                </div>
              );
            }
            const status = indicatorStatus(entry.value, cfg);
            return (
              <div key={key} className={`rounded-lg border p-4 ${STATUS_STYLES[status]}`}>
                <p className="text-xs text-slate-300">{cfg.display_name}</p>
                <p className="mt-1 text-2xl font-semibold tabular-nums">{fmt(entry.value)}</p>
                <p className="mt-1 text-xs text-slate-400">
                  trend: {entry.trend}
                  {entry.trend !== 'n/a' && entry.trend !== 'stable'
                    ? ` (${entry.trend_change > 0 ? '+' : ''}${entry.trend_change.toFixed(1)}%)`
                    : ''}
                </p>
              </div>
            );
          })}
        </div>
      </section>

      <section className="rounded-xl border border-slate-800 bg-slate-900/60 p-6">
        <div className="mb-3 flex items-center justify-between">
          <h2 className="text-sm font-medium uppercase tracking-wide text-slate-400">
            Market Commentary
          </h2>
          {staleCount > 0 && (
            <a href="/audit" className="text-xs text-amber-400 hover:underline">
              ⚠️ {staleCount} stale indicator{staleCount > 1 ? 's' : ''} — see Data Audit
            </a>
          )}
        </div>
        <Markdown text={commentary} />
      </section>

      <p className="text-xs text-slate-500">Data as of {data.fetchedAt.slice(0, 16).replace('T', ' ')} UTC</p>
    </div>
  );
}
