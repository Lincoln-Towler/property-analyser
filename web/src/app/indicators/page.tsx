import { getSiteData } from '@/lib/data';
import {
  calculateMarketScoreV3,
  getIndicatorTrend,
  getWindowContext,
  windowContradictsTrend,
} from '@/lib/scoring/engine';
import { INDICATORS_CONFIG, EXTRA_INDICATOR_NAMES } from '@/lib/scoring/config';
import { indicatorStatus, indicatorTargetText, targetBand, warningLine } from '@/lib/scoring/ui-helpers';
import { scoreTone } from '@/lib/theme';
import { fmtDate, fmtIndicator } from '@/lib/format';
import { IndicatorChart } from '@/components/IndicatorChart';
import { SetupNotice } from '@/components/SetupNotice';
import { Card, DataStamp, PageHeader, StatusPill } from '@/components/ui';
import type { SiteData } from '@/lib/data';

export const revalidate = 3600;

/** Indicators grouped the way the four pillars on the dashboard use them
 *  (see calculateSubScores). Mortgage stress feeds two pillars; it is shown
 *  once, under Affordability. */
const GROUPS: Array<{
  title: string;
  pillar?: 'affordability' | 'supply_demand' | 'financial_stress';
  note: string;
  keys: string[];
}> = [
  {
    title: 'Affordability',
    pillar: 'affordability',
    note: 'Can buyers afford to borrow and keep paying?',
    keys: ['interest_rate', 'household_debt_gdp', 'mortgage_stress_rate'],
  },
  {
    title: 'Supply / Demand',
    pillar: 'supply_demand',
    note: 'Is housing scarce, and are buyers competing for it?',
    keys: ['rental_vacancy_rate', 'building_approvals', 'auction_clearance_rate'],
  },
  {
    title: 'Financial Health',
    pillar: 'financial_stress',
    note: 'Are households under strain? Mortgage stress (above) also feeds this pillar.',
    keys: ['unemployment_rate', 'credit_growth'],
  },
  {
    title: 'Other scored',
    note: 'Counts toward the headline score and momentum, but no single pillar.',
    keys: ['wage_growth'],
  },
];

const STEP_LINE = new Set(['interest_rate']);

function IndicatorCard({ name, data, now }: { name: string; data: SiteData; now: Date }) {
  const cfg = INDICATORS_CONFIG[name];
  const points = (data.series[name] ?? []).slice().sort((a, b) => (a.date < b.date ? -1 : 1));
  const latest = points.length ? points[points.length - 1] : null;
  const displayName = cfg?.display_name ?? EXTRA_INDICATOR_NAMES[name] ?? name.replace(/_/g, ' ');
  const status = cfg ? indicatorStatus(latest?.value ?? null, cfg) : null;
  const target = cfg ? indicatorTargetText(cfg) : null;
  const unit = cfg?.unit ?? '';
  const isAnnualized = data.adjustments.annualized.includes(name);
  // Honest window shape alongside the scored two-point trend — surfaced only
  // when the two disagree (see getWindowContext).
  const scoredPoints = data.scoring[name] ?? [];
  const [scoredTrend] = getIndicatorTrend(scoredPoints, now, 3);
  const windowCtx = getWindowContext(scoredPoints, now, 3);
  const showWindow = windowContradictsTrend(windowCtx, scoredTrend);

  return (
    <Card as="article" tight>
      <header className="mb-3 space-y-1">
        <div className="flex items-start justify-between gap-3">
          <h3 className="font-medium leading-snug">{displayName}</h3>
          {status && <StatusPill status={status} />}
        </div>
        {latest ? (
          <p className="flex flex-wrap items-baseline gap-x-3 gap-y-0.5 text-xs text-slate-400">
            <span className="text-2xl font-semibold tabular-nums tracking-tight text-slate-100">
              {fmtIndicator(latest.value, unit)}
            </span>
            {target && <span>Target {target}</span>}
            <span className="whitespace-nowrap">as of {fmtDate(latest.date)}</span>
            {cfg && <span className="whitespace-nowrap">weight {cfg.weight}</span>}
          </p>
        ) : (
          <p className="text-xs text-slate-400">No data yet</p>
        )}
        {isAnnualized && (
          <p className="text-xs text-sky-400/80">annualised from monthly readings (trailing 12-month rate)</p>
        )}
        {showWindow && windowCtx && (
          <p className="text-xs text-amber-400">
            Scored trend reads <em>{scoredTrend}</em>, but the last {windowCtx.pointsInWindow} readings over{' '}
            {windowCtx.spanDays} days moved {windowCtx.changePct > 0 ? '+' : ''}
            {windowCtx.changePct.toFixed(1)}%
            {windowCtx.monotoneSteps === windowCtx.totalSteps
              ? ` (${windowCtx.totalSteps} of ${windowCtx.totalSteps} consecutive ${
                  windowCtx.direction === 'falling' ? 'declines' : 'rises'
                })`
              : ''}
          </p>
        )}
      </header>
      {points.length >= 2 ? (
        <IndicatorChart
          points={points.map((p) => ({ date: p.date, value: p.value }))}
          unit={unit}
          step={STEP_LINE.has(name)}
          targetBand={cfg ? targetBand(cfg) : null}
          warning={cfg ? warningLine(cfg) : null}
        />
      ) : (
        <p className="rounded-lg bg-slate-800/50 px-3 py-6 text-center text-sm text-slate-400">
          {points.length === 1 ? 'Only one reading so far — nothing to chart yet.' : 'No data — add rows in Supabase or via the n8n feed.'}
        </p>
      )}
    </Card>
  );
}

function Legend() {
  return (
    <div className="flex flex-wrap items-center gap-x-5 gap-y-2 rounded-xl border border-slate-800 bg-slate-900/40 px-4 py-3 text-xs text-slate-400">
      <StatusPill status="success" />
      <StatusPill status="warning" />
      <StatusPill status="danger" />
      <span className="inline-flex items-center gap-1.5">
        <span className="h-3 w-5 rounded-sm bg-emerald-500/20" aria-hidden /> target zone
      </span>
      <span className="inline-flex items-center gap-1.5">
        <span className="w-5 border-t-2 border-dashed border-red-400" aria-hidden /> warning threshold
      </span>
      <span>Charts use real dates — gaps between readings are to scale.</span>
    </div>
  );
}

export default async function IndicatorsPage() {
  const data = await getSiteData();
  if (!data) return <SetupNotice />;

  const now = new Date();
  const sub = calculateMarketScoreV3(data.scoring, data.propertyData, now).breakdown.sub_scores;
  const byWeight = (a: string, b: string) => (INDICATORS_CONFIG[b]?.weight ?? 0) - (INDICATORS_CONFIG[a]?.weight ?? 0);
  const context = Object.keys(data.series).filter((n) => !(n in INDICATORS_CONFIG));

  return (
    <div className="space-y-8">
      <PageHeader title="Economic indicators">
        Every input to the market score, grouped by the pillar it feeds on the dashboard and ordered by
        weight within each group.
      </PageHeader>
      <Legend />

      {GROUPS.map((group) => {
        const pillarScore = group.pillar && sub ? sub[group.pillar] : null;
        return (
          <section key={group.title} className="space-y-3">
            <div className="flex flex-wrap items-baseline justify-between gap-x-4 gap-y-1 border-b border-slate-800 pb-2">
              <div>
                <h2 className="text-lg font-semibold">{group.title}</h2>
                <p className="text-xs text-slate-400">{group.note}</p>
              </div>
              {pillarScore !== null && (
                <p className="text-sm text-slate-400">
                  pillar score{' '}
                  <span className={`text-lg font-semibold tabular-nums ${scoreTone(pillarScore).text}`}>
                    {pillarScore.toFixed(0)}
                  </span>
                </p>
              )}
            </div>
            <div className="grid gap-4 lg:grid-cols-2">
              {[...group.keys].sort(byWeight).map((key) => (
                <IndicatorCard key={key} name={key} data={data} now={now} />
              ))}
            </div>
          </section>
        );
      })}

      {context.length > 0 && (
        <section className="space-y-3">
          <div className="border-b border-slate-800 pb-2">
            <h2 className="text-lg font-semibold">Context only</h2>
            <p className="text-xs text-slate-400">Tracked for background — not part of the score.</p>
          </div>
          <div className="grid gap-4 lg:grid-cols-2">
            {context.map((key) => (
              <IndicatorCard key={key} name={key} data={data} now={now} />
            ))}
          </div>
        </section>
      )}

      <DataStamp fetchedAt={data.fetchedAt} />
    </div>
  );
}
