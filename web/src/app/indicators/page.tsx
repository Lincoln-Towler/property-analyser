import { getSiteData } from '@/lib/data';
import {
  calculateMarketScoreV3,
  getIndicatorTrend,
  getWindowContext,
  windowContradictsTrend,
} from '@/lib/scoring/engine';
import { classifyIndicator } from '@/lib/scoring/commentary';
import { INDICATORS_CONFIG } from '@/lib/scoring/config';
import {
  indicatorLabel,
  indicatorStatus,
  indicatorTargetText,
  statusPhrase,
  targetBand,
  warningLine,
} from '@/lib/scoring/ui-helpers';
import { scoreTone } from '@/lib/theme';
import { fmtDate, fmtIndicator } from '@/lib/format';
import { IndicatorChart } from '@/components/IndicatorChart';
import { OpenFromHash } from '@/components/OpenFromHash';
import { SetupNotice } from '@/components/SetupNotice';
import { Card, DataStamp, PageHeader, Row, SectionTitle, StatusText } from '@/components/ui';
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
    title: 'Supply and demand',
    pillar: 'supply_demand',
    note: 'Is housing scarce, and are buyers competing for it?',
    keys: ['rental_vacancy_rate', 'building_approvals', 'auction_clearance_rate'],
  },
  {
    title: 'Financial health',
    pillar: 'financial_stress',
    note: 'Are households under strain? Mortgage stress also feeds this pillar.',
    keys: ['unemployment_rate', 'credit_growth'],
  },
  {
    title: 'Other scored',
    note: 'Counts toward the headline score and momentum, but no single pillar.',
    keys: ['wage_growth'],
  },
];

const STEP_LINE = new Set(['interest_rate']);

function IndicatorRow({ name, data, now }: { name: string; data: SiteData; now: Date }) {
  const cfg = INDICATORS_CONFIG[name];
  const points = (data.series[name] ?? []).slice().sort((a, b) => (a.date < b.date ? -1 : 1));
  const latest = points.length ? points[points.length - 1] : null;
  const status = cfg ? indicatorStatus(latest?.value ?? null, cfg) : null;
  const target = cfg ? indicatorTargetText(cfg) : null;
  const unit = cfg?.unit ?? '';
  const isAnnualized = data.adjustments.annualized.includes(name);
  const note = cfg && latest ? statusPhrase(classifyIndicator(latest.value, cfg)[1]) : null;
  // Honest window shape alongside the scored two-point trend — surfaced only
  // when the two disagree (see getWindowContext).
  const scoredPoints = data.scoring[name] ?? [];
  const [scoredTrend] = getIndicatorTrend(scoredPoints, now, 3);
  const windowCtx = getWindowContext(scoredPoints, now, 3);
  const showWindow = windowContradictsTrend(windowCtx, scoredTrend);
  const band = cfg ? targetBand(cfg) : null;
  const warning = cfg ? warningLine(cfg) : null;

  return (
    <details id={name} className="group scroll-mt-16 rounded-lg">
      <summary className="grid grid-cols-[minmax(0,1fr)_auto_4.25rem_0.75rem] items-baseline gap-x-3 rounded-lg px-2 py-2 transition-colors hover:bg-tile group-open:bg-tile">
        <span className="text-sm text-secondary group-open:text-primary">{indicatorLabel(name)}</span>
        <span className="text-right text-sm font-medium">{latest ? fmtIndicator(latest.value, unit) : 'No data'}</span>
        <span className="text-right">{status && <StatusText status={status} />}</span>
        <span className="chevron text-center text-tertiary" aria-hidden>
          ›
        </span>
      </summary>
      <div className="px-2 pb-4 pt-3">
        {note && <p className="mb-2 text-sm text-secondary">{note}</p>}
        <dl className="mb-3">
          {target && <Row label="Target">{target}</Row>}
          {latest && <Row label="Latest reading">{fmtDate(latest.date)}</Row>}
          {cfg && <Row label="Weight in the score">{cfg.weight}</Row>}
        </dl>
        {isAnnualized && <p className="mb-2 text-xs text-tertiary">Annualised from monthly readings (trailing 12-month rate).</p>}
        {showWindow && windowCtx && (
          <p className="mb-2 text-xs text-caution">
            The scored trend reads {scoredTrend}, but the last {windowCtx.pointsInWindow} readings over {windowCtx.spanDays} days moved{' '}
            {windowCtx.changePct > 0 ? '+' : ''}
            {windowCtx.changePct.toFixed(1)}%
            {windowCtx.monotoneSteps === windowCtx.totalSteps
              ? ` (${windowCtx.totalSteps} of ${windowCtx.totalSteps} consecutive ${windowCtx.direction === 'falling' ? 'declines' : 'rises'})`
              : ''}
            .
          </p>
        )}
        {points.length >= 2 ? (
          <>
            <IndicatorChart
              points={points.map((p) => ({ date: p.date, value: p.value }))}
              unit={unit}
              step={STEP_LINE.has(name)}
              targetBand={band}
              warning={warning}
            />
            {(band || warning) && (
              <p className="mt-2 text-xs text-tertiary">
                {band ? 'Shaded: target zone. ' : ''}
                {warning ? 'Dashed: warning line.' : ''}
              </p>
            )}
          </>
        ) : (
          <p className="rounded-lg bg-tile px-3 py-6 text-center text-sm text-secondary">
            {points.length === 1 ? 'Only one reading so far, so nothing to chart yet.' : 'No data yet.'}
          </p>
        )}
      </div>
    </details>
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
    <div className="space-y-2.5">
      <OpenFromHash />
      <PageHeader title="Indicators">Everything that feeds the market score. Tap a row for its chart.</PageHeader>

      {GROUPS.map((group) => {
        const pillarScore = group.pillar && sub ? sub[group.pillar] : null;
        return (
          <Card key={group.title}>
            <SectionTitle
              right={
                pillarScore !== null ? (
                  <>
                    Pillar score <span className={`text-sm font-medium ${scoreTone(pillarScore).text}`}>{pillarScore.toFixed(0)}</span>
                  </>
                ) : undefined
              }
            >
              {group.title}
            </SectionTitle>
            <p className="-mt-2 mb-2 text-xs text-tertiary">{group.note}</p>
            <div className="-mx-2">
              {[...group.keys].sort(byWeight).map((key) => (
                <IndicatorRow key={key} name={key} data={data} now={now} />
              ))}
            </div>
          </Card>
        );
      })}

      {context.length > 0 && (
        <Card>
          <SectionTitle>Context only</SectionTitle>
          <p className="-mt-2 mb-2 text-xs text-tertiary">Tracked for background, and not part of the score.</p>
          <div className="-mx-2">
            {context.map((key) => (
              <IndicatorRow key={key} name={key} data={data} now={now} />
            ))}
          </div>
        </Card>
      )}

      <DataStamp fetchedAt={data.fetchedAt} />
    </div>
  );
}
