import Link from 'next/link';
import { getSiteData } from '@/lib/data';
import { calculateMarketScoreV3, buildAudit } from '@/lib/scoring/engine';
import type { IndicatorScore } from '@/lib/scoring/engine';
import { buildCommentaryParts, type CommentaryFactor } from '@/lib/scoring/commentary';
import { INDICATORS_CONFIG } from '@/lib/scoring/config';
import { cycleCaption, indicatorLabel, riskNotice, scoreDaysAgo } from '@/lib/scoring/ui-helpers';
import { STATUS_META, bandDistances, signalBand, scoreTone } from '@/lib/theme';
import { fmtDate, fmtDayMonth, fmtIndicator, sentenceCase, stripLeadingEmoji } from '@/lib/format';
import { ScoreHistoryChart } from '@/components/IndicatorChart';
import { ScoreGauge } from '@/components/ScoreGauge';
import { SetupNotice } from '@/components/SetupNotice';
import { InfoTip } from '@/components/InfoTip';
import { Bar, Card, DataStamp, Fill, SectionTitle } from '@/components/ui';

export const revalidate = 3600;

const KEY_ROWS = ['interest_rate', 'household_debt_gdp', 'rental_vacancy_rate', 'mortgage_stress_rate'];

const PILLARS: Array<[string, 'affordability' | 'supply_demand' | 'financial_stress' | 'momentum', string]> = [
  ['Affordability', 'affordability', 'cash rate, debt, mortgage stress'],
  ['Supply and demand', 'supply_demand', 'vacancy, approvals, clearance'],
  ['Financial health', 'financial_stress', 'unemployment, stress, credit'],
  ['Momentum', 'momentum', 'direction of every indicator'],
];

function FactorColumn({ title, tone, items }: { title: string; tone: string; items: CommentaryFactor[] }) {
  return (
    <div>
      <h3 className={`mb-2 text-[13px] font-semibold ${tone}`}>
        {title} <span className="font-normal text-tertiary">({items.length})</span>
      </h3>
      {items.length ? (
        <ul className="space-y-2.5">
          {items.map((f) => (
            <li key={f.key} className="text-sm">
              {f.display}
              <span className="block text-xs text-secondary">{f.note}</span>
            </li>
          ))}
        </ul>
      ) : (
        <p className="text-sm text-tertiary">None right now.</p>
      )}
    </div>
  );
}

export default async function Dashboard() {
  const data = await getSiteData();
  if (!data) return <SetupNotice />;

  const now = new Date();
  const v3 = calculateMarketScoreV3(data.scoring, data.propertyData, now);
  const parts = buildCommentaryParts(data.scoring, v3.score, now);
  const ci = v3.breakdown.confidence_interval;
  const sub = v3.breakdown.sub_scores;
  const staleCount = buildAudit(data.series, now).filter((a) => a.point_count > 0 && a.stale).length;
  const band = signalBand(v3.score);
  const scores = (v3.breakdown.indicator_scores ?? {}) as Record<string, IndicatorScore>;
  const caption = cycleCaption(
    v3.breakdown.cycle_multiplier as number | undefined,
    v3.breakdown.cycle_position as number | undefined,
  );
  const risks = ((v3.breakdown.risk_warnings as string[] | undefined) ?? []).map(riskNotice);
  const indicatorCount = Object.keys(INDICATORS_CONFIG).length;

  // What the meter's caption says: which way the score has moved in a month, and
  // how close it is to the bands either side.
  const previous = scoreDaysAgo(data.scoreHistory);
  const change = previous ? v3.score - previous.score : null;
  const trend =
    previous && change !== null
      ? Math.abs(change) < 0.05
        ? `Unchanged since ${fmtDayMonth(previous.date)}`
        : `${change > 0 ? 'Up' : 'Down'} ${Math.abs(change).toFixed(1)} since ${fmtDayMonth(previous.date)}`
      : null;
  const near = bandDistances(v3.score);
  const position = [
    near.below && `${near.below.distance.toFixed(1)} above ${sentenceCase(near.below.label)}`,
    near.above && `${near.above.distance.toFixed(1)} below ${sentenceCase(near.above.label)}`,
  ]
    .filter(Boolean)
    .join(' · ');

  return (
    <div className="space-y-2.5">
      <h1 className="sr-only">Market score</h1>

      <div className="grid gap-2.5 md:grid-cols-2">
        <Card>
          <SectionTitle
            right={
              ci?.level && (
                <InfoTip label={`${ci.level} confidence`}>
                  The engine puts the score somewhere from {ci.lower.toFixed(1)} to {ci.upper.toFixed(1)}. It only checks that every indicator
                  has a reading, not how old each one is, so it can read high while some inputs are stale. Audit shows their ages.
                </InfoTip>
              )
            }
          >
            Market score
          </SectionTitle>
          <ScoreGauge score={v3.score} previous={previous} />
          <div className="mt-1 flex items-center justify-center gap-3">
            <span className="text-[56px] font-light leading-none tracking-tight">{v3.score.toFixed(1)}</span>
            <Fill fill={band.fill} className="px-3.5 py-1.5 text-[15px]">
              {sentenceCase(stripLeadingEmoji(v3.signal))}
            </Fill>
          </div>
          <div className="mt-3 space-y-0.5 text-center text-[13px] text-secondary">
            {trend && <p>{trend}</p>}
            <p>{position}</p>
            {caption && <p className="pt-1 text-xs text-tertiary">{caption}</p>}
          </div>
          {risks.map((r) => (
            <p key={r.text} className={`mt-2 text-sm font-medium ${STATUS_META[r.status].text}`}>
              {r.text}
            </p>
          ))}
        </Card>

        <Card className="flex flex-col">
          <SectionTitle right="Out of 100, higher is better">Four pillars</SectionTitle>
          <div className="flex flex-1 flex-col justify-around gap-3">
            {sub &&
              PILLARS.map(([label, key, hint]) => {
                const value = sub[key];
                const tone = scoreTone(value);
                return (
                  <div key={key}>
                    <div className="mb-1 flex items-baseline justify-between gap-2 text-sm">
                      <span className="text-secondary">
                        {label} <span className="hidden text-xs text-tertiary lg:inline">· {hint}</span>
                      </span>
                      <span className={`font-medium ${tone.text}`}>{value.toFixed(0)}</span>
                    </div>
                    <Bar value={value} fill={tone.bar} />
                  </div>
                );
              })}
          </div>
        </Card>

        <Card>
          <SectionTitle>Indicators</SectionTitle>
          <div className="-mx-2 grid grid-cols-[minmax(0,1fr)_auto_2.5rem] gap-x-3 px-2 pb-1 text-[11px] font-medium text-tertiary">
            <span />
            <span className="text-right">Reading</span>
            <span className="text-right">Score</span>
          </div>
          <ul className="-mx-2">
            {KEY_ROWS.map((key) => {
              const entry = scores[key];
              return (
                <li key={key}>
                  <Link
                    href={`/indicators#${key}`}
                    className="grid grid-cols-[minmax(0,1fr)_auto_2.5rem] items-baseline gap-x-3 rounded-lg px-2 py-1.5 text-sm transition-colors hover:bg-tile"
                  >
                    <span className="text-secondary">{indicatorLabel(key)}</span>
                    <span className="text-right">
                      {entry ? fmtIndicator(entry.value, INDICATORS_CONFIG[key].unit) : 'No data'}
                    </span>
                    <span className={`text-right font-medium ${entry ? scoreTone(entry.score).text : ''}`}>
                      {entry ? entry.score : ''}
                    </span>
                  </Link>
                </li>
              );
            })}
          </ul>
          <Link href="/indicators" className="mt-2 inline-block text-[13px] text-accent hover:underline">
            All {indicatorCount} indicators
          </Link>
        </Card>

        {data.scoreHistory.length > 1 && (
          <Card>
            <SectionTitle right={`${data.scoreHistory.length} days`}>Score history</SectionTitle>
            <ScoreHistoryChart points={data.scoreHistory.map((r) => ({ date: r.score_date, score: r.final_score }))} />
          </Card>
        )}
      </div>

      <Card>
        <SectionTitle
          right={
            staleCount > 0 ? (
              <Link href="/audit" className="text-caution hover:underline">
                {staleCount} stale indicator{staleCount > 1 ? 's' : ''}, see Audit
              </Link>
            ) : undefined
          }
        >
          Market commentary
        </SectionTitle>
        <div className="mb-5 rounded-lg bg-tile px-4 py-3">
          <p className="text-sm">{parts.recommendation}</p>
          <p className="mt-1 text-xs text-tertiary">
            Year {parts.cyclePosition.toFixed(1)} of 18.6 ({parts.phase})
          </p>
        </div>
        <div className="grid gap-5 md:grid-cols-3">
          <FactorColumn title="Bearish" tone="text-danger" items={parts.bearish} />
          <FactorColumn title="Bullish" tone="text-go" items={parts.bullish} />
          <FactorColumn title="Neutral or watch" tone="text-primary" items={parts.neutral} />
        </div>
        <p className="mt-5 text-xs text-tertiary">Generated from the latest indicator data on {fmtDate(now.toISOString())}.</p>
      </Card>

      <DataStamp fetchedAt={data.fetchedAt} />
    </div>
  );
}
