import { getSiteData } from '@/lib/data';
import { getBtcData } from '@/lib/btc';
import { buildFeedHealth } from '@/lib/feed-health';
import { buildAudit } from '@/lib/scoring/engine';
import { INDICATORS_CONFIG, STALE_AFTER_DAYS } from '@/lib/scoring/config';
import { indicatorLabel } from '@/lib/scoring/ui-helpers';
import { auditPropertyData } from '@/lib/property-audit';
import { groupByPublisher } from '@/lib/sources';
import { fmtDate, fmtIndicator } from '@/lib/format';
import { SetupNotice } from '@/components/SetupNotice';
import { Card, DataStamp, Fill, PageHeader, SectionTitle } from '@/components/ui';

export const revalidate = 3600;

function AgeBadge({ days }: { days: number | null }) {
  if (days === null) return <span className="text-xs text-tertiary">No data</span>;
  const tone = days > STALE_AFTER_DAYS ? 'text-danger' : days > 35 ? 'text-caution' : 'text-go';
  return (
    <span className={`whitespace-nowrap text-[13px] font-medium ${tone}`}>
      {days}d{days > STALE_AFTER_DAYS ? ', stale' : ''}
    </span>
  );
}

export default async function AuditPage() {
  const data = await getSiteData();
  if (!data) return <SetupNotice />;

  const audit = buildAudit(data.series, new Date());
  const propertyAudit = auditPropertyData(data.propertyData);
  const btc = await getBtcData();
  const health = buildFeedHealth({
    series: data.series,
    propertyData: data.propertyData,
    scoreHistory: data.scoreHistory,
    btc,
    now: new Date(),
  });

  // Scored weight that cannot move: stale, or unable to trend at all.
  const frozen = audit.filter((a) => a.weight > 0 && (a.stale || !a.can_trend));
  const frozenWeightPct = frozen.reduce((sum, a) => sum + a.weight_pct, 0);

  return (
    <div className="space-y-2.5">
      <PageHeader title="Audit">
        Whether the feeds ran, and how fresh each input is. A stale value still feeds the score, so look here first when the headline number
        seems off. Anything older than {STALE_AFTER_DAYS} days is flagged.
      </PageHeader>

      <Card>
        <SectionTitle right={<Fill fill={health.healthy ? 'bg-go' : 'bg-danger'}>{health.healthy ? 'All on schedule' : 'Needs a look'}</Fill>}>
          Automation health
        </SectionTitle>
        <ul>
          {health.jobs.map((j) => (
            <li key={j.name} className="grid grid-cols-[1fr_auto] items-baseline gap-x-4 py-1 sm:grid-cols-[1fr_9rem_8rem_auto]">
              <span className="text-sm">{j.name}</span>
              <span className={`whitespace-nowrap text-[13px] font-medium sm:order-last ${j.ok ? 'text-go' : 'text-danger'}`}>
                {j.ok ? 'On schedule' : 'Late'}
              </span>
              <span className="text-xs text-tertiary">{j.schedule}</span>
              <span className="whitespace-nowrap text-right text-xs text-tertiary sm:text-left">{fmtDate(j.latest)}</span>
            </li>
          ))}
        </ul>
        {health.late.length > 0 && (
          <div className="mt-3">
            <h3 className="mb-1 text-[13px] font-semibold text-secondary">Older than expected</h3>
            <ul className="space-y-1 text-sm text-secondary">
              {health.late.map((l) => (
                <li key={l.indicator}>
                  <span className="text-primary">{indicatorLabel(l.indicator)}</span>:{' '}
                  {l.ageDays === null ? 'no data' : `latest reading ${fmtDate(l.latest)}`}
                  {l.manual ? ' (entered by hand, needs a manual update)' : ' (the feed may have stopped)'}
                </li>
              ))}
            </ul>
          </div>
        )}
        {health.failedVideos.length > 0 && (
          <div className="mt-3">
            <h3 className="mb-1 text-[13px] font-semibold text-secondary">Videos that couldn&apos;t be analysed</h3>
            <ul className="space-y-1 text-sm text-secondary">
              {health.failedVideos.map((v) => (
                <li key={v.title}>
                  {v.title} <span className="text-tertiary">({v.attempts} of 3 attempts)</span>
                </li>
              ))}
            </ul>
          </div>
        )}
        {health.failedChanges.length > 0 && (
          <div className="mt-3">
            <h3 className="mb-1 text-[13px] font-semibold text-secondary">Flag changes that didn&apos;t apply</h3>
            <ul className="space-y-1 text-sm text-secondary">
              {health.failedChanges.map((c) => (
                <li key={c.flag_key}>
                  {c.flag_key}: {c.note}
                </li>
              ))}
            </ul>
          </div>
        )}
      </Card>

      {frozen.length > 0 && (
        <Card>
          <SectionTitle>{frozenWeightPct.toFixed(0)}% of the score is riding on data that can&apos;t move</SectionTitle>
          <p className="-mt-2 mb-3 text-xs text-tertiary">
            These indicators still carry their full weight, but they&apos;re stale or have too few recent readings to show a trend. Nothing is
            excluded from scoring: the fix is more data, not different maths.
          </p>
          <ul className="space-y-2 text-sm text-secondary">
            {frozen.map((a) => (
              <li key={a.indicator}>
                <span className="text-primary">{indicatorLabel(a.indicator)}</span>: {a.weight} of 135 ({a.weight_pct.toFixed(1)}% of the
                score), {a.days_old === null ? 'no data' : `${a.days_old} days old`}, {a.point_count}{' '}
                {a.point_count === 1 ? 'reading' : 'readings'}
                {!a.can_trend && ', cannot trend'}
                {!a.can_measure_volatility && ', volatility unmeasurable'}
                {a.cadence_blocks_trend && (
                  <span className="text-caution">
                    . {a.expected_frequency} data can never fill a 3-month trend window
                    {a.cadence_blocks_volatility ? ' or a 6-month volatility window' : ''}, so backfilling won&apos;t change this
                  </span>
                )}
              </li>
            ))}
          </ul>
        </Card>
      )}

      <Card>
        <SectionTitle>Indicators</SectionTitle>
        <div className="no-scrollbar -mx-2 overflow-x-auto">
          <table className="w-full min-w-[640px] text-sm">
            <thead className="text-left text-[11px] text-tertiary">
              <tr>
                <th className="px-2 pb-1.5 font-medium">Indicator</th>
                <th className="px-2 pb-1.5 font-medium">Weight</th>
                <th className="px-2 pb-1.5 font-medium">Latest value</th>
                <th className="px-2 pb-1.5 font-medium">Latest date</th>
                <th className="px-2 pb-1.5 font-medium">Age</th>
                <th className="px-2 pb-1.5 font-medium">Points (24 months)</th>
                <th className="px-2 pb-1.5 font-medium">Sources</th>
              </tr>
            </thead>
            <tbody>
              {audit.map((row) => {
                const publishers = groupByPublisher(row.sources);
                return (
                  <tr key={row.indicator} className="transition-colors hover:bg-tile">
                    <td className="whitespace-nowrap rounded-l-lg px-2 py-1.5">{indicatorLabel(row.indicator)}</td>
                    <td className="px-2 py-1.5 text-tertiary">{INDICATORS_CONFIG[row.indicator]?.weight ?? '—'}</td>
                    <td className="whitespace-nowrap px-2 py-1.5">{fmtIndicator(row.latest_value, INDICATORS_CONFIG[row.indicator]?.unit ?? '')}</td>
                    <td className="whitespace-nowrap px-2 py-1.5 text-secondary">{fmtDate(row.latest_date)}</td>
                    <td className="px-2 py-1.5">
                      <AgeBadge days={row.days_old} />
                    </td>
                    <td className="px-2 py-1.5 text-secondary">{row.point_count}</td>
                    <td className="rounded-r-lg px-2 py-1.5">
                      {publishers.length ? (
                        <div className="flex flex-wrap gap-1">
                          {publishers.map(([pub, n]) => (
                            <span key={pub} className="whitespace-nowrap rounded-md bg-tile px-1.5 py-0.5 text-xs text-secondary">
                              {pub} <span className="text-tertiary">×{n}</span>
                            </span>
                          ))}
                        </div>
                      ) : (
                        '—'
                      )}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </Card>

      {(data.adjustments.dropped.length > 0 || data.adjustments.annualized.length > 0 || data.adjustments.changesOnly.length > 0) && (
        <Card>
          <SectionTitle>Data adjustments applied</SectionTitle>
          {data.adjustments.dropped.length > 0 && (
            <div className="mb-2 text-sm text-secondary">
              <p className="mb-1">Implausible values left out of charts and scoring, which usually means a broken feed:</p>
              <ul className="list-disc pl-5">
                {data.adjustments.dropped.map((d) => (
                  <li key={`${d.indicator}-${d.date}`}>
                    {indicatorLabel(d.indicator)}: {d.value} on {fmtDate(d.date)}
                  </li>
                ))}
              </ul>
            </div>
          )}
          {data.adjustments.annualized.map((name) => (
            <p key={name} className="text-sm text-secondary">
              <span className="text-primary">{indicatorLabel(name)}</span>: the feed delivers monthly readings, so values are shown and scored as a
              trailing 12-month annualised rate to match the annual thresholds.
            </p>
          ))}
          {data.adjustments.changesOnly.map((name) => (
            <p key={name} className="text-sm text-secondary">
              <span className="text-primary">{indicatorLabel(name)}</span>: saved on every feed run, so the score counts only the readings
              where it actually changed. Otherwise a repeat of the same rate would read as &quot;stable&quot; straight after a rise.
            </p>
          ))}
        </Card>
      )}

      {data.propertyData.length > 0 && (
        <Card>
          <SectionTitle>Property data quality</SectionTitle>
          <p className="-mt-2 mb-3 text-xs text-tertiary">
            {propertyAudit.locations} locations, newest snapshot {fmtDate(propertyAudit.latestDate)}. Nothing here is excluded from charts or
            scoring: these are readings worth a second look, not errors. Locations mix capitals, suburbs and regional towns, so an extreme value
            is often a narrow sub-market sample rather than a bad number.
          </p>

          {propertyAudit.flags.length > 0 && (
            <div className="mb-4">
              <h3 className="mb-1 text-[13px] font-semibold text-secondary">Readings to verify against source</h3>
              <ul className="space-y-1 text-sm text-secondary">
                {propertyAudit.flags.map((f) => (
                  <li key={`${f.location}-${f.metric}-${f.date}`}>
                    <span className="text-primary">{f.location}</span> {f.metric} {f.value} on {fmtDate(f.date)}: {f.note}
                  </li>
                ))}
              </ul>
            </div>
          )}

          {propertyAudit.duplicates.length > 0 && (
            <div className="mb-4">
              <h3 className="mb-1 text-[13px] font-semibold text-secondary">Duplicate snapshots ({propertyAudit.duplicates.length})</h3>
              <p className="mb-1 text-xs text-tertiary">
                Identical values recorded under two dates within three weeks, from a re-import. Every consumer reads the newest row only, so scores
                are unaffected; the extra rows just double up on charts.
              </p>
              <ul className="space-y-1 text-sm text-secondary">
                {propertyAudit.duplicates.slice(0, 8).map((d) => (
                  <li key={`${d.location}-${d.metric}`}>
                    <span className="text-primary">{d.location}</span> {d.metric} = {d.value} on {d.dates.map(fmtDate).join(' and ')}
                  </li>
                ))}
                {propertyAudit.duplicates.length > 8 && <li className="text-tertiary">…and {propertyAudit.duplicates.length - 8} more</li>}
              </ul>
            </div>
          )}

          {propertyAudit.missingFromLatest.length > 0 && (
            <div>
              <h3 className="mb-1 text-[13px] font-semibold text-secondary">Missing from the newest snapshot</h3>
              <p className="text-sm text-secondary">
                {propertyAudit.missingFromLatest.join(', ')}: carried forward from an older date, so these sit beside fresher metrics in any
                comparison.
              </p>
            </div>
          )}

          {propertyAudit.flags.length === 0 && propertyAudit.duplicates.length === 0 && propertyAudit.missingFromLatest.length === 0 && (
            <p className="text-sm text-secondary">No anomalies detected.</p>
          )}
        </Card>
      )}

      <Card>
        <SectionTitle>Score snapshots</SectionTitle>
        <p className="text-sm text-secondary">
          {data.scoreHistory.length
            ? `${data.scoreHistory.length} daily snapshots recorded, latest ${fmtDate(data.scoreHistory[data.scoreHistory.length - 1].score_date)}.`
            : 'No daily score snapshots yet. They appear once the /api/snapshot cron has run (see web/README).'}
        </p>
      </Card>

      <DataStamp fetchedAt={data.fetchedAt} />
    </div>
  );
}
