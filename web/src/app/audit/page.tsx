import { getSiteData } from '@/lib/data';
import { getBtcData } from '@/lib/btc';
import { buildFeedHealth } from '@/lib/feed-health';
import { buildAudit } from '@/lib/scoring/engine';
import { EXTRA_INDICATOR_NAMES, INDICATORS_CONFIG, STALE_AFTER_DAYS } from '@/lib/scoring/config';
import { auditPropertyData } from '@/lib/property-audit';
import { groupByPublisher } from '@/lib/sources';
import { fmtDate, fmtIndicator } from '@/lib/format';
import { SetupNotice } from '@/components/SetupNotice';
import { DataStamp, PageHeader } from '@/components/ui';

export const revalidate = 3600;

function AgeBadge({ days }: { days: number | null }) {
  if (days === null) return <span className="text-xs text-slate-500">no data</span>;
  const tone =
    days > STALE_AFTER_DAYS
      ? 'bg-red-500/15 text-red-300 ring-red-500/30'
      : days > 35
        ? 'bg-amber-500/15 text-amber-300 ring-amber-500/30'
        : 'bg-emerald-500/15 text-emerald-300 ring-emerald-500/30';
  return (
    <span className={`inline-block whitespace-nowrap rounded px-2 py-0.5 text-xs tabular-nums ring-1 ${tone}`}>
      {days}d{days > STALE_AFTER_DAYS ? ' · stale' : ''}
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
    <div className="space-y-6">
      <PageHeader title="Data audit">
        Freshness and coverage per indicator. Anything older than {STALE_AFTER_DAYS} days is flagged — a
        stale value still feeds the weighted score, so this is the first place to look when the headline
        number seems off. Sources are grouped by publisher; <em>n8n feed</em> rows come from{' '}
        <code className="rounded bg-slate-800 px-1">economic_indicators_history</code>, everything else was
        entered manually.
      </PageHeader>

      <section
        className={`rounded-xl border p-5 text-sm ${health.healthy ? 'border-slate-800 bg-slate-900/60' : 'border-red-700/60 bg-red-950/30'}`}
      >
        <h2 className="mb-1 font-medium">
          {health.healthy ? 'Automation health: everything ran on schedule' : 'Automation health: something needs a look'}
        </h2>
        <p className="mb-3 text-xs text-slate-400">
          The feeds run by themselves, so this is where a stopped one shows up.
        </p>
        <ul className="divide-y divide-slate-800">
          {health.jobs.map((j) => (
            <li key={j.name} className="grid grid-cols-[1fr_auto] items-baseline gap-x-4 py-2 sm:grid-cols-[1fr_9rem_8rem_auto]">
              <span className="font-medium">{j.name}</span>
              <span className={`whitespace-nowrap text-xs font-medium sm:order-last ${j.ok ? 'text-emerald-400' : 'text-red-300'}`}>
                {j.ok ? 'On schedule' : 'Late'}
              </span>
              <span className="text-xs text-slate-500">{j.schedule}</span>
              <span className="whitespace-nowrap text-xs tabular-nums text-slate-400">{fmtDate(j.latest)}</span>
            </li>
          ))}
        </ul>
        {health.late.length > 0 && (
          <div className="mt-3">
            <h3 className="mb-1 text-xs uppercase tracking-wide text-slate-500">Older than expected</h3>
            <ul className="space-y-1 text-slate-300">
              {health.late.map((l) => (
                <li key={l.indicator}>
                  <span className="font-medium">{l.displayName}</span> — {l.ageDays === null ? 'no data' : `latest reading ${fmtDate(l.latest)}`}
                  {l.manual ? ' (entered by hand, needs a manual update)' : ' (the feed may have stopped)'}
                </li>
              ))}
            </ul>
          </div>
        )}
        {health.failedVideos.length > 0 && (
          <div className="mt-3">
            <h3 className="mb-1 text-xs uppercase tracking-wide text-slate-500">Videos that couldn&apos;t be analysed</h3>
            <ul className="space-y-1 text-slate-300">
              {health.failedVideos.map((v) => (
                <li key={v.title}>
                  {v.title} <span className="text-slate-500">({v.attempts} of 3 attempts)</span>
                </li>
              ))}
            </ul>
          </div>
        )}
        {health.failedChanges.length > 0 && (
          <div className="mt-3">
            <h3 className="mb-1 text-xs uppercase tracking-wide text-slate-500">Flag changes that didn&apos;t apply</h3>
            <ul className="space-y-1 text-slate-300">
              {health.failedChanges.map((c) => (
                <li key={c.flag_key}>
                  {c.flag_key}: {c.note}
                </li>
              ))}
            </ul>
          </div>
        )}
      </section>

      {frozen.length > 0 && (
        <section className="rounded-xl border border-amber-700/60 bg-amber-950/30 p-5 text-sm">
          <h2 className="mb-1 font-medium">
            {frozenWeightPct.toFixed(0)}% of the score is riding on data that can&apos;t move
          </h2>
          <p className="mb-3 text-xs text-slate-400">
            These indicators still carry their full weight, but are stale or have too few recent
            readings to produce a trend — so they hold the score in place rather than informing
            it. Nothing is excluded from scoring; the fix is more data, not different maths.
          </p>
          <ul className="space-y-1 text-slate-300">
            {frozen.map((a) => (
              <li key={a.indicator}>
                <span className="font-medium">{a.display_name}</span> — carries{' '}
                <span className="tabular-nums">{a.weight}</span>/135 (
                {a.weight_pct.toFixed(1)}% of the score),{' '}
                {a.days_old === null ? 'no data' : `${a.days_old} days old`},{' '}
                {a.point_count} {a.point_count === 1 ? 'reading' : 'readings'}
                {!a.can_trend && ', cannot trend'}
                {!a.can_measure_volatility && ', volatility unmeasurable'}
                {a.cadence_blocks_trend && (
                  <span className="text-amber-300">
                    {' '}
                    — {a.expected_frequency} data can never fill a 3-month trend window
                    {a.cadence_blocks_volatility ? ' or a 6-month volatility window' : ''}, so
                    backfilling will not change this
                  </span>
                )}
              </li>
            ))}
          </ul>
        </section>
      )}

      <div className="overflow-x-auto rounded-xl border border-slate-800">
        <table className="w-full min-w-[640px] text-sm">
          <thead className="bg-slate-900 text-left text-xs uppercase tracking-wide text-slate-400">
            <tr>
              <th className="px-4 py-3">Indicator</th>
              <th className="px-4 py-3">Weight</th>
              <th className="px-4 py-3">Latest value</th>
              <th className="px-4 py-3">Latest date</th>
              <th className="px-4 py-3">Age</th>
              <th className="px-4 py-3">Points (24 mo)</th>
              <th className="px-4 py-3">Sources</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-800">
            {audit.map((row) => {
              const publishers = groupByPublisher(row.sources);
              return (
                <tr key={row.indicator} className={row.stale ? 'bg-red-950/20' : ''}>
                  <td className="px-4 py-3 font-medium">
                    {INDICATORS_CONFIG[row.indicator]?.display_name ?? EXTRA_INDICATOR_NAMES[row.indicator] ?? row.display_name}
                  </td>
                  <td className="px-4 py-3 tabular-nums text-slate-400">
                    {INDICATORS_CONFIG[row.indicator]?.weight ?? '—'}
                  </td>
                  <td className="whitespace-nowrap px-4 py-3 tabular-nums">
                    {fmtIndicator(row.latest_value, INDICATORS_CONFIG[row.indicator]?.unit ?? '')}
                  </td>
                  <td className="whitespace-nowrap px-4 py-3 tabular-nums">{fmtDate(row.latest_date)}</td>
                  <td className="px-4 py-3">
                    <AgeBadge days={row.days_old} />
                  </td>
                  <td className="px-4 py-3 tabular-nums">{row.point_count}</td>
                  <td className="px-4 py-3">
                    {publishers.length ? (
                      <div className="flex flex-wrap gap-1">
                        {publishers.map(([pub, n]) => (
                          <span key={pub} className="whitespace-nowrap rounded bg-slate-800 px-1.5 py-0.5 text-xs text-slate-300">
                            {pub} <span className="text-slate-500">×{n}</span>
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

      {(data.adjustments.dropped.length > 0 || data.adjustments.annualized.length > 0) && (
        <section className="rounded-xl border border-amber-700/60 bg-amber-950/30 p-5 text-sm text-slate-200">
          <h2 className="mb-2 font-medium">Data adjustments applied</h2>
          {data.adjustments.dropped.length > 0 && (
            <div className="mb-2">
              <p className="mb-1">
                Implausible values excluded from charts and scoring (likely a broken feed —
                check the n8n workflow):
              </p>
              <ul className="list-disc pl-5 text-slate-300">
                {data.adjustments.dropped.map((d) => (
                  <li key={`${d.indicator}-${d.date}`}>
                    {d.indicator}: {d.value} on {fmtDate(d.date)}
                  </li>
                ))}
              </ul>
            </div>
          )}
          {data.adjustments.annualized.map((name) => (
            <p key={name}>
              <strong>{name}</strong>: feed delivers monthly readings — values shown and scored
              as a trailing 12-month annualised rate to match the annual thresholds.
            </p>
          ))}
        </section>
      )}

      {data.propertyData.length > 0 && (
        <section className="rounded-xl border border-slate-800 bg-slate-900/60 p-5 text-sm">
          <h2 className="mb-1 font-medium">Property data quality</h2>
          <p className="mb-3 text-xs text-slate-400">
            {propertyAudit.locations} locations, newest snapshot {fmtDate(propertyAudit.latestDate)}.
            Nothing here is excluded from charts or scoring — these are readings worth a second
            look, not errors. Locations mix capitals, suburbs and regional towns, so an extreme
            value is often a narrow sub-market sample rather than a bad number.
          </p>

          {propertyAudit.flags.length > 0 && (
            <div className="mb-4">
              <h3 className="mb-1 text-xs uppercase tracking-wide text-slate-500">
                Readings to verify against source
              </h3>
              <ul className="space-y-1 text-slate-300">
                {propertyAudit.flags.map((f) => (
                  <li key={`${f.location}-${f.metric}-${f.date}`}>
                    <span className="font-medium">{f.location}</span> {f.metric}{' '}
                    <span className="tabular-nums">{f.value}</span> on {fmtDate(f.date)} — {f.note}
                  </li>
                ))}
              </ul>
            </div>
          )}

          {propertyAudit.duplicates.length > 0 && (
            <div className="mb-4">
              <h3 className="mb-1 text-xs uppercase tracking-wide text-slate-500">
                Duplicate snapshots ({propertyAudit.duplicates.length})
              </h3>
              <p className="mb-1 text-xs text-slate-400">
                Identical values recorded under two dates within three weeks — a re-import. Every
                consumer reads the newest row only, so scores are unaffected; the extra rows just
                double up on charts.
              </p>
              <ul className="space-y-1 text-slate-300">
                {propertyAudit.duplicates.slice(0, 8).map((d) => (
                  <li key={`${d.location}-${d.metric}`}>
                    <span className="font-medium">{d.location}</span> {d.metric} ={' '}
                    <span className="tabular-nums">{d.value}</span> on {d.dates.map(fmtDate).join(' and ')}
                  </li>
                ))}
                {propertyAudit.duplicates.length > 8 && (
                  <li className="text-slate-500">
                    …and {propertyAudit.duplicates.length - 8} more
                  </li>
                )}
              </ul>
            </div>
          )}

          {propertyAudit.missingFromLatest.length > 0 && (
            <div>
              <h3 className="mb-1 text-xs uppercase tracking-wide text-slate-500">
                Missing from the newest snapshot
              </h3>
              <p className="text-slate-300">
                {propertyAudit.missingFromLatest.join(', ')} — carried forward from an older date,
                so these sit beside fresher metrics in any comparison.
              </p>
            </div>
          )}

          {propertyAudit.flags.length === 0 &&
            propertyAudit.duplicates.length === 0 &&
            propertyAudit.missingFromLatest.length === 0 && (
              <p className="text-slate-300">No anomalies detected.</p>
            )}
        </section>
      )}

      <section className="rounded-xl border border-slate-800 bg-slate-900/60 p-5 text-sm text-slate-300">
        <h2 className="mb-2 font-medium">Score snapshots</h2>
        <p>
          {data.scoreHistory.length
            ? `${data.scoreHistory.length} daily snapshots recorded, latest ${fmtDate(data.scoreHistory[data.scoreHistory.length - 1].score_date)}.`
            : 'No daily score snapshots yet — they appear once the /api/snapshot cron has run (see web/README).'}
        </p>
      </section>

      <DataStamp fetchedAt={data.fetchedAt} />
    </div>
  );
}
