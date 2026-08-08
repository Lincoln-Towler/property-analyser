import { getSiteData } from '@/lib/data';
import { buildAudit } from '@/lib/scoring/engine';
import { INDICATORS_CONFIG, STALE_AFTER_DAYS } from '@/lib/scoring/config';
import { auditPropertyData } from '@/lib/property-audit';
import { SetupNotice } from '@/components/SetupNotice';

export const revalidate = 3600;

export default async function AuditPage() {
  const data = await getSiteData();
  if (!data) return <SetupNotice />;

  const audit = buildAudit(data.series, new Date());
  const propertyAudit = auditPropertyData(data.propertyData);

  return (
    <div className="space-y-6">
      <header>
        <h1 className="text-xl font-semibold">Data Audit</h1>
        <p className="mt-1 text-sm text-slate-400">
          Freshness and coverage per indicator. Anything older than {STALE_AFTER_DAYS} days is
          flagged — a stale value still feeds the weighted score, so this is the first place to
          look when the headline number seems off. The n8n feed writes to{' '}
          <code className="rounded bg-slate-800 px-1">economic_indicators_history</code>; rows
          labelled <em>manual</em> come from the current table.
        </p>
      </header>

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
            {audit.map((row) => (
              <tr key={row.indicator} className={row.stale ? 'bg-red-950/20' : ''}>
                <td className="px-4 py-3">
                  {row.display_name}
                  {row.stale && <span className="ml-2 text-xs text-red-400">stale</span>}
                </td>
                <td className="px-4 py-3 tabular-nums text-slate-400">
                  {INDICATORS_CONFIG[row.indicator]?.weight ?? '—'}
                </td>
                <td className="px-4 py-3 tabular-nums">
                  {row.latest_value === null
                    ? '—'
                    : Math.abs(row.latest_value) >= 1000
                      ? row.latest_value.toLocaleString('en-US')
                      : row.latest_value}
                </td>
                <td className="px-4 py-3 tabular-nums">{row.latest_date ?? '—'}</td>
                <td className="px-4 py-3 tabular-nums">
                  {row.days_old === null ? 'no data' : `${row.days_old}d`}
                </td>
                <td className="px-4 py-3 tabular-nums">{row.point_count}</td>
                <td className="px-4 py-3 text-xs text-slate-400">
                  {Object.entries(row.sources)
                    .map(([s, n]) => `${s}: ${n}`)
                    .join(', ') || '—'}
                </td>
              </tr>
            ))}
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
                    {d.indicator}: {d.value} on {d.date}
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
            {propertyAudit.locations} locations, newest snapshot {propertyAudit.latestDate ?? '—'}.
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
                    <span className="tabular-nums">{f.value}</span> on {f.date} — {f.note}
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
                    <span className="tabular-nums">{d.value}</span> on {d.dates.join(' and ')}
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
            ? `${data.scoreHistory.length} daily snapshot(s) recorded, latest ${data.scoreHistory[data.scoreHistory.length - 1].score_date}.`
            : 'No daily score snapshots yet — they appear once the /api/snapshot cron has run (see web/README).'}
        </p>
      </section>

      <p className="text-xs text-slate-500">Data as of {data.fetchedAt.slice(0, 16).replace('T', ' ')} UTC</p>
    </div>
  );
}
