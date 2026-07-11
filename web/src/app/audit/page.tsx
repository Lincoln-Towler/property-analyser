import { getSiteData } from '@/lib/data';
import { buildAudit } from '@/lib/scoring/engine';
import { INDICATORS_CONFIG, STALE_AFTER_DAYS } from '@/lib/scoring/config';
import { SetupNotice } from '@/components/SetupNotice';

export const revalidate = 3600;

export default async function AuditPage() {
  const data = await getSiteData();
  if (!data) return <SetupNotice />;

  const audit = buildAudit(data.series, new Date());

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
