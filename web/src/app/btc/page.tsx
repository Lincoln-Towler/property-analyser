import { getBtcData } from '@/lib/btc';
import { BtcHistoryChart } from '@/components/BtcChart';
import { SetupNotice } from '@/components/SetupNotice';

export const revalidate = 3600;

const usd = (n: number) => `$${n.toLocaleString('en-US')}`;

function scoreColor(score: number): string {
  if (score >= 7.5) return 'text-emerald-400';
  if (score >= 6) return 'text-green-400';
  if (score >= 5) return 'text-lime-400';
  if (score >= 4) return 'text-amber-400';
  if (score >= 2.5) return 'text-orange-400';
  return 'text-red-400';
}

const CATEGORY_STYLE: Record<string, string> = {
  MACRO: 'bg-sky-950 text-sky-300',
  ON_CHAIN: 'bg-violet-950 text-violet-300',
  CYCLE: 'bg-amber-950 text-amber-300',
};

export default async function BtcPage() {
  const data = await getBtcData();
  if (!data) return <SetupNotice />;

  const { levels, flags, latest, daily } = data;
  const activeFlags = flags.filter((f) => f.is_active);
  const inactiveFlags = flags.filter((f) => !f.is_active);
  const modifierSum = activeFlags.reduce((s, f) => s + f.score_modifier, 0);

  return (
    <div className="space-y-6">
      <header>
        <h1 className="text-xl font-semibold">BTC Signal</h1>
        <p className="mt-1 text-sm text-slate-400">
          Key-level ladder and context flags maintained manually from research notes; the daily
          signal is computed by <code className="rounded bg-slate-800 px-1">get_btc_signal()</code>{' '}
          and logged each morning by the n8n brief. Base score comes from the price band, then
          active context flags drag it up or down.
        </p>
      </header>

      {latest && (
        <section className="rounded-xl border border-slate-800 bg-slate-900/60 p-5">
          <div className="flex flex-wrap items-baseline gap-x-6 gap-y-2">
            <span className={`text-2xl font-semibold ${scoreColor(latest.score)}`}>
              {latest.signal}
            </span>
            <span className="text-lg tabular-nums text-slate-200">
              {latest.score.toFixed(1)}<span className="text-sm text-slate-500">/10</span>
            </span>
            <span className="text-lg tabular-nums text-slate-200">{usd(latest.price_usd)}</span>
          </div>
          {latest.intensity && <p className="mt-2 text-sm text-slate-300">{latest.intensity}</p>}
          <dl className="mt-4 grid grid-cols-2 gap-3 text-sm sm:grid-cols-4">
            <div>
              <dt className="text-xs uppercase tracking-wide text-slate-500">Band</dt>
              <dd className="mt-0.5 text-slate-200">{latest.current_band ?? '—'}</dd>
            </div>
            <div>
              <dt className="text-xs uppercase tracking-wide text-slate-500">Resistance</dt>
              <dd className="mt-0.5 tabular-nums text-slate-200">
                {latest.resistance_level == null
                  ? '—'
                  : `${usd(latest.resistance_level)}${latest.pct_to_resistance == null ? '' : ` (+${latest.pct_to_resistance}%)`}`}
              </dd>
            </div>
            <div>
              <dt className="text-xs uppercase tracking-wide text-slate-500">Support</dt>
              <dd className="mt-0.5 tabular-nums text-slate-200">
                {latest.support_level == null
                  ? '—'
                  : `${usd(latest.support_level)}${latest.pct_to_support == null ? '' : ` (−${latest.pct_to_support}%)`}`}
              </dd>
            </div>
            <div>
              <dt className="text-xs uppercase tracking-wide text-slate-500">Logged</dt>
              <dd className="mt-0.5 tabular-nums text-slate-200">
                {latest.logged_at.slice(0, 16).replace('T', ' ')} UTC
              </dd>
            </div>
          </dl>
          {latest.base_score != null && latest.modifier_total != null && (
            <p className="mt-3 text-xs text-slate-500">
              band base {latest.base_score.toFixed(1)} {latest.modifier_total >= 0 ? '+' : '−'}{' '}
              {Math.abs(latest.modifier_total).toFixed(1)} from context flags, clamped 1–10
            </p>
          )}
        </section>
      )}

      <section className="rounded-xl border border-slate-800 bg-slate-900/60 p-5">
        <h2 className="mb-3 text-sm font-medium text-slate-300">
          Price vs signal score — daily log
        </h2>
        {daily.length > 1 ? (
          <BtcHistoryChart points={daily} />
        ) : (
          <p className="text-sm text-slate-500">Not enough log entries to chart yet.</p>
        )}
      </section>

      <section>
        <h2 className="mb-2 text-sm font-medium text-slate-300">Key levels</h2>
        <div className="overflow-x-auto rounded-xl border border-slate-800">
          <table className="w-full min-w-[560px] text-sm">
            <thead className="bg-slate-900 text-left text-xs uppercase tracking-wide text-slate-400">
              <tr>
                <th className="px-4 py-3">Level</th>
                <th className="px-4 py-3">Zone</th>
                <th className="px-4 py-3">Signal</th>
                <th className="px-4 py-3">Base score</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800">
              {levels.map((lvl) => {
                const current = latest?.current_band === lvl.label;
                return (
                  <tr key={`${lvl.level_usd}-${lvl.label}`} className={current ? 'bg-amber-950/20' : ''}>
                    <td className="px-4 py-3 tabular-nums">
                      {lvl.emoji ? `${lvl.emoji} ` : ''}
                      {usd(lvl.level_usd)}
                      {current && <span className="ml-2 text-xs text-amber-400">← price is here</span>}
                    </td>
                    <td className="px-4 py-3">
                      {lvl.label}
                      {lvl.notes && (
                        <details className="mt-1 text-xs text-slate-400">
                          <summary className="cursor-pointer text-slate-500">notes</summary>
                          <p className="mt-1 max-w-xl whitespace-pre-wrap">{lvl.notes}</p>
                        </details>
                      )}
                    </td>
                    <td className="px-4 py-3">{lvl.signal.replaceAll('_', ' ')}</td>
                    <td className={`px-4 py-3 tabular-nums ${scoreColor(lvl.base_score)}`}>
                      {lvl.base_score.toFixed(1)}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </section>

      <section>
        <h2 className="mb-2 text-sm font-medium text-slate-300">
          Active context flags{' '}
          <span className="tabular-nums text-slate-500">
            ({activeFlags.length}, net {modifierSum >= 0 ? '+' : ''}
            {modifierSum.toFixed(1)})
          </span>
        </h2>
        <div className="overflow-x-auto rounded-xl border border-slate-800">
          <table className="w-full min-w-[560px] text-sm">
            <thead className="bg-slate-900 text-left text-xs uppercase tracking-wide text-slate-400">
              <tr>
                <th className="px-4 py-3">Modifier</th>
                <th className="px-4 py-3">Flag</th>
                <th className="px-4 py-3">Category</th>
                <th className="px-4 py-3">Updated</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800">
              {activeFlags.map((f) => (
                <tr key={f.flag_key}>
                  <td
                    className={`px-4 py-3 tabular-nums ${f.score_modifier < 0 ? 'text-red-400' : 'text-green-400'}`}
                  >
                    {f.score_modifier > 0 ? '+' : ''}
                    {f.score_modifier.toFixed(1)}
                  </td>
                  <td className="px-4 py-3">
                    {f.flag_label}
                    {f.current_value && (
                      <p className="mt-1 max-w-xl text-xs text-slate-400">{f.current_value}</p>
                    )}
                  </td>
                  <td className="px-4 py-3">
                    <span
                      className={`rounded px-2 py-0.5 text-xs ${CATEGORY_STYLE[f.category ?? ''] ?? 'bg-slate-800 text-slate-300'}`}
                    >
                      {f.category ?? '—'}
                    </span>
                  </td>
                  <td className="px-4 py-3 tabular-nums text-xs text-slate-400">
                    {f.updated_at ? f.updated_at.slice(0, 10) : '—'}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        {inactiveFlags.length > 0 && (
          <details className="mt-2 text-sm text-slate-400">
            <summary className="cursor-pointer text-slate-500">
              {inactiveFlags.length} inactive flag(s)
            </summary>
            <ul className="mt-2 list-inside list-disc space-y-1 text-xs">
              {inactiveFlags.map((f) => (
                <li key={f.flag_key}>
                  {f.flag_label}{' '}
                  <span className="text-slate-500">
                    ({f.score_modifier > 0 ? '+' : ''}
                    {f.score_modifier.toFixed(1)}, off since{' '}
                    {f.updated_at ? f.updated_at.slice(0, 10) : '—'})
                  </span>
                </li>
              ))}
            </ul>
          </details>
        )}
      </section>

      <p className="text-xs text-slate-500">
        Data as of {data.fetchedAt.slice(0, 16).replace('T', ' ')} UTC
      </p>
    </div>
  );
}
