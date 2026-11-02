import { getBtcData } from '@/lib/btc';
import {
  buildLadder,
  CATEGORY_LABEL,
  groupFlags,
  levelSignalClass,
  scoreCeiling,
  type FlagGroup,
} from '@/lib/btc-view';
import { fmtDate, fmtDateTimeUtc, fmtUsd } from '@/lib/format';
import { BtcHistoryChart } from '@/components/BtcChart';
import { SetupNotice } from '@/components/SetupNotice';
import { Card, DataStamp, PageHeader, SectionTitle } from '@/components/ui';

export const revalidate = 3600;

function scoreColor(score: number): string {
  if (score >= 7.5) return 'text-emerald-400';
  if (score >= 6) return 'text-green-400';
  if (score >= 5) return 'text-lime-400';
  if (score >= 4) return 'text-amber-400';
  if (score >= 2.5) return 'text-orange-400';
  return 'text-red-400';
}

const signed = (n: number) => `${n > 0 ? '+' : n < 0 ? '−' : ''}${Math.abs(n).toFixed(1)}`;

function CategorySummary({ group, maxAbs }: { group: FlagGroup; maxAbs: number }) {
  const width = (v: number) => `${(Math.abs(v) / maxAbs) * 50}%`;
  return (
    <div className="rounded-xl border border-slate-800 bg-slate-900/60 p-4">
      <div className="flex items-baseline justify-between">
        <p className="text-sm font-medium">{CATEGORY_LABEL[group.category] ?? group.category}</p>
        <p className={`text-2xl font-semibold tabular-nums ${group.net < 0 ? 'text-red-400' : 'text-emerald-400'}`}>
          {signed(group.net)}
        </p>
      </div>
      {/* Diverging bar: bearish weight grows left of centre, bullish right. */}
      <div className="relative mt-3 h-2 rounded-full bg-slate-800" aria-hidden>
        <div className="absolute inset-y-0 right-1/2 rounded-l-full bg-red-500/80" style={{ width: width(group.bearishSum) }} />
        <div className="absolute inset-y-0 left-1/2 rounded-r-full bg-emerald-500/80" style={{ width: width(group.bullishSum) }} />
        <div className="absolute inset-y-[-2px] left-1/2 w-px bg-slate-500" />
      </div>
      <p className="mt-2 text-xs text-slate-400">
        {group.bearishCount} bearish ({signed(group.bearishSum)}) · {group.bullishCount} bullish ({signed(group.bullishSum)})
      </p>
    </div>
  );
}

export default async function BtcPage() {
  const data = await getBtcData();
  if (!data) return <SetupNotice />;

  const { levels, flags, latest, daily } = data;
  const groups = groupFlags(flags);
  const inactive = flags.filter((f) => !f.is_active);
  const modifierSum = Math.round(groups.reduce((s, g) => s + g.net, 0) * 10) / 10;
  const activeCount = groups.reduce((s, g) => s + g.flags.length, 0);
  const ceiling = scoreCeiling(levels, modifierSum);
  const ladder = buildLadder(levels, latest?.price_usd ?? null, latest?.current_band ?? null);
  const maxAbs = Math.max(0.1, ...groups.flatMap((g) => [Math.abs(g.bearishSum), Math.abs(g.bullishSum)]));
  const priceIsIrrelevant = ceiling !== null && latest !== null && ceiling.signal === latest.signal;

  return (
    <div className="space-y-8">
      <PageHeader title="BTC signal">
        Key-level ladder and context flags maintained from research notes. Each morning the n8n brief
        logs the signal from <code className="rounded bg-slate-800 px-1 text-slate-300">get_btc_signal()</code>:
        the price band sets a base score, then every active flag nudges it up or down.
      </PageHeader>

      {latest && (
        <Card>
          <div className="flex flex-wrap items-baseline gap-x-6 gap-y-2">
            <span className={`text-3xl font-semibold tracking-tight ${scoreColor(latest.score)}`}>{latest.signal}</span>
            <span className="text-xl tabular-nums text-slate-200">
              {latest.score.toFixed(1)}
              <span className="text-sm text-slate-500">/10</span>
            </span>
            <span className="text-xl tabular-nums text-slate-200">{fmtUsd(latest.price_usd)}</span>
          </div>
          {latest.intensity && <p className="mt-2 text-sm text-slate-300">{latest.intensity}</p>}
          <dl className="mt-5 grid grid-cols-2 gap-4 text-sm sm:grid-cols-4">
            <div>
              <dt className="text-xs uppercase tracking-wider text-slate-500">Band</dt>
              <dd className="mt-0.5 text-slate-200">{latest.current_band ?? '—'}</dd>
            </div>
            <div>
              <dt className="text-xs uppercase tracking-wider text-slate-500">Resistance</dt>
              <dd className="mt-0.5 tabular-nums text-slate-200">
                {latest.resistance_level == null ? '—' : fmtUsd(latest.resistance_level)}
                {latest.pct_to_resistance != null && <span className="text-slate-500"> +{latest.pct_to_resistance}%</span>}
              </dd>
            </div>
            <div>
              <dt className="text-xs uppercase tracking-wider text-slate-500">Support</dt>
              <dd className="mt-0.5 tabular-nums text-slate-200">
                {latest.support_level == null ? '—' : fmtUsd(latest.support_level)}
                {latest.pct_to_support != null && <span className="text-slate-500"> −{latest.pct_to_support}%</span>}
              </dd>
            </div>
            <div>
              <dt className="text-xs uppercase tracking-wider text-slate-500">Logged</dt>
              <dd className="mt-0.5 tabular-nums text-slate-200">{fmtDateTimeUtc(latest.logged_at)}</dd>
            </div>
          </dl>
          {latest.base_score != null && (
            <p className="mt-4 text-xs text-slate-500">
              Band base {latest.base_score.toFixed(1)} {signed(modifierSum)} from {activeCount} active flags, clamped 1–10.
            </p>
          )}
          {priceIsIrrelevant && ceiling && (
            <p className="mt-4 rounded-lg border border-amber-800/60 bg-amber-950/30 px-3 py-2 text-sm text-amber-100">
              <strong>Score ceiling {ceiling.score.toFixed(1)}.</strong> With the flags netting {signed(modifierSum)}, even
              the deepest-value band (base {Math.max(...levels.map((l) => l.base_score)).toFixed(1)}) only reaches{' '}
              {ceiling.score.toFixed(1)} — so the signal reads <em>{ceiling.signal}</em> at every price. Retiring flags that
              no longer apply would let price move the signal again.
            </p>
          )}
        </Card>
      )}

      <Card>
        <SectionTitle right={<span className="text-xs text-slate-500">shaded = current band · dashed = key levels</span>}>
          Price vs key levels
        </SectionTitle>
        {daily.length > 1 ? (
          <BtcHistoryChart
            points={daily.map((d) => ({ date: d.date, price: d.price }))}
            levels={levels.map((l) => l.level_usd)}
            support={latest?.support_level ?? null}
            resistance={latest?.resistance_level ?? null}
          />
        ) : (
          <p className="text-sm text-slate-500">Not enough log entries to chart yet.</p>
        )}
      </Card>

      <section>
        <SectionTitle>Key-level ladder</SectionTitle>
        <ol className="relative ml-2 border-l border-slate-700">
          {ladder.map((row) =>
            row.kind === 'price' ? (
              <li key="price" className="relative py-3 pl-6">
                <span className="absolute -left-[9px] top-1/2 h-4 w-4 -translate-y-1/2 rounded-full border-2 border-sky-300 bg-sky-500" aria-hidden />
                <div className="flex flex-wrap items-baseline gap-x-3 rounded-lg border border-sky-700/70 bg-sky-950/40 px-3 py-2">
                  <span className="text-lg font-semibold tabular-nums text-sky-200">{fmtUsd(row.price)}</span>
                  <span className="text-sm text-sky-300">current price</span>
                </div>
              </li>
            ) : (
              <li key={`${row.level.level_usd}-${row.level.label}`} className="relative py-2 pl-6">
                <span
                  className={`absolute -left-[5px] top-[18px] h-2.5 w-2.5 rounded-full ${row.isCurrentBand ? 'bg-amber-400' : 'bg-slate-600'}`}
                  aria-hidden
                />
                <div
                  className={`grid grid-cols-[auto_1fr] items-baseline gap-x-4 gap-y-1 rounded-lg px-3 py-1.5 sm:grid-cols-[6.5rem_1fr_auto_auto] ${
                    row.isCurrentBand ? 'bg-amber-950/30 ring-1 ring-amber-800/60' : ''
                  }`}
                >
                  <span className="font-medium tabular-nums">
                    {row.level.emoji ? `${row.level.emoji} ` : ''}
                    {fmtUsd(row.level.level_usd)}
                  </span>
                  <div className="min-w-0">
                    <p className="text-sm">
                      {row.level.label}
                      {row.isCurrentBand && <span className="ml-2 text-xs text-amber-400">current band</span>}
                    </p>
                    {row.level.notes && (
                      <details className="text-xs text-slate-400">
                        <summary className="text-slate-500 hover:text-slate-300">
                          <span className="chevron inline-block">›</span> notes
                        </summary>
                        <p className="mt-1 max-w-xl whitespace-pre-wrap">{row.level.notes}</p>
                      </details>
                    )}
                  </div>
                  <span className={`col-start-2 w-fit rounded px-2 py-0.5 text-xs ring-1 sm:col-start-auto ${levelSignalClass(row.level.signal)}`}>
                    {row.level.signal.replaceAll('_', ' ')} · {row.level.base_score.toFixed(1)}
                  </span>
                  <span className="col-start-2 text-xs tabular-nums text-slate-500 sm:col-start-auto sm:w-14 sm:text-right">
                    {row.distancePct === null ? '' : `${row.distancePct > 0 ? '+' : ''}${row.distancePct}%`}
                  </span>
                </div>
              </li>
            ),
          )}
        </ol>
      </section>

      <section className="space-y-3">
        <SectionTitle right={<span className="text-sm tabular-nums text-slate-400">net {signed(modifierSum)}</span>}>
          What&apos;s moving the score · {activeCount} active flags
        </SectionTitle>
        <div className="grid gap-3 sm:grid-cols-3">
          {groups.map((g) => (
            <CategorySummary key={g.category} group={g} maxAbs={maxAbs} />
          ))}
        </div>

        <div className="space-y-2">
          {groups.map((g) => (
            <details key={g.category} className="group rounded-xl border border-slate-800 bg-slate-900/60">
              <summary className="flex items-center justify-between gap-3 px-4 py-3">
                <span className="flex items-center gap-2 font-medium">
                  <span className="chevron inline-block text-slate-500">›</span>
                  {CATEGORY_LABEL[g.category] ?? g.category}
                  <span className="text-sm font-normal text-slate-500">{g.flags.length} flags</span>
                </span>
                <span className={`tabular-nums ${g.net < 0 ? 'text-red-400' : 'text-emerald-400'}`}>{signed(g.net)}</span>
              </summary>
              <ul className="divide-y divide-slate-800 border-t border-slate-800">
                {g.flags.map((f) => (
                  <li key={f.flag_key} className="grid grid-cols-[3rem_1fr] gap-x-3 px-4 py-3 sm:grid-cols-[3rem_1fr_auto]">
                    <span className={`tabular-nums font-medium ${f.score_modifier < 0 ? 'text-red-400' : f.score_modifier > 0 ? 'text-emerald-400' : 'text-slate-400'}`}>
                      {signed(f.score_modifier)}
                    </span>
                    <div className="min-w-0">
                      <p className="text-sm">{f.flag_label}</p>
                      {f.current_value && (
                        <p className="mt-0.5 line-clamp-2 text-xs text-slate-400" title={f.current_value}>
                          {f.current_value}
                        </p>
                      )}
                    </div>
                    <span className="col-start-2 whitespace-nowrap text-xs tabular-nums text-slate-500 sm:col-start-auto">
                      {fmtDate(f.updated_at)}
                    </span>
                  </li>
                ))}
              </ul>
            </details>
          ))}
          {inactive.length > 0 && (
            <details className="rounded-xl border border-slate-800/60 px-4 py-3 text-sm text-slate-400">
              <summary className="flex items-center gap-2">
                <span className="chevron inline-block text-slate-500">›</span>
                {inactive.length} inactive flag{inactive.length > 1 ? 's' : ''}
              </summary>
              <ul className="mt-2 list-inside list-disc space-y-1 text-xs">
                {inactive.map((f) => (
                  <li key={f.flag_key}>
                    {f.flag_label}{' '}
                    <span className="text-slate-500">
                      ({signed(f.score_modifier)}, off since {fmtDate(f.updated_at)})
                    </span>
                  </li>
                ))}
              </ul>
            </details>
          )}
        </div>
      </section>

      <DataStamp fetchedAt={data.fetchedAt} />
    </div>
  );
}
