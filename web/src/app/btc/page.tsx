import { getBtcData } from '@/lib/btc';
import {
  BTC_MAX_ACTIVE_FLAGS,
  BTC_MODIFIER_CAP,
  btcSignalForScore,
  buildLadder,
  capModifiers,
  CATEGORY_LABEL,
  CHANGE_LABEL,
  changeName,
  effectiveScore,
  groupFlags,
  levelSignalClass,
  scoreCeiling,
  signalTextClass,
  stanceClass,
  type FlagGroup,
} from '@/lib/btc-view';
import { fmtAdelaideTime, fmtDate, fmtUsd, sentenceCase } from '@/lib/format';
import { BtcHistoryChart } from '@/components/BtcChart';
import { SetupNotice } from '@/components/SetupNotice';
import { InfoTip } from '@/components/InfoTip';
import { Card, DataStamp, Row, SectionTitle } from '@/components/ui';

export const revalidate = 3600;

const signed = (n: number) => `${n > 0 ? '+' : n < 0 ? '−' : ''}${Math.abs(n).toFixed(1)}`;
const modifierTone = (n: number) => (n < 0 ? 'text-danger' : n > 0 ? 'text-go' : 'text-tertiary');

function CategorySummary({ group, maxAbs }: { group: FlagGroup; maxAbs: number }) {
  const width = (v: number) => `${(Math.abs(v) / maxAbs) * 50}%`;
  return (
    <div className="rounded-lg bg-tile p-3">
      <div className="flex items-baseline justify-between">
        <p className="text-sm">{CATEGORY_LABEL[group.category] ?? group.category}</p>
        <p className={`text-[22px] font-light leading-none ${modifierTone(group.net)}`}>{signed(group.net)}</p>
      </div>
      {/* Diverging bar: bearish weight grows left of centre, bullish right. */}
      <div className="relative mt-2.5 h-1.5 rounded-full bg-surface" aria-hidden>
        <div className="absolute inset-y-0 right-1/2 rounded-l-full bg-danger" style={{ width: width(group.bearishSum) }} />
        <div className="absolute inset-y-0 left-1/2 rounded-r-full bg-go" style={{ width: width(group.bullishSum) }} />
      </div>
      <p className="mt-2 text-xs text-tertiary">
        {group.bearishCount} bearish ({signed(group.bearishSum)}), {group.bullishCount} bullish ({signed(group.bullishSum)})
      </p>
    </div>
  );
}

export default async function BtcPage() {
  const data = await getBtcData();
  if (!data) return <SetupNotice />;

  const { levels, flags, latest, daily, videos, changes } = data;
  const analysed = videos.filter((v) => v.status === 'analysed').slice(0, 3);
  const groups = groupFlags(flags);
  const inactive = flags.filter((f) => !f.is_active);
  const modifierSum = Math.round(groups.reduce((s, g) => s + g.net, 0) * 10) / 10;
  const cappedSum = capModifiers(modifierSum);
  const activeCount = groups.reduce((s, g) => s + g.flags.length, 0);
  // The last log row may predate today's flag changes (or the cap itself):
  // say what the next run at the same price would score.
  const flagsChanged = latest?.modifier_total != null && latest.modifier_total !== cappedSum;
  const projected = latest?.base_score != null ? effectiveScore(latest.base_score, modifierSum) : null;
  const ceiling = scoreCeiling(levels, modifierSum);
  const ladder = buildLadder(levels, latest?.price_usd ?? null, latest?.current_band ?? null);
  const maxAbs = Math.max(0.1, ...groups.flatMap((g) => [Math.abs(g.bearishSum), Math.abs(g.bullishSum)]));
  const priceIsIrrelevant = ceiling !== null && latest !== null && ceiling.signal === latest.signal;
  const readingTime = latest ? fmtAdelaideTime(latest.logged_at) : null;

  return (
    <div className="space-y-2.5">
      <h1 className="sr-only">BTC signal</h1>

      {latest && (
        <section className={`rounded-xl p-4 sm:p-5 ${levelSignalClass(latest.signal)}`}>
          <p className="text-[26px] font-semibold leading-tight tracking-tight">{sentenceCase(latest.signal)}</p>
          <p className="mt-1.5 text-[15px] font-medium">
            {latest.score.toFixed(1)} out of 10 at {fmtUsd(latest.price_usd)}
          </p>
          {latest.intensity && <p className="mt-1 text-sm">{latest.intensity}</p>}
          <p className="mt-1 text-xs">Reading from {readingTime}</p>
        </section>
      )}

      <div className="grid gap-2.5 md:grid-cols-2">
        {latest && (
          <Card>
            <SectionTitle>Why</SectionTitle>
            <dl>
              <Row label="Price band">{latest.current_band ?? '—'}</Row>
              {latest.base_score != null && <Row label="Band score">{latest.base_score.toFixed(1)}</Row>}
              {latest.modifier_total != null && (
                <Row label="Flags">
                  {signed(latest.modifier_total)}
                  {modifierSum !== cappedSum && !flagsChanged ? ` (net ${signed(modifierSum)}, capped at ±${BTC_MODIFIER_CAP})` : ''}
                </Row>
              )}
              <Row label="Resistance">
                {latest.resistance_level == null ? '—' : fmtUsd(latest.resistance_level)}
                {latest.pct_to_resistance != null && <span className="text-tertiary"> +{latest.pct_to_resistance}%</span>}
              </Row>
              <Row label="Support">
                {latest.support_level == null ? '—' : fmtUsd(latest.support_level)}
                {latest.pct_to_support != null && <span className="text-tertiary"> −{latest.pct_to_support}%</span>}
              </Row>
            </dl>
            {flagsChanged && projected !== null && (
              <p className="mt-2 text-xs text-caution">
                Flags have changed since this reading. They now net {signed(modifierSum)}
                {modifierSum !== cappedSum ? `, capped to ${signed(cappedSum)}` : ''}, so the next run at this price scores{' '}
                {projected.toFixed(1)} ({sentenceCase(btcSignalForScore(projected))}).
              </p>
            )}
            {priceIsIrrelevant && ceiling && (
              <p className="mt-2 text-xs text-caution">
                The best possible score right now is {ceiling.score.toFixed(1)}. With flags counting as {signed(cappedSum)}, the signal reads{' '}
                {sentenceCase(ceiling.signal)} at every price.
              </p>
            )}
          </Card>
        )}

        <Card>
          <SectionTitle right="Shaded: current band">Price and key levels</SectionTitle>
          {daily.length > 1 ? (
            <BtcHistoryChart
              points={daily.map((d) => ({ date: d.date, price: d.price }))}
              levels={levels.map((l) => l.level_usd)}
              support={latest?.support_level ?? null}
              resistance={latest?.resistance_level ?? null}
            />
          ) : (
            <p className="text-sm text-tertiary">Not enough readings to chart yet.</p>
          )}
        </Card>
      </div>

      <Card>
        <SectionTitle right={`Scores include flags at ${signed(cappedSum)}`}>Key levels</SectionTitle>
        <ol className="-mx-2">
          {ladder.map((row) =>
            row.kind === 'price' ? (
              <li key="price" className="my-0.5 flex items-baseline justify-between gap-3 rounded-lg bg-tile px-2 py-2">
                <span className="text-sm font-semibold">{fmtUsd(row.price)}</span>
                <span className="text-xs text-secondary">Price at {readingTime}</span>
              </li>
            ) : (
              <li
                key={`${row.level.level_usd}-${row.level.label}`}
                className="grid grid-cols-[1fr_auto] items-baseline gap-x-3 gap-y-0.5 px-2 py-1.5 sm:grid-cols-[5rem_minmax(0,1fr)_auto]"
              >
                <span className="text-sm font-medium">{fmtUsd(row.level.level_usd)}</span>
                <div className="col-span-2 row-start-2 min-w-0 sm:col-span-1 sm:col-start-2 sm:row-start-1">
                  <p className={`text-[13px] ${row.isCurrentBand ? 'text-primary' : 'text-secondary'}`}>{row.level.label}</p>
                  {row.level.notes && (
                    <details className="text-xs text-tertiary">
                      <summary className="hover:text-secondary">
                        <span className="chevron inline-block">›</span> Notes
                      </summary>
                      <p className="mt-1 max-w-xl whitespace-pre-wrap">{row.level.notes}</p>
                    </details>
                  )}
                </div>
                {(() => {
                  const now = effectiveScore(row.level.base_score, modifierSum);
                  const signal = btcSignalForScore(now);
                  return (
                    <span className="flex items-baseline justify-end gap-2 sm:col-start-3 sm:row-start-1">
                      <span className="text-xs text-tertiary">
                        {row.distancePct === null ? '' : `${row.distancePct > 0 ? '+' : ''}${row.distancePct}%`}
                      </span>
                      <InfoTip
                        label={`${sentenceCase(signal.split(' / ')[0])} ${now.toFixed(1)}`}
                        className={`whitespace-nowrap text-[13px] font-medium ${signalTextClass(signal)}`}
                      >
                        Band score {row.level.base_score.toFixed(1)}, {signed(cappedSum)} from flags, makes {now.toFixed(1)}.
                      </InfoTip>
                    </span>
                  );
                })()}
              </li>
            ),
          )}
        </ol>
      </Card>

      <Card>
        <SectionTitle
          right={
            <>
              {activeCount} of {BTC_MAX_ACTIVE_FLAGS} active, net {signed(modifierSum)}
              {modifierSum !== cappedSum && <>, counts as {signed(cappedSum)}</>}
            </>
          }
        >
          What&apos;s moving the score
        </SectionTitle>
        <div className="grid gap-2.5 sm:grid-cols-3">
          {groups.map((g) => (
            <CategorySummary key={g.category} group={g} maxAbs={maxAbs} />
          ))}
        </div>

        <div className="-mx-2 mt-3">
          {groups.map((g) => (
            <details key={g.category} className="group rounded-lg">
              <summary className="flex items-center justify-between gap-3 rounded-lg px-2 py-2 transition-colors hover:bg-tile group-open:bg-tile">
                <span className="flex items-baseline gap-2 text-sm">
                  <span className="chevron inline-block text-tertiary">›</span>
                  {CATEGORY_LABEL[g.category] ?? g.category}
                  <span className="text-xs text-tertiary">{g.flags.length} flags</span>
                </span>
                <span className={`text-sm font-medium ${modifierTone(g.net)}`}>{signed(g.net)}</span>
              </summary>
              <ul className="space-y-2 px-2 pb-2 pt-2">
                {g.flags.map((f) => (
                  <li key={f.flag_key} className="grid grid-cols-[2.5rem_1fr] gap-x-2 sm:grid-cols-[2.5rem_1fr_auto]">
                    <span className={`text-sm font-medium ${modifierTone(f.score_modifier)}`}>{signed(f.score_modifier)}</span>
                    <div className="min-w-0">
                      <p className="text-sm">{f.flag_label}</p>
                      {f.current_value && <p className="mt-0.5 text-xs text-tertiary">{f.current_value}</p>}
                    </div>
                    <span className="col-start-2 whitespace-nowrap text-xs text-tertiary sm:col-start-auto">{fmtDate(f.updated_at)}</span>
                  </li>
                ))}
              </ul>
            </details>
          ))}
          {inactive.length > 0 && (
            <details className="rounded-lg text-sm text-secondary">
              <summary className="flex items-baseline gap-2 rounded-lg px-2 py-2 transition-colors hover:bg-tile">
                <span className="chevron inline-block text-tertiary">›</span>
                {inactive.length} retired flag{inactive.length > 1 ? 's' : ''}
              </summary>
              <ul className="space-y-1 px-2 pb-2 pt-1 text-xs">
                {inactive.map((f) => (
                  <li key={f.flag_key}>
                    {f.flag_label}{' '}
                    <span className="text-tertiary">
                      ({signed(f.score_modifier)}, off since {fmtDate(f.updated_at)})
                    </span>
                  </li>
                ))}
              </ul>
            </details>
          )}
        </div>
      </Card>

      {analysed.length > 0 && (
        <Card>
          <SectionTitle right="AI-written summaries, not his words">Latest from Benjamin Cowen</SectionTitle>
          <div className="space-y-5">
            {analysed.map((v) => (
              <article key={v.video_id}>
                <div className="flex flex-wrap items-baseline justify-between gap-x-3 gap-y-1">
                  <h3 className="text-sm font-medium">
                    {v.url ? (
                      <a href={v.url} target="_blank" rel="noopener noreferrer" className="hover:text-accent">
                        {v.title}
                      </a>
                    ) : (
                      v.title
                    )}
                  </h3>
                  <span className="flex items-baseline gap-2 text-xs text-tertiary">
                    {v.stance && <span className={`font-medium ${stanceClass(v.stance)}`}>{sentenceCase(v.stance)}</span>}
                    {fmtDate(v.published_at)}
                  </span>
                </div>
                {v.summary && <p className="mt-1.5 text-[13px] leading-relaxed text-secondary">{v.summary}</p>}
                {v.key_levels && <p className="mt-1.5 text-xs text-tertiary">Levels: {v.key_levels}</p>}
              </article>
            ))}
          </div>
        </Card>
      )}

      {changes.length > 0 && (
        <Card>
          <SectionTitle right="Applied automatically, newest first">Recent flag changes</SectionTitle>
          <ul className="space-y-3">
            {changes.map((c, i) => (
              <li key={`${c.created_at}-${c.flag_key}-${i}`} className="grid grid-cols-[1fr_auto] gap-x-3">
                <div className="min-w-0">
                  <p className="text-sm">
                    <span className="text-tertiary">{CHANGE_LABEL[c.action] ?? c.action}:</span> {changeName(c, flags)}
                    {c.action !== 'retire' && c.score_modifier != null && (
                      <span className={`ml-2 font-medium ${modifierTone(c.score_modifier)}`}>{signed(c.score_modifier)}</span>
                    )}
                  </p>
                  {c.reason && <p className="mt-0.5 text-xs text-tertiary">{c.reason}</p>}
                  {c.status === 'failed' && <p className="mt-0.5 text-xs text-danger">Not applied: {c.apply_note}</p>}
                  {c.status === 'applied' && c.apply_note && <p className="mt-0.5 text-xs text-caution">{c.apply_note}</p>}
                </div>
                <span className="whitespace-nowrap text-xs text-tertiary">{fmtDate(c.created_at)}</span>
              </li>
            ))}
          </ul>
        </Card>
      )}

      <p className="px-1 text-xs text-tertiary">
        The price band sets the score, and the flags nudge it by at most ±{BTC_MODIFIER_CAP} in total. Flags update automatically each morning
        from new Benjamin Cowen videos.
      </p>
      <DataStamp fetchedAt={data.fetchedAt} />
    </div>
  );
}
