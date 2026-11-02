import { andersonState, type CyclePhase } from '@/lib/scoring/anderson';
import { Markdown } from '@/components/Markdown';
import { Card, PageHeader, SectionTitle } from '@/components/ui';

export const revalidate = 3600;

const PHASE_COLORS = ['#22c55e', '#eab308', '#f97316', '#ef4444', '#991b1b'];

function StatusMark({ status }: { status: CyclePhase['status'] }) {
  if (status === 'completed') return <span className="text-slate-500">✓ done</span>;
  if (status === 'in_progress') return <span className="font-medium text-amber-300">● now</span>;
  if (status === 'approaching') return <span className="text-amber-400/80">next</span>;
  return <span className="text-slate-500">upcoming</span>;
}

export default function CyclePage() {
  const s = andersonState(new Date());
  const pct = (s.cyclePosition / s.cycleLength) * 100;
  const widths = s.phases.map((p) => (p.end - p.start) / s.cycleLength);

  return (
    <div className="space-y-8">
      <PageHeader title="Anderson 18.6-year cycle">
        Cycle anchored at {s.cycleStart} (GFC bottom) — the same anchor the market score uses.
      </PageHeader>

      <Card>
        <div className="mb-4 flex flex-wrap items-baseline justify-between gap-2">
          <h2 className="text-3xl font-semibold tracking-tight">
            Year {s.cyclePosition.toFixed(1)}{' '}
            <span className="text-base font-normal text-slate-400">of {s.cycleLength}</span>
          </h2>
          <p className="text-lg">
            {s.phaseName} <span className="text-slate-400">· {s.phaseDescriptor}</span>
          </p>
        </div>

        <div className="relative">
          <div className="flex h-4 w-full overflow-hidden rounded-full">
            {s.phases.map((p, i) => (
              <div
                key={p.name}
                style={{
                  width: `${widths[i] * 100}%`,
                  background: PHASE_COLORS[i],
                  opacity: p.status === 'completed' ? 0.45 : 1,
                }}
              />
            ))}
          </div>
          <div
            className="absolute -top-1.5 h-7 w-1.5 rounded-full bg-white shadow-[0_0_0_3px_rgba(2,6,23,0.9)]"
            style={{ left: `calc(${pct}% - 3px)` }}
            title={`Year ${s.cyclePosition.toFixed(1)}`}
          />
        </div>
        <div className="mt-1.5 flex justify-between text-xs text-slate-500">
          <span>{s.cycleStart}</span>
          <span>{s.cycleEndYear}</span>
        </div>

        {/* Desktop: each phase labelled directly under its slice of the bar. */}
        <ol
          className="mt-4 hidden gap-3 md:grid"
          style={{ gridTemplateColumns: widths.map((w) => `${w}fr`).join(' ') }}
        >
          {s.phases.map((p, i) => (
            <li
              key={p.name}
              className={`min-w-0 border-t-2 pt-2 ${p.status === 'in_progress' ? 'rounded-b-lg bg-amber-950/30 px-2 pb-2' : ''}`}
              style={{ borderColor: PHASE_COLORS[i] }}
            >
              <p className="text-sm font-medium leading-tight">{p.name}</p>
              <p className="text-xs text-slate-400">{p.descriptor}</p>
              <p className="text-[11px] text-slate-500">{p.calendarRange}</p>
              <ul className="mt-1 space-y-0.5 text-[11px] leading-snug text-slate-400 [overflow-wrap:anywhere]">
                {p.bullets.map((b) => (
                  <li key={b}>{b}</li>
                ))}
              </ul>
              <p className="mt-1 text-[11px]">
                <StatusMark status={p.status} />
              </p>
            </li>
          ))}
        </ol>

        {/* Mobile: the same steps as a vertical list. */}
        <ol className="mt-5 space-y-3 md:hidden">
          {s.phases.map((p, i) => (
            <li
              key={p.name}
              className={`flex gap-3 rounded-lg p-2 ${p.status === 'in_progress' ? 'bg-amber-950/30 ring-1 ring-amber-800/60' : ''}`}
            >
              <span className="mt-1 h-3 w-3 shrink-0 rounded-full" style={{ background: PHASE_COLORS[i] }} aria-hidden />
              <div className="min-w-0 flex-1">
                <div className="flex items-baseline justify-between gap-2">
                  <p className="text-sm font-medium">
                    {p.name} <span className="text-xs font-normal text-slate-400">· {p.descriptor}</span>
                  </p>
                  <span className="text-xs">
                    <StatusMark status={p.status} />
                  </span>
                </div>
                <p className="text-xs text-slate-500">
                  {p.calendarRange} · {p.bullets.join(' · ')}
                </p>
              </div>
            </li>
          ))}
        </ol>
      </Card>

      <section className="grid gap-4 md:grid-cols-2">
        {(
          [
            ['Warning signs to monitor', s.warningSigns],
            ['What to watch next', s.whatToWatch],
          ] as Array<[string, string[]]>
        ).map(([title, items]) => (
          <Card key={title} as="div" tight>
            <SectionTitle>{title}</SectionTitle>
            <ul className="list-disc space-y-1 pl-4 text-sm text-slate-300">
              {items.map((item) => (
                <li key={item}>{item}</li>
              ))}
            </ul>
          </Card>
        ))}
      </section>

      <section className="rounded-xl border border-amber-800/60 bg-amber-950/20 p-5 sm:p-6">
        <h2 className="mb-2 text-lg font-semibold">According to Anderson&apos;s theory</h2>
        <p className="mb-3 text-sm text-slate-300">
          We are at roughly <strong>Year {s.cyclePosition.toFixed(1)} of the {s.cycleLength}-year cycle</strong> (
          {s.currentYear}) — in the <strong>{s.phaseName}</strong> phase ({s.phaseDescriptor}). Predicted peak{' '}
          <strong>{s.predictedPeakYear}</strong>; predicted downturn{' '}
          <strong>
            {s.crashStartYear}–{s.crashEndYear}
          </strong>
          .
        </p>
        <Markdown text={s.recommendations.map((r) => `- ${r}`).join('\n')} />
        <p className="mt-3 text-xs text-slate-400">
          One theory among many. Historically accurate but not guaranteed — use it as one input, not the only factor.
        </p>
      </section>
    </div>
  );
}
