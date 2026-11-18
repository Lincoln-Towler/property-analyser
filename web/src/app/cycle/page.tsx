import { andersonState, type CyclePhase } from '@/lib/scoring/anderson';
import { Markdown } from '@/components/Markdown';
import { Card, PageHeader, SectionTitle } from '@/components/ui';

export const revalidate = 3600;

const PHASE_COLORS = ['#47c280', '#c8bc55', '#e08850', '#e96660', '#a8433f'];

function StatusMark({ status }: { status: CyclePhase['status'] }) {
  if (status === 'completed') return <span className="text-tertiary">Done</span>;
  if (status === 'in_progress') return <span className="font-medium text-caution">Now</span>;
  if (status === 'approaching') return <span className="text-secondary">Next</span>;
  return <span className="text-tertiary">Upcoming</span>;
}

export default function CyclePage() {
  const s = andersonState(new Date());
  const pct = (s.cyclePosition / s.cycleLength) * 100;
  const widths = s.phases.map((p) => (p.end - p.start) / s.cycleLength);

  return (
    <div className="space-y-2.5">
      <PageHeader title="Anderson cycle">
        The 18.6-year cycle, anchored at {s.cycleStart} (GFC bottom), the same anchor the market score uses.
      </PageHeader>

      <Card>
        <div className="mb-4 flex flex-wrap items-baseline justify-between gap-2">
          <h2 className="text-[40px] font-light leading-none tracking-tight">
            Year {s.cyclePosition.toFixed(1)} <span className="text-sm font-normal tracking-normal text-tertiary">of {s.cycleLength}</span>
          </h2>
          <p className="text-[15px] font-medium">
            {s.phaseName} <span className="font-normal text-secondary">· {s.phaseDescriptor}</span>
          </p>
        </div>

        <div className="relative">
          <div className="flex h-3 w-full gap-0.5 overflow-hidden rounded-full">
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
            className="absolute -top-1 h-5 w-1 rounded-full bg-primary"
            style={{ left: `calc(${pct}% - 2px)` }}
            aria-hidden
          />
        </div>
        <div className="mt-1.5 flex justify-between text-xs text-tertiary">
          <span>{s.cycleStart}</span>
          <span>{s.cycleEndYear}</span>
        </div>

        {/* Desktop: the phases side by side, tied to their slice of the bar by colour.
            Equal columns: at true scale the two-year phases were too narrow for their notes. */}
        <ol className="mt-4 hidden gap-2 md:grid" style={{ gridTemplateColumns: `repeat(${s.phases.length}, minmax(0, 1fr))` }}>
          {s.phases.map((p, i) => (
            <li key={p.name} className={`min-w-0 rounded-lg p-2 ${p.status === 'in_progress' ? 'bg-tile' : ''}`}>
              <p className="flex items-center gap-1.5 text-sm font-medium leading-tight">
                <span className="h-2.5 w-2.5 shrink-0 rounded-full" style={{ background: PHASE_COLORS[i] }} aria-hidden />
                {p.name}
              </p>
              <p className="text-xs text-secondary">{p.descriptor}</p>
              <p className="text-[11px] text-tertiary">{p.calendarRange}</p>
              <ul className="mt-1 space-y-0.5 text-[11px] leading-snug text-tertiary">
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
        <ol className="mt-4 space-y-1 md:hidden">
          {s.phases.map((p, i) => (
            <li key={p.name} className={`flex gap-3 rounded-lg p-2 ${p.status === 'in_progress' ? 'bg-tile' : ''}`}>
              <span className="mt-1 h-3 w-3 shrink-0 rounded-full" style={{ background: PHASE_COLORS[i] }} aria-hidden />
              <div className="min-w-0 flex-1">
                <div className="flex items-baseline justify-between gap-2">
                  <p className="text-sm font-medium">
                    {p.name} <span className="text-xs font-normal text-secondary">· {p.descriptor}</span>
                  </p>
                  <span className="text-xs">
                    <StatusMark status={p.status} />
                  </span>
                </div>
                <p className="text-xs text-tertiary">
                  {p.calendarRange} · {p.bullets.join(' · ')}
                </p>
              </div>
            </li>
          ))}
        </ol>
      </Card>

      <div className="grid gap-2.5 md:grid-cols-2">
        {(
          [
            ['Warning signs to monitor', s.warningSigns],
            ['What to watch next', s.whatToWatch],
          ] as Array<[string, string[]]>
        ).map(([title, items]) => (
          <Card key={title} as="div">
            <SectionTitle>{title}</SectionTitle>
            <ul className="list-disc space-y-1 pl-4 text-sm text-secondary">
              {items.map((item) => (
                <li key={item}>{item}</li>
              ))}
            </ul>
          </Card>
        ))}
      </div>

      <Card>
        <SectionTitle>According to Anderson&apos;s theory</SectionTitle>
        <p className="mb-3 text-sm text-secondary">
          We are at roughly{' '}
          <strong className="font-semibold text-primary">
            year {s.cyclePosition.toFixed(1)} of the {s.cycleLength}-year cycle
          </strong>{' '}
          ({s.currentYear}), in the <strong className="font-semibold text-primary">{s.phaseName}</strong> phase ({s.phaseDescriptor}).
          Predicted peak <strong className="font-semibold text-primary">{s.predictedPeakYear}</strong>; predicted downturn{' '}
          <strong className="font-semibold text-primary">
            {s.crashStartYear}–{s.crashEndYear}
          </strong>
          .
        </p>
        <Markdown text={s.recommendations.map((r) => `- ${r}`).join('\n')} />
        <p className="mt-3 text-xs text-tertiary">
          One theory among many. Historically accurate but not guaranteed, so use it as one input, not the only factor.
        </p>
      </Card>
    </div>
  );
}
