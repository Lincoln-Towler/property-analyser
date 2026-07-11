import { andersonState } from '@/lib/scoring/anderson';
import { Markdown } from '@/components/Markdown';

export const revalidate = 3600;

const STATUS_LABEL: Record<string, string> = {
  completed: '✅ Completed',
  in_progress: '⚠️ In progress',
  approaching: '⚠️ Approaching',
  upcoming: '⏳ Upcoming',
};

const PHASE_COLORS = ['#22c55e', '#eab308', '#f97316', '#ef4444', '#7f1d1d'];

export default function CyclePage() {
  const s = andersonState(new Date());
  const pct = (s.cyclePosition / s.cycleLength) * 100;

  return (
    <div className="space-y-8">
      <header>
        <h1 className="text-xl font-semibold">Anderson 18.6-Year Cycle</h1>
        <p className="mt-1 text-sm text-slate-400">
          Cycle anchored at {s.cycleStart} (GFC bottom) — the same anchor the market score uses.
        </p>
      </header>

      <section className="rounded-xl border border-slate-800 bg-slate-900/60 p-6">
        <div className="mb-2 flex items-baseline justify-between">
          <h2 className="text-2xl font-semibold">
            Year {s.cyclePosition.toFixed(1)} <span className="text-base font-normal text-slate-400">of {s.cycleLength}</span>
          </h2>
          <p className="text-lg">
            {s.phaseName} <span className="text-slate-400">· {s.phaseDescriptor}</span>
          </p>
        </div>
        <div className="relative h-5 w-full overflow-hidden rounded-full">
          <div className="absolute inset-0 flex">
            {s.phases.map((p, i) => (
              <div
                key={p.name}
                style={{
                  width: `${((p.end - p.start) / s.cycleLength) * 100}%`,
                  background: PHASE_COLORS[i],
                }}
              />
            ))}
          </div>
          <div
            className="absolute top-0 h-full w-1 rounded bg-white shadow"
            style={{ left: `calc(${pct}% - 2px)` }}
            title={`Year ${s.cyclePosition.toFixed(1)}`}
          />
        </div>
        <div className="mt-2 flex justify-between text-xs text-slate-500">
          <span>{s.cycleStart}</span>
          <span>{s.cycleEndYear}</span>
        </div>
      </section>

      <section className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
        {s.phases.map((p) => (
          <div
            key={p.name}
            className={`rounded-lg border p-4 ${
              p.status === 'in_progress'
                ? 'border-amber-600 bg-amber-950/30'
                : 'border-slate-800 bg-slate-900/60'
            }`}
          >
            <h3 className="font-medium">
              {p.name} <span className="text-sm text-slate-400">· {p.descriptor}</span>
            </h3>
            <p className="text-xs text-slate-500">
              Years {Math.trunc(p.start)}–{Math.round(p.end)} ({p.calendarRange})
            </p>
            <ul className="mt-2 list-disc pl-4 text-sm text-slate-300">
              {p.bullets.map((b) => (
                <li key={b}>{b}</li>
              ))}
            </ul>
            <p className="mt-2 text-sm">{STATUS_LABEL[p.status]}</p>
          </div>
        ))}
      </section>

      <section className="grid gap-4 lg:grid-cols-3">
        {(
          [
            ['✅ Signals We’ve Seen', s.seenSignals],
            ['⚠️ Warning Signs to Monitor', s.warningSigns],
            ['🔮 What to Watch Next', s.whatToWatch],
          ] as Array<[string, string[]]>
        ).map(([title, items]) => (
          <div key={title} className="rounded-lg border border-slate-800 bg-slate-900/60 p-4">
            <h3 className="mb-2 text-sm font-medium">{title}</h3>
            <ul className="list-disc space-y-1 pl-4 text-sm text-slate-300">
              {items.map((item) => (
                <li key={item}>{item}</li>
              ))}
            </ul>
          </div>
        ))}
      </section>

      <section className="rounded-xl border border-amber-700/60 bg-amber-950/30 p-6">
        <h2 className="mb-2 font-medium">According to Anderson&apos;s theory</h2>
        <p className="mb-3 text-sm text-slate-300">
          We are approximately at <strong>Year {s.cyclePosition.toFixed(1)} of the {s.cycleLength}-year cycle</strong> ({s.currentYear}) —
          currently in the <strong>{s.phaseName}</strong> phase ({s.phaseDescriptor}). Predicted peak:{' '}
          <strong>{s.predictedPeakYear}</strong>; predicted downturn window:{' '}
          <strong>{s.crashStartYear}–{s.crashEndYear}</strong>; historical comparison: {s.accuracy}.
        </p>
        <Markdown text={s.recommendations.map((r) => `- ${r}`).join('\n')} />
        <p className="mt-3 text-xs text-slate-400">
          ⚠️ One theory among many. Historically accurate but not guaranteed — use as one input, not the only factor.
        </p>
      </section>
    </div>
  );
}
