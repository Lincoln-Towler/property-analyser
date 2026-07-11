import { getSiteData } from '@/lib/data';
import { INDICATORS_CONFIG, EXTRA_INDICATOR_NAMES } from '@/lib/scoring/config';
import { indicatorStatus, indicatorTargetText } from '@/lib/scoring/ui-helpers';
import { IndicatorChart } from '@/components/IndicatorChart';
import { SetupNotice } from '@/components/SetupNotice';

export const revalidate = 3600;

const STATUS_DOT: Record<string, string> = {
  success: 'bg-emerald-500',
  warning: 'bg-amber-500',
  danger: 'bg-red-500',
};

/** Threshold shown on the chart: warning/danger level for inverse
 *  indicators, healthy floor for direct ones (same rule as the old app). */
function trendThreshold(key: string): number | null {
  const cfg = INDICATORS_CONFIG[key];
  if (!cfg) return key === 'mortgage_arrears_rate' ? 2.0 : null;
  if (cfg.impact === 'inverse') {
    return cfg.warning_above ?? cfg.danger_above ?? cfg.optimal_below ?? cfg.oversupply_above ?? null;
  }
  return cfg.healthy_above ?? cfg.optimal_above ?? null;
}

export default async function IndicatorsPage() {
  const data = await getSiteData();
  if (!data) return <SetupNotice />;

  const names = [
    ...Object.keys(INDICATORS_CONFIG),
    ...Object.keys(data.series).filter((n) => !(n in INDICATORS_CONFIG)),
  ];

  return (
    <div className="space-y-6">
      <h1 className="text-xl font-semibold">Economic Indicators</h1>
      <div className="grid gap-6 lg:grid-cols-2">
        {names.map((key) => {
          const cfg = INDICATORS_CONFIG[key];
          const points = (data.series[key] ?? [])
            .slice()
            .sort((a, b) => (a.date < b.date ? -1 : 1));
          const latest = points.length ? points[points.length - 1] : null;
          const displayName =
            cfg?.display_name ?? EXTRA_INDICATOR_NAMES[key] ?? key.replace(/_/g, ' ');
          const status = cfg ? indicatorStatus(latest?.value ?? null, cfg) : 'warning';
          const target = cfg ? indicatorTargetText(cfg) : null;
          const unit = cfg?.unit ?? '';

          return (
            <section key={key} className="rounded-xl border border-slate-800 bg-slate-900/60 p-5">
              <header className="mb-2 flex items-start justify-between gap-2">
                <div>
                  <h2 className="font-medium">{displayName}</h2>
                  <p className="text-xs text-slate-400">
                    {latest ? (
                      <>
                        Current:{' '}
                        <span className="font-semibold text-slate-200 tabular-nums">
                          {Math.abs(latest.value) >= 1000
                            ? latest.value.toLocaleString('en-US')
                            : latest.value}
                          {unit}
                        </span>
                        {target ? ` · Target: ${target}` : ''} · as of {latest.date}
                      </>
                    ) : (
                      'No data yet'
                    )}
                  </p>
                </div>
                {cfg && <span className={`mt-1 h-3 w-3 shrink-0 rounded-full ${STATUS_DOT[status]}`} />}
              </header>
              {points.length >= 2 ? (
                <IndicatorChart
                  points={points.map((p) => ({ date: p.date, value: p.value }))}
                  threshold={trendThreshold(key)}
                  thresholdLabel={target ? `target ${target}` : undefined}
                  unit={unit}
                />
              ) : (
                <p className="rounded-md bg-slate-800/60 px-3 py-6 text-center text-sm text-slate-400">
                  {points.length === 1
                    ? '1 data point — no trend to chart yet'
                    : 'No data — add rows in Supabase or via the n8n feed'}
                </p>
              )}
            </section>
          );
        })}
      </div>
    </div>
  );
}
