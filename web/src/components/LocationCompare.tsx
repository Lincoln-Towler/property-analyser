'use client';

// Head-to-head location comparison — the interactive part of the old
// Streamlit Location Analysis page (two selectors + metric table + score
// gauges + median-price trend + recommendation).

import { useState } from 'react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  Legend,
  ResponsiveContainer,
  CartesianGrid,
} from 'recharts';
import {
  METRIC_LABELS,
  METRIC_HIGHER_IS_BETTER,
  isComparableMetric,
  formatMetric,
  type ScoredLocation,
} from '@/lib/scoring/location';

export interface PriceSeriesPoint {
  date: string;
  [location: string]: string | number;
}

const COMPARE_METRICS = [
  'median_price',
  'annual_growth',
  'rental_yield',
  'vacancy_rate',
  'days_on_market',
  'sales_volume',
];

function scoreColor(score: number): string {
  if (score >= 70) return 'text-emerald-400';
  if (score >= 40) return 'text-amber-400';
  return 'text-red-400';
}

function ScoreBar({ score }: { score: number }) {
  return (
    <div className="h-2 w-full rounded-full bg-slate-800">
      <div
        className={`h-2 rounded-full ${
          score >= 70 ? 'bg-emerald-500' : score >= 40 ? 'bg-amber-500' : 'bg-red-500'
        }`}
        style={{ width: `${score}%` }}
      />
    </div>
  );
}

export function LocationCompare({
  scored,
  priceSeries,
}: {
  scored: ScoredLocation[];
  priceSeries: PriceSeriesPoint[];
}) {
  const names = scored.map((s) => s.location);
  const [left, setLeft] = useState(names[0] ?? '');
  const [right, setRight] = useState(names[1] ?? names[0] ?? '');

  const a = scored.find((s) => s.location === left);
  const b = scored.find((s) => s.location === right);
  if (!a || !b) return null;

  const sameLocation = left === right;

  // Only keep dates where at least one of the two has a price reading
  const trend = priceSeries
    .filter((p) => p[left] !== undefined || p[right] !== undefined)
    .map((p) => ({ date: p.date, [left]: p[left], [right]: p[right] }));

  return (
    <div className="space-y-6">
      <div className="grid gap-3 sm:grid-cols-2">
        {(
          [
            ['Primary location', left, setLeft],
            ['Compare with', right, setRight],
          ] as Array<[string, string, (v: string) => void]>
        ).map(([label, value, setter]) => (
          <label key={label} className="block">
            <span className="mb-1 block text-xs uppercase tracking-wide text-slate-400">
              {label}
            </span>
            <select
              value={value}
              onChange={(e) => setter(e.target.value)}
              className="w-full rounded-lg border border-slate-700 bg-slate-900 px-3 py-2 text-sm text-slate-100"
            >
              {names.map((n) => (
                <option key={n} value={n}>
                  {n}
                </option>
              ))}
            </select>
          </label>
        ))}
      </div>

      {sameLocation && (
        <p className="rounded-lg border border-amber-700/60 bg-amber-950/30 px-4 py-3 text-sm">
          Pick two different locations to compare them.
        </p>
      )}

      <div className="grid gap-4 sm:grid-cols-2">
        {[a, b].map((loc, i) => (
          <div
            key={`${loc.location}-${i}`}
            className="rounded-xl border border-slate-800 bg-slate-900/60 p-5"
          >
            <p className="text-sm text-slate-400">{loc.location}</p>
            <p className={`text-3xl font-semibold tabular-nums ${scoreColor(loc.score)}`}>
              {loc.score}
              <span className="text-base font-normal text-slate-500">/100</span>
            </p>
            <div className="mt-3">
              <ScoreBar score={loc.score} />
            </div>
            <p className="mt-2 text-xs text-slate-500">
              Latest data: {loc.latestDate ?? 'none'}
            </p>
          </div>
        ))}
      </div>

      <div className="overflow-x-auto rounded-xl border border-slate-800">
        <table className="w-full min-w-[520px] text-sm">
          <thead className="bg-slate-900 text-left text-xs uppercase tracking-wide text-slate-400">
            <tr>
              <th className="px-4 py-3">Metric</th>
              <th className="px-4 py-3">{a.location}</th>
              <th className="px-4 py-3">{b.location}</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-800">
            {COMPARE_METRICS.map((metric) => {
              const va = a.metrics[metric]?.value;
              const vb = b.metrics[metric]?.value;
              // Only declare a winner for metrics with a defined direction.
              // sales_volume has none — its readings mix geographic scope and
              // period, so highlighting the larger number is meaningless.
              let winner: 'a' | 'b' | null = null;
              if (
                isComparableMetric(metric) &&
                va !== undefined &&
                vb !== undefined &&
                va !== vb &&
                !sameLocation
              ) {
                const higherBetter = METRIC_HIGHER_IS_BETTER[metric];
                winner = (higherBetter ? va > vb : va < vb) ? 'a' : 'b';
              }
              return (
                <tr key={metric}>
                  <td className="px-4 py-3 text-slate-300">
                    {METRIC_LABELS[metric] ?? metric}
                    {!isComparableMetric(metric) && (
                      <span
                        className="ml-2 text-xs text-slate-500"
                        title="Readings mix geographic scope and reporting period across locations, so they are not comparable."
                      >
                        not comparable
                      </span>
                    )}
                  </td>
                  <td
                    className={`px-4 py-3 tabular-nums ${
                      winner === 'a' ? 'font-semibold text-emerald-400' : ''
                    }`}
                  >
                    {formatMetric(metric, va)}
                    {a.metrics[metric] && (
                      <span className="ml-2 text-xs text-slate-500">{a.metrics[metric]!.date}</span>
                    )}
                  </td>
                  <td
                    className={`px-4 py-3 tabular-nums ${
                      winner === 'b' ? 'font-semibold text-emerald-400' : ''
                    }`}
                  >
                    {formatMetric(metric, vb)}
                    {b.metrics[metric] && (
                      <span className="ml-2 text-xs text-slate-500">{b.metrics[metric]!.date}</span>
                    )}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      <section className="rounded-xl border border-slate-800 bg-slate-900/60 p-5">
        <h3 className="mb-3 text-sm font-medium uppercase tracking-wide text-slate-400">
          Median price trend
        </h3>
        {trend.length >= 2 ? (
          <div className="h-64 w-full">
            <ResponsiveContainer>
              <LineChart data={trend} margin={{ top: 8, right: 16, bottom: 0, left: 8 }}>
                <CartesianGrid stroke="#1e293b" strokeDasharray="3 3" />
                <XAxis dataKey="date" tick={{ fill: '#94a3b8', fontSize: 11 }} />
                <YAxis
                  tick={{ fill: '#94a3b8', fontSize: 11 }}
                  width={62}
                  domain={['auto', 'auto']}
                  tickFormatter={(v: number) => `$${(v / 1000).toFixed(0)}K`}
                />
                <Tooltip
                  contentStyle={{ background: '#0f172a', border: '1px solid #334155', borderRadius: 8 }}
                  labelStyle={{ color: '#cbd5e1' }}
                  formatter={(value: number) => `$${value.toLocaleString('en-AU')}`}
                />
                <Legend wrapperStyle={{ fontSize: 12 }} />
                <Line
                  type="monotone"
                  dataKey={left}
                  stroke="#38bdf8"
                  strokeWidth={2}
                  connectNulls
                  isAnimationActive={false}
                />
                {!sameLocation && (
                  <Line
                    type="monotone"
                    dataKey={right}
                    stroke="#f472b6"
                    strokeWidth={2}
                    connectNulls
                    isAnimationActive={false}
                  />
                )}
              </LineChart>
            </ResponsiveContainer>
          </div>
        ) : (
          <p className="rounded-md bg-slate-800/60 px-3 py-6 text-center text-sm text-slate-400">
            Need median_price readings on at least two dates to chart a trend.
          </p>
        )}
      </section>

      {!sameLocation && (
        <section className="rounded-xl border border-slate-800 bg-slate-900/60 p-5 text-sm">
          <h3 className="mb-2 font-medium">Recommendation</h3>
          {a.score === b.score ? (
            <p className="text-slate-300">
              Both locations score {a.score}/100 on current data. Compare the individual metrics
              above to decide which suits your strategy.
            </p>
          ) : (
            <p className="text-slate-300">
              <strong>{a.score > b.score ? a.location : b.location}</strong> shows stronger
              fundamentals on current data — {Math.max(a.score, b.score)}/100 vs{' '}
              {Math.min(a.score, b.score)}/100. Consider timing and overall market conditions
              (see the Dashboard score) before acting.
            </p>
          )}
        </section>
      )}
    </div>
  );
}
