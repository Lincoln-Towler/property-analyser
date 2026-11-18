'use client';

// Head-to-head location comparison — the interactive part of the old
// Streamlit Location Analysis page (two selectors + metric table + score
// gauges + asking-price trend + recommendation). The metric is still stored as
// median_price, but since November it holds SQM's asking price, not a sold median.

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
import { COLORS, LOCATION_GOOD, TOOLTIP_STYLE, AXIS_TICK, scoreTone } from '@/lib/theme';
import { fmtDate, fmtMoneyTick, isoToTs, niceScale, timeAxis, tsToIso } from '@/lib/format';
import { Bar, StaleDate } from '@/components/ui';
import { InfoTip } from '@/components/InfoTip';

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
    .map((p) => ({ t: isoToTs(p.date), [left]: p[left], [right]: p[right] }));
  const axis = timeAxis(
    priceSeries.filter((p) => p[left] !== undefined || p[right] !== undefined).map((p) => p.date),
  );
  const yScale = niceScale(
    trend.flatMap((p) => [p[left], p[right]]).filter((v): v is number => typeof v === 'number'),
    4,
  );

  return (
    <div className="space-y-5">
      <div className="grid gap-2.5 sm:grid-cols-2">
        {(
          [
            ['First location', left, setLeft],
            ['Compare with', right, setRight],
          ] as Array<[string, string, (v: string) => void]>
        ).map(([label, value, setter]) => (
          <label key={label} className="block">
            <span className="mb-1 block text-[13px] text-tertiary">{label}</span>
            <span className="relative block">
              <select
                value={value}
                onChange={(e) => setter(e.target.value)}
                className="w-full appearance-none rounded-lg bg-tile py-2 pl-3 pr-8 text-sm text-primary"
              >
                {names.map((n) => (
                  <option key={n} value={n}>
                    {n}
                  </option>
                ))}
              </select>
              <span className="pointer-events-none absolute right-3 top-1/2 -translate-y-1/2 rotate-90 text-primary" aria-hidden>
                ›
              </span>
            </span>
          </label>
        ))}
      </div>

      {sameLocation && <p className="text-sm text-caution">Pick two different locations to compare them.</p>}

      <div className="grid gap-2.5 sm:grid-cols-2">
        {[a, b].map((loc, i) => {
          const tone = scoreTone(loc.score, LOCATION_GOOD);
          return (
            <div key={`${loc.location}-${i}`} className="rounded-lg bg-tile p-4">
              <p className="text-sm text-secondary">{loc.location}</p>
              <p className={`text-[40px] font-light leading-tight tracking-tight ${tone.text}`}>
                {loc.score}
                <span className="text-sm font-normal tracking-normal text-tertiary"> out of 100</span>
              </p>
              <div className="mt-2">
                <Bar value={loc.score} fill={tone.bar} track="bg-surface" />
              </div>
              <p className="mt-2 text-xs text-tertiary">Latest data {loc.latestDate ? fmtDate(loc.latestDate) : 'none'}</p>
            </div>
          );
        })}
      </div>

      <div className="no-scrollbar -mx-2 overflow-x-auto">
        <table className="w-full min-w-[520px] text-sm">
          <thead className="text-left text-[11px] text-tertiary">
            <tr>
              <th className="px-2 pb-1.5 font-medium">Metric</th>
              <th className="px-2 pb-1.5 font-medium">{a.location}</th>
              <th className="px-2 pb-1.5 font-medium">{b.location}</th>
            </tr>
          </thead>
          <tbody>
            {COMPARE_METRICS.map((metric) => {
              const va = a.metrics[metric]?.value;
              const vb = b.metrics[metric]?.value;
              // Only declare a winner for metrics with a defined direction.
              // sales_volume has none — its readings mix geographic scope and
              // period, so highlighting the larger number is meaningless.
              let winner: 'a' | 'b' | null = null;
              if (isComparableMetric(metric) && va !== undefined && vb !== undefined && va !== vb && !sameLocation) {
                const higherBetter = METRIC_HIGHER_IS_BETTER[metric];
                winner = (higherBetter ? va > vb : va < vb) ? 'a' : 'b';
              }
              return (
                <tr key={metric}>
                  <td className="px-2 py-1.5 text-secondary">
                    {METRIC_LABELS[metric] ?? metric}
                    {!isComparableMetric(metric) && (
                      <InfoTip label="not comparable" className="ml-2 text-xs text-tertiary">
                        These readings mix geographic scope and reporting period across locations, so they can&apos;t be compared.
                      </InfoTip>
                    )}
                  </td>
                  <td className={`px-2 py-1.5 ${winner === 'a' ? 'font-medium text-go' : ''}`}>
                    {formatMetric(metric, va)}
                    <StaleDate date={a.metrics[metric]?.date} latest={a.latestDate} />
                  </td>
                  <td className={`px-2 py-1.5 ${winner === 'b' ? 'font-medium text-go' : ''}`}>
                    {formatMetric(metric, vb)}
                    <StaleDate date={b.metrics[metric]?.date} latest={b.latestDate} />
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      <div>
        <h3 className="mb-3 text-[13px] font-semibold text-secondary">Asking price trend</h3>
        {trend.length >= 2 ? (
          <div className="h-64 w-full">
            <ResponsiveContainer>
              <LineChart data={trend} margin={{ top: 8, right: 16, bottom: 0, left: 8 }}>
                <CartesianGrid stroke={COLORS.grid} vertical={false} />
                <XAxis
                  dataKey="t"
                  type="number"
                  scale="time"
                  domain={axis.domain}
                  ticks={axis.ticks}
                  tickFormatter={axis.format}
                  tick={AXIS_TICK}
                  tickMargin={6}
                  axisLine={false}
                  tickLine={false}
                />
                <YAxis tick={AXIS_TICK} axisLine={false} tickLine={false} width={56} domain={yScale.domain} ticks={yScale.ticks} tickFormatter={fmtMoneyTick} />
                <Tooltip
                  {...TOOLTIP_STYLE}
                  labelFormatter={(t: number) => fmtDate(tsToIso(t))}
                  formatter={(value: number) => `$${value.toLocaleString('en-AU')}`}
                />
                <Legend wrapperStyle={{ fontSize: 12 }} />
                <Line
                  type="linear"
                  dataKey={left}
                  stroke={COLORS.accent}
                  strokeWidth={2}
                  dot={{ r: 3, stroke: 'none', fill: COLORS.accent }}
                  connectNulls
                  isAnimationActive={false}
                />
                {!sameLocation && (
                  <Line
                    type="linear"
                    dataKey={right}
                    stroke={COLORS.primary}
                    strokeWidth={2}
                    dot={{ r: 3, stroke: 'none', fill: COLORS.primary }}
                    connectNulls
                    isAnimationActive={false}
                  />
                )}
              </LineChart>
            </ResponsiveContainer>
          </div>
        ) : (
          <p className="rounded-lg bg-tile px-3 py-6 text-center text-sm text-secondary">
            SQM asking prices start in late October 2026, so this fills in as weekly readings arrive.
          </p>
        )}
      </div>

      {!sameLocation && (
        <p className="rounded-lg bg-tile px-4 py-3 text-sm text-secondary">
          {a.score === b.score ? (
            <>Both locations score {a.score} out of 100 on current data. Compare the individual metrics above to decide which suits your strategy.</>
          ) : (
            <>
              <strong className="font-semibold text-primary">{a.score > b.score ? a.location : b.location}</strong> shows stronger fundamentals on
              current data, {Math.max(a.score, b.score)} out of 100 against {Math.min(a.score, b.score)}. Consider timing and overall market
              conditions (see the dashboard score) before acting.
            </>
          )}
        </p>
      )}
    </div>
  );
}
