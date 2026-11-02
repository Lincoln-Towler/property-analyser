'use client';

// History charts (client components — Recharts needs the DOM).
//
// Honesty rules shared by every chart here:
//  - the x-axis is real time, so a five-month gap looks like five months;
//  - lines are straight between readings (or stepped for policy rates) —
//    no smoothing that invents shape between sparse points;
//  - the target zone is shaded and the y-axis widens to include its edge,
//    so you can see how far a reading sits from where it should be.

import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  ReferenceLine,
  ReferenceArea,
  ResponsiveContainer,
  CartesianGrid,
} from 'recharts';
import { COLORS, TOOLTIP_STYLE, AXIS_TICK, SIGNAL_BANDS } from '@/lib/theme';
import { fmtDate, fmtIndicator, fmtIndicatorTick, isoToTs, niceScale, timeAxis, tsToIso } from '@/lib/format';

export interface ChartPoint {
  date: string;
  value: number;
}

export function IndicatorChart({
  points,
  unit = '',
  step = false,
  targetBand,
  warning,
}: {
  points: ChartPoint[];
  unit?: string;
  /** Draw as a step line (policy rates change in discrete decisions). */
  step?: boolean;
  targetBand?: [number | null, number | null] | null;
  warning?: { value: number; label: string } | null;
}) {
  const data = points.map((p) => ({ t: isoToTs(p.date), value: p.value }));
  const axis = timeAxis(points.map((p) => p.date));
  const showDots = points.length <= 30;
  // The target's finite edge is part of the range, so "below 100%" stays
  // visible even when every reading sits at 114%. An open side of the zone
  // runs to the edge of the chart.
  const values = points.map((p) => p.value);
  const targetEdges: number[] = [];
  if (targetBand?.[0] != null) targetEdges.push(targetBand[0]);
  if (targetBand?.[1] != null) targetEdges.push(targetBand[1]);
  // For a one-sided target ("below 100%"), reach a little past the edge so a
  // visible slice of the zone shows even when every reading is far away.
  const span = Math.max(...values, ...targetEdges) - Math.min(...values, ...targetEdges) || 1;
  if (targetBand && targetBand[0] === null && targetBand[1] !== null) targetEdges.push(targetBand[1] - span * 0.1);
  if (targetBand && targetBand[1] === null && targetBand[0] !== null) targetEdges.push(targetBand[0] + span * 0.1);
  const y = niceScale([...values, ...targetEdges]);
  const band = targetBand ? [targetBand[0] ?? y.domain[0], targetBand[1] ?? y.domain[1]] : null;

  return (
    <div className="h-56 w-full">
      <ResponsiveContainer>
        <LineChart data={data} margin={{ top: 10, right: 12, bottom: 0, left: 0 }}>
          <CartesianGrid stroke={COLORS.grid} strokeDasharray="3 3" vertical={false} />
          <XAxis
            dataKey="t"
            type="number"
            scale="time"
            domain={axis.domain}
            ticks={axis.ticks}
            tickFormatter={axis.format}
            tick={AXIS_TICK}
            tickMargin={6}
          />
          <YAxis tick={AXIS_TICK} width={48} domain={y.domain} ticks={y.ticks} tickFormatter={fmtIndicatorTick} />
          {band && <ReferenceArea y1={band[0]} y2={band[1]} fill={COLORS.good} fillOpacity={0.1} stroke="none" />}
          {warning && (
            <ReferenceLine
              y={warning.value}
              stroke={COLORS.bad}
              strokeDasharray="6 4"
              label={{ value: warning.label, fill: COLORS.bad, fontSize: 11, position: 'insideTopRight' }}
            />
          )}
          <Tooltip
            {...TOOLTIP_STYLE}
            labelFormatter={(t: number) => fmtDate(tsToIso(t))}
            formatter={(value: number) => [fmtIndicator(value, unit), 'reading']}
          />
          <Line
            type={step ? 'stepAfter' : 'linear'}
            dataKey="value"
            stroke={COLORS.accent}
            strokeWidth={2}
            dot={showDots ? { r: 2.5, fill: COLORS.accent } : false}
            activeDot={{ r: 4 }}
            isAnimationActive={false}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}

/**
 * Dashboard score history. The y-axis is zoomed to the range the score has
 * actually moved through (in 10-point steps) instead of 0–100, and the
 * signal zones are shaded behind the line so a move from Hold into Caution
 * is visible at a glance.
 */
export function ScoreHistoryChart({ points }: { points: Array<{ date: string; score: number }> }) {
  const scores = points.map((p) => p.score);
  const lo = Math.max(0, Math.floor((Math.min(...scores) - 4) / 10) * 10);
  const hi = Math.min(100, Math.ceil((Math.max(...scores) + 4) / 10) * 10);
  const data = points.map((p) => ({ t: isoToTs(p.date), score: p.score }));
  const axis = timeAxis(points.map((p) => p.date));
  const zones = SIGNAL_BANDS.filter((b) => b.to > lo && b.from < hi);

  return (
    <div className="h-36 w-full">
      <ResponsiveContainer>
        <LineChart data={data} margin={{ top: 4, right: 76, bottom: 0, left: 0 }}>
          {/* Zone names sit in the right-hand gutter, outside the plot, so the
              line can never run through them. */}
          {zones.map((z) => (
            <ReferenceArea
              key={z.label}
              y1={Math.max(z.from, lo)}
              y2={Math.min(z.to, hi)}
              fill={z.color}
              fillOpacity={0.09}
              stroke="none"
              label={{ value: z.label, position: 'right', fill: z.color, fontSize: 10 }}
            />
          ))}
          <XAxis
            dataKey="t"
            type="number"
            scale="time"
            domain={axis.domain}
            ticks={axis.ticks}
            tickFormatter={axis.format}
            tick={AXIS_TICK}
            tickMargin={4}
          />
          <YAxis domain={[lo, hi]} ticks={zones.map((z) => Math.max(z.from, lo)).concat(hi)} tick={AXIS_TICK} width={28} />
          <Tooltip
            {...TOOLTIP_STYLE}
            labelFormatter={(t: number) => fmtDate(tsToIso(t))}
            formatter={(value: number) => [value.toFixed(1), 'score']}
          />
          <Line type="stepAfter" dataKey="score" stroke="#e2e8f0" strokeWidth={2} dot={false} isAnimationActive={false} />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
