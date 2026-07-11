'use client';

// Interactive history chart for one indicator (client component — Recharts
// needs the DOM). Threshold line drawn from config where one exists.

import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  ReferenceLine,
  ResponsiveContainer,
  CartesianGrid,
} from 'recharts';

export interface ChartPoint {
  date: string;
  value: number;
}

export function IndicatorChart({
  points,
  threshold,
  thresholdLabel,
  unit,
}: {
  points: ChartPoint[];
  threshold?: number | null;
  thresholdLabel?: string;
  unit?: string;
}) {
  return (
    <div className="h-56 w-full">
      <ResponsiveContainer>
        <LineChart data={points} margin={{ top: 8, right: 16, bottom: 0, left: 0 }}>
          <CartesianGrid stroke="#1e293b" strokeDasharray="3 3" />
          <XAxis dataKey="date" tick={{ fill: '#94a3b8', fontSize: 11 }} tickMargin={6} />
          <YAxis
            tick={{ fill: '#94a3b8', fontSize: 11 }}
            width={52}
            domain={['auto', 'auto']}
            tickFormatter={(v: number) => (Math.abs(v) >= 1000 ? `${(v / 1000).toFixed(0)}k` : String(v))}
          />
          <Tooltip
            contentStyle={{ background: '#0f172a', border: '1px solid #334155', borderRadius: 8 }}
            labelStyle={{ color: '#cbd5e1' }}
            formatter={(value: number) => [`${value}${unit ?? ''}`, 'value']}
          />
          {threshold != null && (
            <ReferenceLine
              y={threshold}
              stroke="#f87171"
              strokeDasharray="6 4"
              label={{ value: thresholdLabel ?? `threshold ${threshold}`, fill: '#f87171', fontSize: 11, position: 'insideTopRight' }}
            />
          )}
          <Line type="monotone" dataKey="value" stroke="#38bdf8" strokeWidth={2} dot={{ r: 3 }} isAnimationActive={false} />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}

/** Small inline sparkline for the dashboard score history. */
export function ScoreSparkline({ points }: { points: Array<{ date: string; score: number }> }) {
  return (
    <div className="h-24 w-full">
      <ResponsiveContainer>
        <LineChart data={points} margin={{ top: 4, right: 4, bottom: 0, left: 4 }}>
          <YAxis domain={[0, 100]} hide />
          <XAxis dataKey="date" hide />
          <Tooltip
            contentStyle={{ background: '#0f172a', border: '1px solid #334155', borderRadius: 8 }}
            labelStyle={{ color: '#cbd5e1' }}
            formatter={(value: number) => [value.toFixed(1), 'score']}
          />
          <Line type="monotone" dataKey="score" stroke="#4ade80" strokeWidth={2} dot={false} isAnimationActive={false} />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
