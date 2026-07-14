'use client';

// Daily BTC price + signal score history (client component — Recharts needs
// the DOM). Price on the left axis, 1–10 score on the right.

import {
  ComposedChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  CartesianGrid,
} from 'recharts';

export interface BtcChartPoint {
  date: string;
  price: number;
  score: number;
}

export function BtcHistoryChart({ points }: { points: BtcChartPoint[] }) {
  return (
    <div className="h-64 w-full">
      <ResponsiveContainer>
        <ComposedChart data={points} margin={{ top: 8, right: 8, bottom: 0, left: 0 }}>
          <CartesianGrid stroke="#1e293b" strokeDasharray="3 3" />
          <XAxis dataKey="date" tick={{ fill: '#94a3b8', fontSize: 11 }} tickMargin={6} />
          <YAxis
            yAxisId="price"
            tick={{ fill: '#94a3b8', fontSize: 11 }}
            width={56}
            domain={['auto', 'auto']}
            tickFormatter={(v: number) => `$${(v / 1000).toFixed(0)}k`}
          />
          <YAxis
            yAxisId="score"
            orientation="right"
            domain={[0, 10]}
            tick={{ fill: '#4ade80', fontSize: 11 }}
            width={32}
          />
          <Tooltip
            contentStyle={{ background: '#0f172a', border: '1px solid #334155', borderRadius: 8 }}
            labelStyle={{ color: '#cbd5e1' }}
            formatter={(value: number, name: string) =>
              name === 'price'
                ? [`$${value.toLocaleString('en-US')}`, 'price']
                : [value.toFixed(1), 'score /10']
            }
          />
          <Line
            yAxisId="price"
            type="monotone"
            dataKey="price"
            stroke="#f59e0b"
            strokeWidth={2}
            dot={false}
            isAnimationActive={false}
          />
          <Line
            yAxisId="score"
            type="monotone"
            dataKey="score"
            stroke="#4ade80"
            strokeWidth={2}
            dot={false}
            isAnimationActive={false}
          />
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  );
}
