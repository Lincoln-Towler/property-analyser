'use client';

// Daily BTC price against the key-level ladder (client component — Recharts
// needs the DOM). The current band (support → resistance) is shaded and
// every key level near the price range is drawn as a reference line, so the
// chart shows where price sits in the framework rather than a score that
// has not moved.

import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  ReferenceArea,
  ReferenceLine,
  ResponsiveContainer,
  CartesianGrid,
} from 'recharts';
import { COLORS, TOOLTIP_STYLE, AXIS_TICK } from '@/lib/theme';
import { fmtDate, fmtMoneyTick, fmtUsd, isoToTs, niceScale, timeAxis, tsToIso } from '@/lib/format';

export interface BtcChartPoint {
  date: string;
  price: number;
}

export function BtcHistoryChart({
  points,
  levels,
  support,
  resistance,
}: {
  points: BtcChartPoint[];
  levels: number[];
  support: number | null;
  resistance: number | null;
}) {
  const data = points.map((p) => ({ t: isoToTs(p.date), price: p.price }));
  const axis = timeAxis(points.map((p) => p.date), 6);
  const prices = points.map((p) => p.price);
  const lo = Math.min(...prices);
  const hi = Math.max(...prices);
  // Levels within ~8% of the traded range; anything further would only
  // squash the price line.
  const visible = [...new Set(levels)].filter((l) => l >= lo * 0.92 && l <= hi * 1.08);
  const y = niceScale([lo, hi, ...visible], 6);

  return (
    <div className="h-72 w-full">
      <ResponsiveContainer>
        <LineChart data={data} margin={{ top: 8, right: 56, bottom: 0, left: 0 }}>
          <CartesianGrid stroke={COLORS.grid} strokeDasharray="3 3" vertical={false} />
          {support !== null && resistance !== null && (
            <ReferenceArea y1={support} y2={resistance} fill={COLORS.btc} fillOpacity={0.08} stroke="none" />
          )}
          {visible.map((level) => {
            const isBand = level === support || level === resistance;
            return (
              <ReferenceLine
                key={level}
                y={level}
                stroke={isBand ? COLORS.btc : '#475569'}
                strokeOpacity={isBand ? 0.9 : 0.7}
                strokeDasharray={isBand ? '6 3' : '2 4'}
                label={{
                  value: fmtMoneyTick(level),
                  position: 'right',
                  fill: isBand ? COLORS.btc : COLORS.axis,
                  fontSize: 10,
                }}
              />
            );
          })}
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
          <YAxis tick={AXIS_TICK} width={52} domain={y.domain} ticks={y.ticks} tickFormatter={fmtMoneyTick} />
          <Tooltip
            {...TOOLTIP_STYLE}
            labelFormatter={(t: number) => fmtDate(tsToIso(t))}
            formatter={(value: number) => [fmtUsd(value), 'price']}
          />
          <Line type="linear" dataKey="price" stroke={COLORS.btc} strokeWidth={2} dot={false} isAnimationActive={false} />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
