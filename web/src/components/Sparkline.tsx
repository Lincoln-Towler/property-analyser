// Tiny server-rendered sparkline (plain SVG — no chart library, no client
// JavaScript). Positions points by real date so uneven gaps stay uneven.

import { isoToTs } from '@/lib/format';

export function Sparkline({
  points,
  stroke,
  height = 36,
}: {
  points: Array<{ date: string; value: number }>;
  stroke: string;
  height?: number;
}) {
  if (points.length < 2) return <div style={{ height }} aria-hidden />;
  const ts = points.map((p) => isoToTs(p.date));
  const vs = points.map((p) => p.value);
  const [t0, t1] = [Math.min(...ts), Math.max(...ts)];
  const [v0, v1] = [Math.min(...vs), Math.max(...vs)];
  const W = 100;
  const pad = 3;
  const x = (t: number) => (t1 === t0 ? W / 2 : ((t - t0) / (t1 - t0)) * W);
  const y = (v: number) => (v1 === v0 ? height / 2 : pad + (1 - (v - v0) / (v1 - v0)) * (height - pad * 2));
  const path = points.map((p, i) => `${i ? 'L' : 'M'}${x(ts[i]).toFixed(2)},${y(p.value).toFixed(2)}`).join(' ');

  return (
    <svg viewBox={`0 0 ${W} ${height}`} preserveAspectRatio="none" className="w-full" style={{ height }} aria-hidden>
      <path d={path} fill="none" stroke={stroke} strokeWidth={2} vectorEffect="non-scaling-stroke" strokeLinejoin="round" />
    </svg>
  );
}
