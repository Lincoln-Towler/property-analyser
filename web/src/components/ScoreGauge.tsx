// Server-renderable SVG half-circle gauge for the 0-100 market score — no
// chart library needed for a static arc. The score is marked by a knob on
// the arc itself, so nothing crosses the number in the middle.

import { SIGNAL_BANDS } from '@/lib/theme';

function polar(cx: number, cy: number, r: number, pct: number): [number, number] {
  const rad = ((pct / 100) * 180 - 180) * (Math.PI / 180);
  return [cx + r * Math.cos(rad), cy + r * Math.sin(rad)];
}

function arcPath(cx: number, cy: number, r: number, fromPct: number, toPct: number): string {
  const [x1, y1] = polar(cx, cy, r, fromPct);
  const [x2, y2] = polar(cx, cy, r, toPct);
  const large = toPct - fromPct > 50 ? 1 : 0;
  return `M ${x1.toFixed(2)} ${y1.toFixed(2)} A ${r} ${r} 0 ${large} 1 ${x2.toFixed(2)} ${y2.toFixed(2)}`;
}

const CX = 100;
const CY = 96;
const R = 78;

export function ScoreGauge({ score, range }: { score: number; range?: [number, number] | null }) {
  const clamped = Math.max(0, Math.min(100, score));
  const [kx, ky] = polar(CX, CY, R, clamped);
  const showRange = range && range[0] !== range[1];

  return (
    <svg viewBox="0 0 200 112" className="w-full max-w-xs" role="img" aria-label={`Market score ${score.toFixed(1)} out of 100`}>
      {SIGNAL_BANDS.map((band) => {
        const active = clamped >= band.from && (clamped < band.to || band.to === 100);
        return (
          <path
            key={band.label}
            d={arcPath(CX, CY, R, band.from + 0.6, band.to - 0.6)}
            stroke={band.color}
            strokeOpacity={active ? 1 : 0.35}
            strokeWidth={12}
            fill="none"
          />
        );
      })}
      {showRange && (
        <path d={arcPath(CX, CY, R + 11, range![0], range![1])} stroke="#cbd5e1" strokeOpacity={0.6} strokeWidth={2} fill="none" />
      )}
      <circle cx={kx} cy={ky} r={9} fill="#0f172a" stroke="#f8fafc" strokeWidth={3} />
      <text x={CX} y={CY - 10} textAnchor="middle" className="fill-white" fontSize={34} fontWeight={700}>
        {score.toFixed(1)}
      </text>
      <text x={CX} y={CY + 9} textAnchor="middle" className="fill-slate-400" fontSize={10}>
        {showRange ? `range ${range![0]}–${range![1]}` : 'out of 100'}
      </text>
    </svg>
  );
}
