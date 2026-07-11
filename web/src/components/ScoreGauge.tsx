// Server-renderable SVG half-circle gauge for the 0-100 market score — no
// chart library needed for a static arc.

const BANDS: Array<{ upTo: number; color: string }> = [
  { upTo: 30, color: '#dc2626' },
  { upTo: 40, color: '#ea580c' },
  { upTo: 50, color: '#eab308' },
  { upTo: 60, color: '#a3e635' },
  { upTo: 75, color: '#22c55e' },
  { upTo: 100, color: '#15803d' },
];

function polar(cx: number, cy: number, r: number, angleDeg: number): [number, number] {
  const rad = ((angleDeg - 180) * Math.PI) / 180;
  return [cx + r * Math.cos(rad), cy + r * Math.sin(rad)];
}

function arcPath(cx: number, cy: number, r: number, fromPct: number, toPct: number): string {
  const [x1, y1] = polar(cx, cy, r, (fromPct / 100) * 180);
  const [x2, y2] = polar(cx, cy, r, (toPct / 100) * 180);
  const large = toPct - fromPct > 50 ? 1 : 0;
  return `M ${x1.toFixed(2)} ${y1.toFixed(2)} A ${r} ${r} 0 ${large} 1 ${x2.toFixed(2)} ${y2.toFixed(2)}`;
}

export function ScoreGauge({ score, label }: { score: number; label: string }) {
  const clamped = Math.max(0, Math.min(100, score));
  const [nx, ny] = polar(100, 95, 62, (clamped / 100) * 180);
  let prev = 0;
  return (
    <svg viewBox="0 0 200 110" className="w-full max-w-sm" role="img" aria-label={`Market score ${score.toFixed(1)} out of 100`}>
      {BANDS.map((band) => {
        const path = arcPath(100, 95, 80, prev, band.upTo);
        prev = band.upTo;
        return <path key={band.upTo} d={path} stroke={band.color} strokeWidth={14} fill="none" strokeLinecap="butt" />;
      })}
      <line x1={100} y1={95} x2={nx} y2={ny} stroke="#e2e8f0" strokeWidth={3} strokeLinecap="round" />
      <circle cx={100} cy={95} r={5} fill="#e2e8f0" />
      <text x={100} y={78} textAnchor="middle" className="fill-white" fontSize={26} fontWeight={700}>
        {score.toFixed(1)}
      </text>
      <text x={100} y={106} textAnchor="middle" className="fill-slate-400" fontSize={9}>
        {label}
      </text>
    </svg>
  );
}
