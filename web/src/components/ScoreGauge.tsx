// The market score as a meter: the six signal bands round a half dial, a needle
// on the score, and a faint trail back to where it stood 30 days ago. The dial
// is the dashboard's one standout (web/DESIGN.md).
//
// Pure SVG, rendered on the server. The needle's sweep and the band lighting up
// are CSS animations in globals.css, so the page ships no script for them. The
// numbers sit under the dial as HTML, where the needle can't cross them and
// screen readers get real text, so the drawing itself is aria-hidden.

import { COLORS, SIGNAL_BANDS, signalBand } from '@/lib/theme';

const CX = 155;
const CY = 150;
const R = 124; // centre line of the band arc
const BAND_WIDTH = 14; // so the arc runs from 117 to 131
const GAP = 1.2; // score points left dark between neighbouring bands

const TICK_OUTER = 112;
const TRAIL_R = 100;
const LABEL_R = 143; // outside the arc, where the needle never reaches
const NEEDLE_TIP = 113;

const BOUNDARIES = [0, ...SIGNAL_BANDS.map((b) => b.to)]; // 0, 30, 40, 50, 60, 75, 100

const clamp = (v: number) => Math.max(0, Math.min(100, v));

/** A score's point on a circle of radius r: 0 at the left, 50 at the top, 100 at the right. */
function polar(r: number, score: number): [number, number] {
  const a = Math.PI + (score / 100) * Math.PI;
  return [CX + r * Math.cos(a), CY + r * Math.sin(a)];
}

function arc(r: number, from: number, to: number): string {
  const [x1, y1] = polar(r, from);
  const [x2, y2] = polar(r, to);
  return `M ${x1.toFixed(2)} ${y1.toFixed(2)} A ${r} ${r} 0 0 1 ${x2.toFixed(2)} ${y2.toFixed(2)}`;
}

function Tick({ score, inner, opacity, width }: { score: number; inner: number; opacity: number; width: number }) {
  const [x1, y1] = polar(inner, score);
  const [x2, y2] = polar(TICK_OUTER, score);
  return <line x1={x1} y1={y1} x2={x2} y2={y2} stroke={COLORS.primary} strokeOpacity={opacity} strokeWidth={width} />;
}

// Drawn pointing straight up (a score of 50), then rotated onto the score.
// Wide at the pivot with a slightly blunt tip, so it reads at a glance; a short
// tail past the pivot makes it a needle rather than an arrow.
const NEEDLE = [
  [CX - 1.4, CY - NEEDLE_TIP],
  [CX + 1.4, CY - NEEDLE_TIP],
  [CX + 6, CY],
  [CX + 4.2, CY + 13],
  [CX - 4.2, CY + 13],
  [CX - 6, CY],
]
  .map(([x, y]) => `${x},${y}`)
  .join(' ');

export function ScoreGauge({ score, previous }: { score: number; previous?: { score: number } | null }) {
  const value = clamp(score);
  const active = signalBand(value);
  const from = previous && Math.abs(previous.score - value) >= 0.3 ? clamp(previous.score) : null;

  return (
    <svg viewBox="0 0 310 164" className="mx-auto block w-full max-w-[24rem]" aria-hidden>
      {SIGNAL_BANDS.map((band) => {
        const isActive = band === active;
        return (
          <path
            key={band.label}
            d={arc(R, band.from === 0 ? 0 : band.from + GAP / 2, band.to === 100 ? 100 : band.to - GAP / 2)}
            stroke={band.color}
            strokeWidth={BAND_WIDTH}
            fill="none"
            opacity={isActive ? 1 : 0.28}
            className={isActive ? 'meter-band-active' : undefined}
          />
        );
      })}

      {Array.from({ length: 21 }, (_, i) => i * 5)
        .filter((s) => !BOUNDARIES.includes(s))
        .map((s) => (
          <Tick key={s} score={s} inner={TICK_OUTER - 4} opacity={0.2} width={1} />
        ))}
      {BOUNDARIES.map((s) => (
        <Tick key={s} score={s} inner={TICK_OUTER - 8} opacity={0.5} width={1.6} />
      ))}
      {BOUNDARIES.map((s) => {
        const [x, y] = polar(LABEL_R, s);
        return (
          <text key={s} x={x} y={y} dy="0.35em" textAnchor="middle" fontSize={10} fill={COLORS.axis}>
            {s}
          </text>
        );
      })}

      {from !== null && (
        <g opacity={0.6}>
          <path d={arc(TRAIL_R, Math.min(from, value), Math.max(from, value))} stroke={COLORS.primary} strokeWidth={2.5} strokeLinecap="round" fill="none" />
          <circle cx={polar(TRAIL_R, from)[0]} cy={polar(TRAIL_R, from)[1]} r={3.2} fill={COLORS.primary} />
        </g>
      )}

      <g className="meter-needle" style={{ transform: `rotate(${((value - 50) * 1.8).toFixed(2)}deg)`, transformOrigin: `${CX}px ${CY}px` }}>
        <polygon points={NEEDLE} fill={COLORS.primary} />
      </g>
      <circle cx={CX} cy={CY} r={11} fill={COLORS.primary} />
      <circle cx={CX} cy={CY} r={4} fill={COLORS.surface} />
    </svg>
  );
}
