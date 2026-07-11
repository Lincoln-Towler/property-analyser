// Python-compatible float formatting. Python's format(x, '.0f') rounds
// half-to-even ("banker's rounding"): format(58.5, '.0f') == '58', whereas
// JS (58.5).toFixed(0) == '59'. The commentary text is golden-mastered
// against the Python output, so formatting must match exactly.

export function roundHalfEven(x: number, digits = 0): number {
  const m = Math.pow(10, digits);
  const y = x * m;
  const floor = Math.floor(y);
  const diff = y - floor;
  let r: number;
  if (diff > 0.5) r = floor + 1;
  else if (diff < 0.5) r = floor;
  else r = floor % 2 === 0 ? floor : floor + 1;
  return r / m;
}

/** Mimic Python f"{x:.{digits}f}" */
export function fmtF(x: number, digits: number): string {
  const r = roundHalfEven(x, digits);
  return r.toFixed(digits);
}

/** Mimic Python f"{x:,.0f}" (thousands separators, banker's rounding) */
export function fmtComma0(x: number): string {
  const r = roundHalfEven(x, 0);
  return r.toLocaleString('en-US', { maximumFractionDigits: 0 });
}

/** Python round(x, 1) — half-even, used for breakdown display fields */
export function round1(x: number): number {
  return roundHalfEven(x, 1);
}
