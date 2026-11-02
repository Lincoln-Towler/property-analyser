// Display formatting shared by every page, so dates, money and indicator
// values read the same everywhere. Pure functions — safe in server and
// client components. Dates are parsed from their ISO parts rather than via
// Date objects so a 'YYYY-MM-DD' never shifts a day across time zones.

const MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];

function isoParts(iso: string): [number, number, number] | null {
  const m = /^(\d{4})-(\d{2})-(\d{2})/.exec(iso);
  return m ? [Number(m[1]), Number(m[2]), Number(m[3])] : null;
}

/** '2026-11-01' → '1 Nov 2026' */
export function fmtDate(iso: string | null | undefined): string {
  if (!iso) return '—';
  const p = isoParts(iso);
  if (!p) return iso;
  return `${p[2]} ${MONTHS[p[1] - 1]} ${p[0]}`;
}

/** '2026-11-01' → "Nov '26" */
export function fmtMonthYear(iso: string | null | undefined): string {
  if (!iso) return '—';
  const p = isoParts(iso);
  if (!p) return iso;
  return `${MONTHS[p[1] - 1]} '${String(p[0]).slice(2)}`;
}

/** ISO timestamp → '2 Nov 2026, 02:53 UTC' */
export function fmtDateTimeUtc(iso: string | null | undefined): string {
  if (!iso) return '—';
  const time = /T(\d{2}:\d{2})/.exec(iso)?.[1];
  return time ? `${fmtDate(iso)}, ${time} UTC` : fmtDate(iso);
}

/** Epoch ms → ISO date, for chart axes that work in timestamps. */
export function tsToIso(ts: number): string {
  return new Date(ts).toISOString().slice(0, 10);
}

/** 'YYYY-MM-DD' → epoch ms at UTC midnight. */
export function isoToTs(iso: string): number {
  const p = isoParts(iso);
  return p ? Date.UTC(p[0], p[1] - 1, p[2]) : Date.parse(iso);
}

/** Australian property prices: 1150000 → '$1.15M', 955000 → '$955K'. */
export function fmtPriceCompact(value: number): string {
  if (Math.abs(value) >= 1_000_000) {
    return `$${(value / 1_000_000).toFixed(2).replace(/\.?0+$/, '')}M`;
  }
  if (Math.abs(value) >= 1000) return `$${Math.round(value / 1000)}K`;
  return `$${Math.round(value)}`;
}

/** Whole dollars with separators: 77704.64 → '$77,705'. */
export function fmtUsd(value: number): string {
  return `$${Math.round(value).toLocaleString('en-US')}`;
}

/** Chart axis label for money: 83000 → '$83K', 1250000 → '$1.25M'. */
export function fmtMoneyTick(value: number): string {
  return fmtPriceCompact(value);
}

/** Indicator reading with its unit: 202729 → '202,729', 4.6 → '4.6%'. */
export function fmtIndicator(value: number | null | undefined, unit = ''): string {
  if (value === null || value === undefined) return '—';
  if (Math.abs(value) >= 1000) return `${Math.round(value).toLocaleString('en-US')}${unit}`;
  return `${Number(value.toFixed(2))}${unit}`;
}

/** Compact axis tick for indicator values: 202729 → '203k'. */
export function fmtIndicatorTick(value: number): string {
  if (Math.abs(value) >= 1000) return `${Math.round(value / 1000)}k`;
  return String(Number(value.toFixed(2)));
}

/** Signal strings from the engine lead with an emoji ('🟡 Hold'); the UI
 *  renders its own coloured badge, so drop the glyph for display. */
export function stripLeadingEmoji(text: string): string {
  return text.replace(/^[^\p{L}\p{N}]+/u, '').trim();
}

/**
 * Month-aligned ticks for a time axis between two timestamps, at most
 * `maxTicks` of them. Steps through 1/2/3/6/12-month spacing; very short
 * spans fall back to 7/14-day spacing.
 */
export function timeTicks(minTs: number, maxTs: number, maxTicks = 6): number[] {
  if (!(maxTs > minTs)) return [minTs];
  const spanDays = (maxTs - minTs) / 86_400_000;
  if (spanDays < 45) {
    const step = spanDays <= 21 ? 7 : 14;
    const ticks: number[] = [];
    for (let t = minTs; t <= maxTs; t += step * 86_400_000) ticks.push(t);
    return ticks;
  }
  const start = new Date(minTs);
  for (const stepMonths of [1, 2, 3, 6, 12, 24]) {
    const ticks: number[] = [];
    const y = start.getUTCFullYear();
    // Months counted from January of the start year; aligning to multiples of
    // the step puts quarterly ticks on Jan/Apr/Jul/Oct and yearly on 1 Jan.
    let m = Math.ceil(start.getUTCMonth() / stepMonths) * stepMonths;
    for (;;) {
      const t = Date.UTC(y + Math.floor(m / 12), m % 12, 1);
      if (t > maxTs) break;
      if (t >= minTs) ticks.push(t);
      m += stepMonths;
    }
    if (ticks.length <= maxTicks) return ticks.length ? ticks : [minTs, maxTs];
  }
  return [minTs, maxTs];
}

/**
 * A y-axis range with round tick steps (1/2/2.5/5 × 10ⁿ) covering every value
 * passed in — readings plus any target edge that must stay in view.
 */
export function niceScale(values: number[], targetTicks = 4): { domain: [number, number]; ticks: number[] } {
  const finite = values.filter((v) => Number.isFinite(v));
  let min = Math.min(...finite);
  let max = Math.max(...finite);
  if (!finite.length) return { domain: [0, 1], ticks: [0, 1] };
  if (min === max) {
    const pad = Math.abs(min) * 0.05 || 1;
    min -= pad;
    max += pad;
  }
  const raw = (max - min) / targetTicks;
  const mag = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => s >= raw) ?? 10 * mag;
  const lo = Math.floor(min / step) * step;
  const hi = Math.ceil(max / step) * step;
  const ticks: number[] = [];
  for (let t = lo; t <= hi + step / 1e6; t += step) ticks.push(Number(t.toPrecision(12)));
  return { domain: [ticks[0], ticks[ticks.length - 1]], ticks };
}

/** Everything a Recharts time x-axis needs for a set of ISO dates. */
export function timeAxis(dates: string[], maxTicks = 5) {
  const ts = dates.map(isoToTs);
  const min = Math.min(...ts);
  const max = Math.max(...ts);
  const spanDays = (max - min) / 86_400_000;
  return {
    domain: [min, max] as [number, number],
    ticks: timeTicks(min, max, maxTicks),
    format: (t: number) => fmtTimeTick(t, spanDays),
  };
}

/** Tick label matching timeTicks' spacing. */
export function fmtTimeTick(ts: number, spanDays: number): string {
  const iso = tsToIso(ts);
  if (spanDays < 45) {
    const p = isoParts(iso)!;
    return `${p[2]} ${MONTHS[p[1] - 1]}`;
  }
  return fmtMonthYear(iso);
}
