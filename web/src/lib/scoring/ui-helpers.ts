// UI helpers derived from config thresholds, ported from
// indicator_target_text / indicator_progress_fraction / indicator_status.

import type { IndicatorConfig } from './config';
import { sentenceCase, stripLeadingEmoji } from '../format';

export function indicatorTargetText(config: IndicatorConfig): string | null {
  const unit = config.unit ?? '';
  if (config.impact === 'inverse') {
    if (config.optimal_below !== undefined) return `<${config.optimal_below}${unit}`;
    if (config.healthy_below !== undefined) return `<${config.healthy_below}${unit}`;
    if (config.optimal_max !== undefined && config.optimal_min !== undefined)
      return `${config.optimal_min}-${config.optimal_max}${unit}`;
  } else if (config.impact === 'direct') {
    if (config.optimal_above !== undefined) {
      const v = config.optimal_above;
      return v >= 1000 ? `>${v.toLocaleString('en-US')}${unit}` : `>${v}${unit}`;
    }
    if (config.healthy_above !== undefined) return `>${config.healthy_above}${unit}`;
    if (config.healthy_range !== undefined) {
      const [lo, hi] = config.healthy_range;
      return `${lo}-${hi}${unit}`;
    }
  }
  return null;
}

/**
 * The target zone as numbers, for shading on charts. Mirrors
 * indicatorTargetText so the shaded band always matches the "Target:" text.
 * A null bound means the zone is open-ended on that side.
 */
export function targetBand(config: IndicatorConfig): [number | null, number | null] | null {
  if (config.impact === 'inverse') {
    if (config.optimal_below !== undefined) return [null, config.optimal_below];
    if (config.healthy_below !== undefined) return [null, config.healthy_below];
    if (config.optimal_max !== undefined && config.optimal_min !== undefined)
      return [config.optimal_min, config.optimal_max];
  } else if (config.impact === 'direct') {
    if (config.optimal_above !== undefined) return [config.optimal_above, null];
    if (config.healthy_above !== undefined) return [config.healthy_above, null];
    if (config.healthy_range !== undefined) return [config.healthy_range[0], config.healthy_range[1]];
  }
  return null;
}

/**
 * The first threshold where the indicator turns bad, labelled with what it
 * actually is — so a line drawn at the 30% warning level reads "warning >30%",
 * never the 25% target.
 */
export function warningLine(config: IndicatorConfig): { value: number; label: string } | null {
  const unit = config.unit ?? '';
  const fmt = (v: number) => (v >= 1000 ? v.toLocaleString('en-US') : String(v));
  if (config.impact === 'inverse') {
    if (config.warning_above !== undefined)
      return { value: config.warning_above, label: `warning >${fmt(config.warning_above)}${unit}` };
    if (config.oversupply_above !== undefined)
      return { value: config.oversupply_above, label: `oversupply >${fmt(config.oversupply_above)}${unit}` };
    if (config.danger_above !== undefined)
      return { value: config.danger_above, label: `danger >${fmt(config.danger_above)}${unit}` };
  } else if (config.impact === 'direct') {
    if (config.weak_below !== undefined)
      return { value: config.weak_below, label: `weak <${fmt(config.weak_below)}${unit}` };
    if (config.deficit_below !== undefined)
      return { value: config.deficit_below, label: `deficit <${fmt(config.deficit_below)}${unit}` };
    if (config.crisis_below !== undefined)
      return { value: config.crisis_below, label: `crisis <${fmt(config.crisis_below)}${unit}` };
  }
  return null;
}

/** Short status phrase for a reading ('Above warning', 'In optimal zone'),
 *  taken from the commentary classifier's note so the two never disagree. */
export function statusPhrase(note: string): string {
  const phrase = note.replace(/\s*\(.*$/, '').trim();
  return phrase.charAt(0).toUpperCase() + phrase.slice(1);
}

export function indicatorProgressFraction(value: number | null, config: IndicatorConfig): number {
  if (value === null) return 0;
  if (config.impact === 'inverse') {
    const ceiling =
      config.danger_above ?? config.oversupply_above ?? config.warning_above ?? config.optimal_max;
    if (ceiling) return Math.max(0, Math.min(1, value / (ceiling * 1.1)));
  } else if (config.impact === 'direct') {
    const ceiling = config.optimal_above ?? config.strong_above ?? config.healthy_above;
    if (ceiling) return Math.max(0, Math.min(1, value / ceiling));
  }
  return 0.5;
}

export type IndicatorStatus = 'success' | 'warning' | 'danger';

export function indicatorStatus(value: number | null, config: IndicatorConfig): IndicatorStatus {
  if (value === null) return 'warning';
  if (config.impact === 'inverse') {
    if (config.danger_above !== undefined && value > config.danger_above) return 'danger';
    if (config.oversupply_above !== undefined && value > config.oversupply_above) return 'danger';
    if (config.warning_above !== undefined && value > config.warning_above) return 'warning';
    if (config.optimal_below !== undefined && value < config.optimal_below) return 'success';
    if (config.healthy_below !== undefined && value < config.healthy_below) return 'success';
    if (
      config.optimal_min !== undefined &&
      config.optimal_min <= value &&
      value <= (config.optimal_max ?? config.optimal_min)
    )
      return 'success';
    return 'warning';
  }
  if (config.impact === 'direct') {
    if (config.crisis_below !== undefined && value < config.crisis_below) return 'danger';
    if (config.weak_below !== undefined && value < config.weak_below) return 'danger';
    if (config.deficit_below !== undefined && value < config.deficit_below) return 'warning';
    if (config.strong_above !== undefined && value >= config.strong_above) return 'success';
    if (config.optimal_above !== undefined && value >= config.optimal_above) return 'success';
    if (config.healthy_above !== undefined && value >= config.healthy_above) return 'success';
    return 'warning';
  }
  return 'warning';
}

const INDICATOR_LABELS: Record<string, string> = {
  interest_rate: 'Cash rate',
  household_debt_gdp: 'Household debt to GDP',
  rental_vacancy_rate: 'Rental vacancy',
  building_approvals: 'Building approvals',
  mortgage_stress_rate: 'Mortgage stress',
  unemployment_rate: 'Unemployment',
  auction_clearance_rate: 'Auction clearance',
  credit_growth: 'Credit growth',
  wage_growth: 'Wage growth',
  mortgage_arrears_rate: 'Mortgage arrears',
  dwelling_supply_deficit: 'Dwelling supply deficit',
  population_growth: 'Population growth',
};

/** Short sentence-case name for the UI. The config's display_name stays as it
 *  is, because the parity-tested commentary uses it. */
export function indicatorLabel(name: string): string {
  return INDICATOR_LABELS[name] ?? sentenceCase(name.replace(/_/g, ' '));
}

/** "Late cycle, year 18 of 18.6", from the same numbers the score uses. */
export function cycleCaption(multiplier: number | undefined, position: number | undefined): string | null {
  if (position === undefined) return null;
  const phase = multiplier === 0.9 ? 'Late cycle' : multiplier === 1.1 ? 'Early cycle' : multiplier === 1.05 ? 'Mid cycle' : 'Cycle';
  return `${phase}, year ${Math.floor(position)} of 18.6`;
}

/** An engine risk warning as calm text, with the status it carries. */
export function riskNotice(warning: string): { status: 'success' | 'warning' | 'danger'; text: string } {
  const text = sentenceCase(stripLeadingEmoji(warning));
  if (/^opportunity/i.test(text)) return { status: 'success', text };
  if (/^oversupply/i.test(text)) return { status: 'warning', text };
  return { status: 'danger', text };
}

/** The newest daily snapshot at least `days` before the latest one: where the
 *  meter's trail starts. Null when the history is shorter than that. */
export function scoreDaysAgo(
  rows: Array<{ score_date: string; final_score: number }>,
  days = 30,
): { score: number; date: string } | null {
  if (!rows.length) return null;
  const sorted = [...rows].sort((a, b) => a.score_date.localeCompare(b.score_date));
  const cutoff = Date.parse(sorted[sorted.length - 1].score_date) - days * 86_400_000;
  let found: { score_date: string; final_score: number } | null = null;
  for (const r of sorted) {
    if (Date.parse(r.score_date) > cutoff) break;
    found = r;
  }
  return found ? { score: Number(found.final_score), date: found.score_date } : null;
}
