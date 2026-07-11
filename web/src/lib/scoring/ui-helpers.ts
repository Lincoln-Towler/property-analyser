// UI helpers derived from config thresholds, ported from
// indicator_target_text / indicator_progress_fraction / indicator_status.

import type { IndicatorConfig } from './config';

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
