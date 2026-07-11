// Auto-generated market commentary, ported from generate_auto_commentary()
// and _classify_indicator(). Output is golden-mastered against the Python
// original, so string formatting mimics Python f-strings exactly (banker's
// rounding via pyformat).

import { INDICATORS_CONFIG, IndicatorConfig, CYCLE_START_YEAR, CYCLE_LENGTH } from './config';
import { fmtF, fmtComma0 } from './pyformat';
import { commentaryPhase } from './anderson';
import type { SeriesMap } from './engine';

export function classifyIndicator(
  value: number,
  cfg: IndicatorConfig,
): ['bullish' | 'bearish' | 'neutral', string] {
  const unit = cfg.unit ?? '';
  const impact = cfg.impact;
  const fmt = Math.abs(value) >= 1000 ? `${fmtComma0(value)}${unit}` : `${fmtF(value, 2)}${unit}`;

  if (impact === 'inverse') {
    if (cfg.danger_above !== undefined && value > cfg.danger_above)
      return ['bearish', `at danger level (${fmt}, >${cfg.literals?.danger_above ?? cfg.danger_above}${unit})`];
    if (cfg.warning_above !== undefined && value > cfg.warning_above)
      return ['bearish', `above warning (${fmt}, >${cfg.literals?.warning_above ?? cfg.warning_above}${unit})`];
    if (cfg.oversupply_above !== undefined && value > cfg.oversupply_above)
      return ['bearish', `oversupply (${fmt}, >${cfg.literals?.oversupply_above ?? cfg.oversupply_above}${unit})`];
    if (cfg.optimal_below !== undefined && value < cfg.optimal_below)
      return ['bullish', `in optimal zone (${fmt})`];
    if (cfg.healthy_below !== undefined && value < cfg.healthy_below)
      return ['bullish', `healthy (${fmt})`];
    if (cfg.optimal_min !== undefined && cfg.optimal_min <= value && value <= (cfg.optimal_max ?? cfg.optimal_min))
      return ['bullish', `in optimal range (${fmt})`];
    return ['neutral', `at ${fmt}`];
  }

  if (impact === 'direct') {
    if (cfg.crisis_below !== undefined && value < cfg.crisis_below)
      return ['bearish', `at crisis level (${fmt})`];
    if (cfg.deficit_below !== undefined && value < cfg.deficit_below)
      return ['bearish', `deficit (${fmt})`];
    if (cfg.weak_below !== undefined && value < cfg.weak_below)
      return ['bearish', `weak (${fmt})`];
    if (cfg.optimal_above !== undefined && value >= cfg.optimal_above)
      return ['bullish', `above optimal (${fmt})`];
    if (cfg.healthy_above !== undefined && value >= cfg.healthy_above)
      return ['bullish', `healthy (${fmt})`];
    if (cfg.healthy_range !== undefined) {
      const [lo, hi] = cfg.healthy_range;
      if (lo <= value && value <= hi) return ['bullish', `healthy (${fmt})`];
      if (value < lo) return ['bearish', `below healthy (${fmt})`];
      return ['neutral', `above healthy (${fmt})`];
    }
    return ['neutral', `at ${fmt}`];
  }

  return ['neutral', `at ${fmt}`];
}

function latestValue(series: SeriesMap, name: string): number | null {
  const points = series[name];
  if (!points || !points.length) return null;
  return [...points].sort((a, b) => (a.date < b.date ? 1 : -1))[0].value;
}

/**
 * Build the commentary markdown.
 * `score`/`signal` come from calculateMarketScoreV3 (pass null when unavailable).
 * `cycleStart` is parameterised so the parity tests can pin it to the old
 * app's 2011 anchor while production uses CYCLE_START_YEAR.
 */
export function generateAutoCommentary(
  series: SeriesMap,
  score: number | null,
  signal: string | null,
  now: Date,
  cycleStart: number = CYCLE_START_YEAR,
): string {
  const bullish: string[] = [];
  const bearish: string[] = [];
  const neutral: string[] = [];

  for (const [key, cfg] of Object.entries(INDICATORS_CONFIG)) {
    const value = latestValue(series, key);
    if (value === null) continue;
    const [category, note] = classifyIndicator(value, cfg);
    const display = cfg.display_name ?? key.replace(/_/g, ' ');
    const entry = `- **${display}**: ${note}`;
    if (category === 'bullish') bullish.push(entry);
    else if (category === 'bearish') bearish.push(entry);
    else neutral.push(entry);
  }

  let cyclePosition = (now.getUTCFullYear() - cycleStart) % CYCLE_LENGTH;
  if (cyclePosition < 0) cyclePosition += CYCLE_LENGTH;
  const phase = commentaryPhase(cyclePosition);

  let recommendation: string;
  if (score === null) {
    recommendation = 'Insufficient data — add more indicator values in Data Management.';
  } else if (score >= 75) {
    recommendation = 'Strong fundamentals across the board — consider acquisitions in well-located suburbs.';
  } else if (score >= 60) {
    recommendation = 'Constructive conditions — continue buying selectively with a margin of safety.';
  } else if (score >= 50) {
    recommendation = 'Balanced conditions — focus on quality over quantity and stress-test cash flow.';
  } else if (score >= 40) {
    recommendation = 'Mixed signals — hold existing positions, avoid aggressive leverage, watch trends.';
  } else if (score >= 30) {
    recommendation = 'Elevated risks — prioritise cash reserves and avoid speculative purchases.';
  } else {
    recommendation = 'Downturn risk high — hold cash, wait for clearer bottom signals.';
  }

  const lines: string[] = ['**Current Market Assessment**', ''];
  if (score !== null) {
    lines.push(`Market Score: **${fmtF(score, 0)}/100** — ${signal}`);
    lines.push(`Cycle Position: **Year ${fmtF(cyclePosition, 1)} of ${CYCLE_LENGTH}** (${phase})`);
    lines.push('');
  }
  if (bearish.length) {
    lines.push('**Bearish Factors:**', ...bearish, '');
  }
  if (bullish.length) {
    lines.push('**Bullish Factors:**', ...bullish, '');
  }
  if (neutral.length) {
    lines.push('**Neutral / Watch:**', ...neutral, '');
  }
  lines.push(`**Recommendation:** ${recommendation}`);
  lines.push('');
  const iso = now.toISOString().slice(0, 10);
  lines.push(`_Auto-generated from latest indicator data on ${iso}._`);

  return lines.join('\n');
}
