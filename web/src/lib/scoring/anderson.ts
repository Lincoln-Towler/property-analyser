// Anderson 18.6-year cycle tracker, ported from show_anderson_tracker().
// Pure computation — no data dependency. Anchored at CYCLE_START_YEAR (2008),
// which unifies the tracker with the score's cycle multiplier: as of 2026
// this puts us at Year 18 — Crash/Correction phase. (The old Streamlit page
// used 2011 and showed Year 15 / Winner's Curse; the anchor decision was
// made deliberately in June 2026.)

import { CYCLE_START_YEAR, CYCLE_LENGTH } from './config';

export interface CyclePhase {
  start: number;
  end: number;
  name: string;
  descriptor: string;
  bullets: string[];
  status: 'completed' | 'in_progress' | 'approaching' | 'upcoming';
  calendarRange: string;
}

export interface AndersonState {
  currentYear: number;
  cycleStart: number;
  cycleLength: number;
  cyclePosition: number;
  cycleEndYear: number;
  phaseName: string;
  phaseDescriptor: string;
  phases: CyclePhase[];
  predictedPeakYear: number;
  crashStartYear: number;
  crashEndYear: number;
  accuracy: string;
  seenSignals: string[];
  warningSigns: string[];
  whatToWatch: string[];
  recommendations: string[];
  bestBuyStart: number;
  bestBuyEnd: number;
}

const PHASE_DEFS: Array<[number, number, string, string, string[]]> = [
  [0, 7, 'Phase 1', 'Recovery', ['Steady growth after crash', 'Rebuilding confidence']],
  [7, 9, 'Mid-Cycle', 'Slowdown', ['Correction/recession', 'Temporary pause']],
  [9, 14, 'Phase 2', 'Boom', ['Explosive growth', 'Credit expansion']],
  [14, 16, "Winner's Curse", 'Peak', ['Speculation peak', 'Final blow-off top']],
  [16, CYCLE_LENGTH, 'Crash', 'Correction', ['Major downturn', 'Best buying opportunity']],
];

export function andersonState(now: Date, cycleStart: number = CYCLE_START_YEAR): AndersonState {
  const currentYear = now.getUTCFullYear();
  const yearsElapsed = currentYear - cycleStart;
  let cyclePosition = yearsElapsed % CYCLE_LENGTH;
  if (cyclePosition < 0) cyclePosition += CYCLE_LENGTH;
  const cycleEndYear = cycleStart + Math.round(CYCLE_LENGTH);

  const phases: CyclePhase[] = PHASE_DEFS.map(([start, end, name, descriptor, bullets]) => {
    const startYear = cycleStart + Math.trunc(start);
    const endYear = cycleStart + Math.round(end);
    let status: CyclePhase['status'];
    if (cyclePosition >= end) status = 'completed';
    else if (start <= cyclePosition && cyclePosition < end) status = 'in_progress';
    else if (start - cyclePosition <= 1) status = 'approaching';
    else status = 'upcoming';
    return {
      start,
      end,
      name,
      descriptor,
      bullets,
      status,
      calendarRange: endYear > startYear ? `${startYear}-${endYear}` : String(startYear),
    };
  });

  const current = phases.find((p) => p.start <= cyclePosition && cyclePosition < p.end) ?? phases[phases.length - 1];

  const predictedPeakYear = cycleStart + 15;
  const crashStartYear = cycleStart + 16;
  const crashEndYear = cycleStart + Math.round(CYCLE_LENGTH);

  let accuracy: string;
  if (cyclePosition < 14) accuracy = '⏳ In Progress';
  else if (cyclePosition < 16) accuracy = '⚠️ Approaching Peak';
  else if (cyclePosition < CYCLE_LENGTH) accuracy = '🔻 Crash Phase';
  else accuracy = '✅ Cycle Complete';

  const seenSignals: string[] = [];
  if (cyclePosition >= 7) seenSignals.push(`Phase 1 Recovery (${cycleStart}-${cycleStart + 7}) ✓`);
  if (cyclePosition >= 9) seenSignals.push(`Mid-cycle dip (${cycleStart + 7}-${cycleStart + 9}) ✓`);
  if (cyclePosition >= 14) seenSignals.push(`Phase 2 Boom (${cycleStart + 9}-${cycleStart + 14}) ✓`);
  if (cyclePosition >= 16) seenSignals.push(`Winner's Curse (${cycleStart + 14}-${cycleStart + 16}) ✓`);
  if (cyclePosition >= CYCLE_LENGTH) seenSignals.push(`Crash/Reset (${cycleStart + 16}-${cycleEndYear}) ✓`);
  if (!seenSignals.length) seenSignals.push('Cycle recently started — no completed phases yet');

  let warningSigns: string[];
  let whatToWatch: string[];
  if (cyclePosition < 7) {
    warningSigns = ['Credit still tight', 'Sentiment negative', 'Low transaction volumes', 'Slow price recovery'];
    whatToWatch = ['First rate cuts signalling easing', 'Credit conditions loosening', 'Early price stabilisation', 'Investor confidence returning'];
  } else if (cyclePosition < 9) {
    warningSigns = ['Temporary price correction', 'Economic slowdown signals', 'Rising unemployment risk', 'Credit pullback'];
    whatToWatch = ['Recovery in clearance rates', 'Stabilising prices post-dip', 'Government stimulus measures', 'Phase 2 boom ignition signals'];
  } else if (cyclePosition < 14) {
    warningSigns = ['Leverage increasing rapidly', 'FOMO-driven buying spreading', 'Speculative activity rising', 'Regional markets overheating'];
    whatToWatch = ['Rate hike cycle starting', 'Credit tightening signals', 'Supply pipeline ramping up', 'Affordability stress building'];
  } else if (cyclePosition < 16) {
    warningSigns = ['Capital city price weakness', 'Extreme household debt levels', 'Universal bullish sentiment', 'Rate hikes squeezing borrowers', 'Speculation disconnected from fundamentals'];
    whatToWatch = ['Price falls in lead markets', 'Credit tightening accelerating', 'Unemployment ticking up', 'Forced sales increasing', 'Sentiment shift from greed to fear'];
  } else {
    warningSigns = ['Prices falling across markets', 'Credit freeze or severe tightening', 'Rising mortgage defaults', 'Forced sales and distressed assets'];
    whatToWatch = ['Price stabilisation signals (bottom)', 'Central bank pivoting to rate cuts', 'Credit loosening for first-home buyers', `Next cycle start (~${cycleEndYear})`];
  }

  const bestBuyStart = cycleStart + 17;
  const bestBuyEnd = cycleStart + 18;

  let recommendations: string[];
  if (cyclePosition < 7) {
    recommendations = [
      '🟢 **BUY** - early recovery phase, prices still low',
      '🟢 **ACCUMULATE** quality assets while sentiment is weak',
      '🟡 Watch for credit loosening as recovery strengthens',
    ];
  } else if (cyclePosition < 9) {
    recommendations = [
      '🟡 **SELECTIVE BUYING** - mid-cycle slowdown creates opportunities',
      '🟡 **HOLD** existing positions - temporary pause, not a crash',
      '🟢 Prepare for Phase 2 boom in next 1-2 years',
    ];
  } else if (cyclePosition < 14) {
    recommendations = [
      '🟢 **BUY** - Phase 2 boom, strong momentum',
      '🟡 Avoid overleveraging as cycle matures',
      `🟡 Start planning exit strategy for peak (${predictedPeakYear})`,
    ];
  } else if (cyclePosition < 16) {
    recommendations = [
      "🔴 **DO NOT BUY** aggressively - we're at/near the peak",
      '🟡 **HOLD** existing properties if you have equity buffers',
      `🟢 **PREPARE CASH** for the predicted ${crashStartYear}-${crashEndYear} downturn`,
      `🟢 **BEST BUYING OPPORTUNITY** predicted for ${bestBuyStart}-${bestBuyEnd}`,
    ];
  } else {
    recommendations = [
      '🔴 **DO NOT BUY** yet - crash phase in progress',
      '🟢 **HOLD CASH** - wait for clear bottom signals',
      `🟢 **BEST BUYING OPPORTUNITY** predicted for ${bestBuyStart}-${bestBuyEnd}`,
      '🟡 Watch for stabilisation in vacancy rates and clearance rates',
    ];
  }

  return {
    currentYear,
    cycleStart,
    cycleLength: CYCLE_LENGTH,
    cyclePosition,
    cycleEndYear,
    phaseName: current.name,
    phaseDescriptor: current.descriptor,
    phases,
    predictedPeakYear,
    crashStartYear,
    crashEndYear,
    accuracy,
    seenSignals,
    warningSigns,
    whatToWatch,
    recommendations,
    bestBuyStart,
    bestBuyEnd,
  };
}

/** Phase label used by the auto-commentary (same bands as the tracker). */
export function commentaryPhase(cyclePosition: number): string {
  if (cyclePosition < 7) return 'Recovery';
  if (cyclePosition < 9) return 'Mid-Cycle Slowdown';
  if (cyclePosition < 14) return 'Phase 2 Boom';
  if (cyclePosition < 16) return "Winner's Curse / Peak";
  return 'Crash / Correction';
}
