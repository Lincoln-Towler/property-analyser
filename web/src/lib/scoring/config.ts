// Single source of truth for indicator thresholds — scoring AND UI read from
// here, ported verbatim from INDICATORS_CONFIG in property_analysis_dashboard.py.
// Field names stay snake_case to match the Python original, the golden-master
// fixtures, and the persisted breakdown jsonb.

export type Impact = 'inverse' | 'direct';

export interface IndicatorConfig {
  weight: number;
  impact: Impact;
  trend_matters: boolean;
  description: string;
  display_name: string;
  unit: string;
  optimal_below?: number;
  healthy_below?: number;
  warning_above?: number;
  danger_above?: number;
  optimal_min?: number;
  optimal_max?: number;
  optimal_above?: number;
  healthy_above?: number;
  healthy_range?: [number, number];
  oversupply_above?: number;
  deficit_below?: number;
  crisis_below?: number;
  weak_below?: number;
  strong_above?: number;
  /** Exact Python literal for thresholds interpolated into commentary text
   *  where str() differs from JS (e.g. 5.0 renders as "5.0" in Python, "5"
   *  in JS). Only needed for x.0 floats. */
  literals?: Partial<Record<'warning_above' | 'danger_above' | 'oversupply_above', string>>;
}

// Key order matters: the engine iterates in insertion order, same as the
// Python dict.
export const INDICATORS_CONFIG: Record<string, IndicatorConfig> = {
  interest_rate: {
    weight: 30,
    optimal_min: 2.5,
    optimal_max: 4.0,
    danger_above: 5.5,
    impact: 'inverse',
    trend_matters: true,
    description: 'RBA Cash Rate',
    display_name: 'Interest Rate (RBA Cash Rate)',
    unit: '%',
  },
  household_debt_gdp: {
    weight: 25,
    warning_above: 110,
    danger_above: 120,
    optimal_below: 100,
    impact: 'inverse',
    trend_matters: true,
    description: 'Debt to GDP Ratio',
    display_name: 'Household Debt to GDP',
    unit: '%',
  },
  rental_vacancy_rate: {
    weight: 20,
    optimal_below: 2.0,
    healthy_range: [1.5, 2.5],
    oversupply_above: 3.5,
    impact: 'inverse',
    trend_matters: true,
    description: 'Rental Vacancy',
    display_name: 'Rental Vacancy Rate',
    unit: '%',
  },
  building_approvals: {
    weight: 15,
    optimal_above: 240000,
    deficit_below: 180000,
    crisis_below: 160000,
    impact: 'direct',
    trend_matters: true,
    description: 'Annual Building Approvals',
    display_name: 'Building Approvals (Annual)',
    unit: '',
  },
  mortgage_stress_rate: {
    weight: 15,
    warning_above: 30,
    danger_above: 40,
    healthy_below: 25,
    impact: 'inverse',
    trend_matters: true,
    description: 'Mortgage Stress %',
    display_name: 'Mortgage Stress Rate',
    unit: '%',
  },
  unemployment_rate: {
    weight: 10,
    warning_above: 5.0,
    danger_above: 6.0,
    literals: { warning_above: '5.0', danger_above: '6.0' },
    healthy_below: 4.5,
    impact: 'inverse',
    trend_matters: true,
    description: 'Unemployment Rate',
    display_name: 'Unemployment Rate',
    unit: '%',
  },
  auction_clearance_rate: {
    weight: 10,
    healthy_above: 65,
    strong_above: 75,
    weak_below: 55,
    impact: 'direct',
    trend_matters: true,
    description: 'Auction Clearance %',
    display_name: 'Auction Clearance Rate',
    unit: '%',
  },
  credit_growth: {
    weight: 5,
    healthy_range: [0.3, 0.8],
    strong_above: 1.0,
    weak_below: 0.2,
    impact: 'direct',
    trend_matters: false,
    description: 'Monthly Credit Growth %',
    display_name: 'Credit Growth (Monthly)',
    unit: '%',
  },
  wage_growth: {
    weight: 5,
    healthy_range: [3.0, 4.0],
    strong_above: 4.0,
    weak_below: 2.5,
    impact: 'direct',
    trend_matters: false,
    description: 'Annual Wage Growth %',
    display_name: 'Wage Growth (Annual)',
    unit: '%',
  },
};

// 18.6-year Anderson cycle. Anchor decided June 2026: 2008 (GFC bottom),
// unifying the score (which always used 2008) and the cycle tracker page
// (which previously used 2011 — with 2008, 2026 sits at Year 18: Crash phase).
export const CYCLE_START_YEAR = 2008;
export const CYCLE_LENGTH = 18.6;

// Non-scored indicators that exist in the database and get display names
export const EXTRA_INDICATOR_NAMES: Record<string, string> = {
  mortgage_arrears_rate: 'Mortgage Arrears Rate',
  dwelling_supply_deficit: 'Dwelling Supply Deficit',
  population_growth: 'Population Growth',
};

// An indicator whose newest point is older than this is flagged stale in the
// audit and its confidence contribution is suspect (four indicators in the
// live DB have exactly one data point ever).
export const STALE_AFTER_DAYS = 90;
