# Score Calculation Pipeline

The market score flows through three phases: **v2 (base)** -> **v3 (volatility + confidence)**.

## Pipeline Flow

```
v2: Weighted Indicators -> Trend Adjustment -> Composite Risks -> Cycle Adjustment
         |                                                              |
         v                                                              v
    base_score (0-100)                                         final_score_v2
         |
         v
v3: base_score -> Volatility Penalty -> Cycle Adjustment -> Confidence Interval
                                                |
                                                v
                                         final_score_v3 (displayed)
```

## Phase 1: Weighted Indicators (`calculate_market_score_v2`)

Each indicator gets a score (0-100) based on value thresholds, then weighted.

| Indicator | Weight | Impact | Key Thresholds |
|-----------|--------|--------|----------------|
| interest_rate | 30 | inverse | optimal 2.5-4.0, danger >5.5 |
| household_debt_gdp | 25 | inverse | optimal <100, warning >110, danger >120 |
| rental_vacancy_rate | 20 | inverse | optimal <2.0, oversupply >3.5 |
| building_approvals | 15 | direct | optimal >240k, deficit <180k, crisis <160k |
| mortgage_stress_rate | 15 | inverse | healthy <25, warning >30, danger >40 |
| unemployment_rate | 10 | inverse | healthy <4.5, warning >5.0, danger >6.0 |
| auction_clearance_rate | 10 | direct | healthy >65, strong >75, weak <55 |
| credit_growth | 5 | direct | healthy 0.3-0.8, strong >1.0 |
| wage_growth | 5 | direct | healthy 3.0-4.0, strong >4.0 |

**Total weight: 135** (not 100 - the base_score is `weighted_sum / total_weight`)

### Impact Types
- **inverse**: Lower value = better for property market (higher score)
- **direct**: Higher value = better for property market (higher score)

### Score Mapping
| Condition | Score |
|-----------|-------|
| Optimal / Strong | 75-85 |
| Healthy | 65-75 |
| Neutral (default) | 50 |
| Warning / Weak | 35-40 |
| Danger / Crisis | 15 |

## Phase 2: Trend Adjustment

For indicators where `trend_matters=True` (7 of 9), `get_indicator_trend()` fetches the **2 most recent values within the last 3 months** from the combined view.

- Change > 5%: `rising` or `falling`
- Change <= 5%: `stable`

| Impact | Rising | Falling | Stable |
|--------|--------|---------|--------|
| inverse | -10 | +10 | 0 |
| direct | +10 | -10 | 0 |

Final indicator score = `base + trend_bonus`, clamped to [0, 100].

## Phase 3: Composite Risk Overrides

After calculating base_score, composite conditions can override it:

| Condition | Triggers | Effect |
|-----------|----------|--------|
| Perfect Storm | debt >120 AND rate >5.5 AND stress >35 | cap at 25 |
| Crisis | stress >40 AND unemployment >6.0 | cap at 30 |
| Supply Squeeze | vacancy <1.5 AND approvals <180k AND clearance >70 | +15 |
| Oversupply | vacancy >3.5 AND approvals >250k | -15 |

## Phase 4: 18.6-Year Cycle Adjustment

```
cycle_start_year = 2008 (GFC bottom)
cycle_position = (current_year - 2008) % 18.6
```

| Cycle Position | Zone | Multiplier |
|---------------|------|------------|
| 0-4 | Bottom (Opportunity) | 1.10 |
| 7-11 | Mid (Growth) | 1.05 |
| 14-18 | Peak (Caution) | 0.90 |
| Other | Neutral | 1.00 |

**2026**: position = 18.0 -> Peak Zone -> 0.90x multiplier

## Phase 5: Volatility (`calculate_market_score_v3`)

v3 takes the **pre-cycle base_score** from v2 (not the final), then applies volatility.

`calculate_volatility_penalty()` runs on 5 key indicators over the last **6 months**:
- interest_rate, household_debt_gdp, mortgage_stress_rate, rental_vacancy_rate, unemployment_rate

Calculates coefficient of variation: `CV = (std_dev / mean) * 100`

| CV | Level | Penalty |
|----|-------|---------|
| >20% | extreme | -10 |
| >10% | high | -5 |
| >5% | moderate | -2 |
| <=5% | low | 0 |

Needs 3+ data points, otherwise returns `insufficient_data` (no penalty).

## Phase 6: Sub-Scores (`calculate_sub_scores`)

Four pillar scores (0-100 each), displayed on Ultimate Analysis page:

1. **Affordability** - interest_rate, household_debt_gdp, mortgage_stress_rate
2. **Supply/Demand** - rental_vacancy_rate, building_approvals, auction_clearance_rate
3. **Financial Health** - unemployment_rate, mortgage_stress_rate, credit_growth
4. **Market Momentum** - count of rising vs falling trends

## Phase 7: Confidence Interval

Based on data_completeness + volatility levels:
- **High**: >70% completeness, mostly low volatility
- **Medium**: 50-70% completeness or some high volatility
- **Low**: <50% completeness or extreme volatility

## Signal Thresholds

| Score | Signal |
|-------|--------|
| >= 75 | Strong Buy |
| >= 60 | Buy |
| >= 50 | Moderate Buy |
| >= 40 | Hold |
| >= 30 | Caution |
| < 30 | Wait |

## Score Data Audit

The Ultimate Analysis page includes a **Score Data Audit** expander that shows:
- Row counts: current table, history table, combined view
- Per-indicator: latest value, date, score, trend data points, volatility stats
- Source breakdown: how many rows from current vs history per indicator

## Related Notes

- [[Property Analyser - Architecture]]
- [[Economic Indicators Integration]]
