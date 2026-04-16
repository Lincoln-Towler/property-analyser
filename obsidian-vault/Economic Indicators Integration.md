# Economic Indicators Integration

## Two-Table Architecture

Economic indicator data comes from two sources:

### `economic_indicators` (Current / Manual)
- Data entered manually via the Data Management tab or CSV import
- Has `id`, `date`, `indicator_name`, `value`, `source`, `created_at`
- `UNIQUE(date, indicator_name)` constraint
- Supports full CRUD (create, read, update, delete) via the UI

### `economic_indicators_history` (Automated / n8n)
- Fed by **n8n workflow automation** into Supabase
- Has `id`, `date`, `indicator_name`, `value`, `recorded_at`
- **No `source` or `created_at` columns**
- Read-only from the app's perspective
- Only exists in PostgreSQL (Supabase), not SQLite

## Combined View

`economic_indicators_view(conn)` creates and caches a combined view:

```sql
-- PostgreSQL
DROP VIEW IF EXISTS economic_indicators_combined;
CREATE VIEW economic_indicators_combined AS
SELECT date, indicator_name, value, source
FROM (
    SELECT date, indicator_name, value, source,
           ROW_NUMBER() OVER (
               PARTITION BY date, indicator_name
               ORDER BY priority ASC
           ) AS rn
    FROM (
        SELECT date, indicator_name, value, source, 1 AS priority
        FROM economic_indicators
        UNION ALL
        SELECT date, indicator_name, value,
               'history' AS source, 2 AS priority
        FROM economic_indicators_history
    ) combined
) ranked
WHERE rn = 1
```

### Key Design Decisions

1. **Current data wins**: When the same `(date, indicator_name)` exists in both tables, the current table row is kept (priority 1 vs 2)
2. **Dynamic column detection**: Before creating the view, inspects the history table's columns via `information_schema` to check if `source` exists. If not, substitutes `'history'` as a literal
3. **DROP then CREATE**: PostgreSQL's `CREATE OR REPLACE VIEW` cannot drop columns from an existing view. Must `DROP VIEW IF EXISTS` first
4. **SQLite support**: Uses `NOT EXISTS` subquery instead of window functions for the same dedup logic
5. **Smart caching**: Caches the view name but re-checks whether the history table exists each call, so mid-session imports are detected
6. **Error visibility**: View creation failures show a `st.warning()` instead of being silently swallowed

## Which Queries Use What

### Read from Combined View (10 queries)
All scoring, trend, volatility, charting, and export queries go through `{ei_view}`:
- `calculate_market_score_v2()` - latest value per indicator
- `get_indicator_trend()` - 2 most recent values within 3 months
- `calculate_volatility_penalty()` - all values within 6 months
- `show_economic_indicators()` - current values display + historical trend charts
- Export data - `SELECT * FROM {ei_view}`

### Read from Base Table (1 query)
The **View/Edit Data** tab queries `economic_indicators` directly because it needs `id` for delete and `source` for editing. This is intentional - you can only edit/delete manually entered data.

### Write to Base Table (5 queries)
All inserts, updates, and deletes target `economic_indicators` only:
- Manual entry: `INSERT ... ON CONFLICT DO UPDATE`
- CSV import: `INSERT ... ON CONFLICT DO UPDATE`
- Edit entry: `UPDATE ... WHERE id = %s`
- Delete entry: `DELETE ... WHERE id = %s`

## Tracked Indicators

| Indicator Key | Description | Source |
|--------------|-------------|--------|
| interest_rate | RBA Cash Rate | RBA |
| household_debt_gdp | Household Debt to GDP Ratio | ABS/RBA |
| mortgage_stress_rate | % of households in mortgage stress | Various |
| rental_vacancy_rate | Rental vacancy rate | SQM Research |
| auction_clearance_rate | Weekly auction clearance rate | Domain/REA |
| unemployment_rate | National unemployment rate | ABS |
| building_approvals | Annual dwelling approvals count | ABS |
| credit_growth | Monthly housing credit growth % | RBA |
| mortgage_arrears_rate | Mortgage arrears rate | S&P/RBA |
| dwelling_supply_deficit | Estimated dwelling supply shortfall | Various |
| population_growth | Annual population growth % | ABS |
| wage_growth | Annual wage growth % | ABS |

## Past Issues & Fixes

| Issue | Root Cause | Fix |
|-------|-----------|-----|
| `UndefinedColumn` errors | History table lacks `source` column, view referenced it | Dynamic column inspection before SQL generation |
| View creation silently failing | Bare `except` swallowed errors, fell back to base table | Added `st.warning()` for error visibility |
| `cannot drop columns from view` | `CREATE OR REPLACE VIEW` can't remove columns | Changed to `DROP VIEW IF EXISTS` then `CREATE VIEW` |
| Graphs showing only current data | View creation was failing silently (see above) | All fixes above combined |
| `get_indicator_data()` broken on Postgres | Used SQLite `date('now')` syntax | Added `is_postgres()` branch with `CURRENT_DATE - INTERVAL` |

## Related Notes

- [[Property Analyser - Architecture]]
- [[Score Calculation Pipeline]]
