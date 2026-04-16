# Changelog

## April 2026 - Claude Code Session

### Score Data Audit (e4ceb4f)
- Added **Score Data Audit** expander to Ultimate Analysis page
- Shows row counts from current table, history table, and combined view
- Per-indicator breakdown: latest value, date, score, trend data points, volatility stats
- Source breakdown showing how many rows come from current vs history per indicator

### Combined View Fix - DROP before CREATE (47ccc6b)
- PostgreSQL `CREATE OR REPLACE VIEW` cannot drop columns from an existing view
- Changed to `DROP VIEW IF EXISTS` then `CREATE VIEW` to handle column schema changes
- Fixed error: "cannot drop columns from view"

### Combined View Fix - Missing Source Column (3c57561)
- Root cause found: `economic_indicators_history` has no `source` column
- View creation was failing silently, falling back to current-table-only
- This is why graphs were only showing current data, not historical
- Now dynamically inspects history table columns before building SQL
- Fixed `get_indicator_data()` using SQLite date syntax on PostgreSQL
- View creation errors now surface via `st.warning()` instead of silent catch

### Combined View Improvements (36457e0)
- Replaced `UNION` with `UNION ALL` + `ROW_NUMBER` deduplication
- Current table data takes priority over history when same `(date, indicator_name)` exists
- Added SQLite support for the combined view (was PostgreSQL-only)
- Fixed global view cache to re-check for history table each call
- Included `source` column in the combined view

### UndefinedColumn Fix (5e2617d)
- Simplified combined view to 3 columns (`date`, `indicator_name`, `value`)
- View/Edit Data tab now uses base table directly (needs `id` and `source` for CRUD)
- Export report handles missing `source` column gracefully

### History Table Integration Attempts (4bf19f7, 176f780, edb2593, ffc0ca0)
- Series of fixes to integrate `economic_indicators_history` table
- Progressed through: inline subquery -> CREATE VIEW -> column mapping -> dynamic inspection
- Each iteration addressed a new `UndefinedColumn` error variant

### Data Management Updates (64ab712)
- Added leading indicator CSV imports to Data Management tab
- Added delete functionality for all leading indicator data types
- CSV Import tab now supports 6 data types
- View/Edit Data tab now supports 10 data types

### Code Simplification (c46f5e2)
- Extracted 3 helper functions: `get_ph()`, `db_upsert()`, `csv_import_section()`
- Eliminated 9 repeated placeholder detections, 5 paired pg/sqlite blocks, 4 CSV import workflows
- Net reduction: -106 lines

### Leading Indicators Implementation (b4085af area)
- Built 5 new modules: Infrastructure Tracker, Migration Monitor, Jobs Growth Tracker, Supply/Demand Analyzer, Suburb Scorer
- Created 5 new database tables
- Added navigation entry and full UI with tabs
- Created 4 CSV template files
- Added scoring functions for each module (0-10 scale, 0-50 composite)

### Score Calculation Fix (earlier)
- Fixed v3 score using already cycle-adjusted score from v2
- Now extracts true pre-cycle `base_score` from v2 breakdown
- Applies volatility penalty, then cycle multiplier, in correct order

## Pre-Claude Code

Initial development via GitHub web editor:
- Core dashboard with economic indicators, property data, market commentary
- Market score v1/v2/v3 calculation pipeline
- Location comparison and Anderson Cycle Tracker
- Streamlit Cloud deployment with Supabase integration

## Related Notes

- [[Property Analyser - Architecture]]
- [[Score Calculation Pipeline]]
- [[Economic Indicators Integration]]
- [[Leading Indicators Modules]]
