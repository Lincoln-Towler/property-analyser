# Changelog

## November 2026 - Data pipelines, BTC review loop, UI rework

### Data pipelines (2 Nov, see [[Data Pipelines Handoff 2026-11-02]])
- **Diagnosis:** only the cash rate was still updating. The credit branch requested an RBA page that usually isn't published yet, and its 404 stopped every later branch. Several parsers and URLs were outdated, and the cash-rate "Update row" node was overwriting the unemployment row.
- **Economic Indicators workflow fixed:** new ABS API host, correct series (seasonally adjusted unemployment, housing credit), cash-rate parser, credit-page fallback, isolated failures, range checks; "Update row" nodes removed.
- **Data backfilled and cleaned** after a full backup (migration `0007`): ~45 sourced rows, official figures aligned, every series dated first-of-period with one point per period.
- **Location data:** new SQM workflow replaces Hotspotting for vacancy, yield, growth and asking price on all 11 locations.
- **BTC:** `btc_video_analysis` and `btc_flag_suggestions` tables with approve-to-apply, supersede and 20-flag-limit triggers; `get_btc_signal` caps the flags' combined effect at ±2 (migrations `0008`–`0010`). First review cycle: 11 suggestions applied, 8 rejected.
- **Repo:** workflow source, tests and importable JSON committed under `n8n/`; `CLAUDE.md` added for future Claude sessions.

### Cash rate and price chart (4 Nov)
- **Data review:** the Cowen automation ran on 3 and 4 Nov and applied four flag changes with no failures. Every fed indicator except the cash rate matched its official source, using a dry run of the workflow's own parsers. BTC prices were within 0.07% of CoinGecko.
- **Cash rate:** the RBA moved to 4.85% on 4 Nov, and the Monday-only feed wouldn't have seen it until 9 Nov. Added by hand (`0017`), with the RBA as its source.
- **Score now counts real rate changes only:** repeated readings made the trend read "stable" a week after a rise. The engine is unchanged; `prepareSeries` gives it the changes only. Score: 43.02 before, 41.04 with 4.85% and a rising trend, holding until about 30 Dec, when the 30 Sep decision leaves the 3-month window.
- **Indicators feed runs daily:** new figures arrive within a day (needs a re-import).
- **`/locations` chart:** now plots SQM asking prices only. It had joined them to older sold medians, which drew moves that never happened (Sydney "+39%").

### Close-out (2 Nov, see [[Handoff 2026-11-02 (close-out)]])
- **No more approvals:** BTC flag changes apply as soon as the workflow saves them (migration `0014`). `btc_flag_suggestions` stays as the log. When 20 flags are active, the stalest retires to make room.
- **Website can't change anything:** the public key and signed-up users lost every write permission (migration `0015`). Tested by attempting writes with the public key; all were refused. The `/admin` and approval-screen plans are dropped.
- **BTC flags 36 → 19:** applied the 13 waiting suggestions and retired 12 duplicates or stale flags (`0012`). The raw total went from −6.7 to −3.4, which still caps at −2, so scores didn't move.
- **Price bands 16 → 14 (`0013`):** a price below every band now gets the deepest band's score instead of reading SELL; the $38K and $39K rows merged; $57K and $42K put in order; three labels refreshed.
- **Site:** `/btc` shows Cowen summaries and recent flag changes; `/audit` has an automation health panel; Locations calls its price an asking price and shows older readings' dates.
- **Legacy rows stay:** the plan to delete odd `property_data` rows was dropped, as it contradicted the rule against deleting anomalies.
- **Repo:** Streamlit and its files moved to `legacy/`, which stops the old deployed app; missing migration records added; docs brought up to date.

### Site UI rework (PR #26)
- Mobile nav no longer scrolls the page sideways; active page highlighted.
- Charts use real time axes, straight or stepped lines, shaded target zones and correctly labelled warning lines.
- Dashboard: gauge marker, zoned score history, coloured pillars, tile sparklines, commentary columns. Indicators grouped by pillar.
- BTC page: price against key levels, price ladder, flags grouped by category. Locations, Audit and Cycle tidied; contradictory cycle advice fixed.
## August 2026 - Data quality sweep + Location Analysis

### `/locations` — Location Analysis rebuilt
- Deferred during the rebuild because `property_data` was stale; revived once an Aug 2026 snapshot landed for all locations
- Regional divergence summary, all-locations ranked table, head-to-head comparison (two selectors, metric table, dual median-price chart, recommendation)
- `locationScore()` ported from `calculate_location_score()`; `sales_volume` declares no winner (readings mix geographic scope and period)

### Outlier audit — 31 findings, zero deletions
- Adversarial review **rejected every proposed deletion**. Several accusations were wrong: Brisbane's −30.93% growth was called impossible by comparing two non-overlapping windows; Sydney's 19.83% vacancy sits inside the project's own plausibility band and comes from a sub-market sample (sales_volume 13, DOM 106)
- Standing rule recorded in `CLAUDE.md`: **surface anomalies, never delete them**
- `lib/property-audit.ts` + `/audit` section: readings to verify, duplicate snapshots, metrics missing from the newest snapshot
- Only data touched all month: the four n8n scraper zeros deleted in July

### Scoring honesty (no score changes)
- `getWindowContext` — the two-point trend rule called auction clearance "stable" while it fell 17.2% over 6 readings (5 of 5 consecutive declines). The contradiction is now shown on `/indicators`; scoring and the parity suite untouched
- **Weight at risk** on `/audit`: ~43% of the score rides on indicators that cannot move; `household_debt_gdp` alone is 18.5% off one 404-day-old reading
- **Cadence flags**: quarterly indicators can *never* satisfy a 3-month trend or 6-month volatility window — proven by test, so `/audit` no longer implies a backfill would help
- Zero-valued readings now count as real values in `locationScore`, matching `calculateRegionalDivergence` (the Python's `if x and x > n` was a None-check, not a decision about zero)

### Data-layer fixes
- Building-approvals annualisation no longer emits partial-window points (a widening window alone drew a fake uptrend)
- `getSiteData` now records query errors: a denied read (RLS) and an empty table looked identical, so `/locations` could report "no data" when the query had actually failed
- Migration `0006`: plausibility guard — hard CHECK on manual entry, skip-trigger on the n8n feed
- Household debt series identified as **ABS 5232.0** (113.x family, not IMF 108–112 or CEIC 119–120); no figures inserted, `supabase/backfill-indicator.sql` records which family to use

### `CLAUDE.md` added
Project context for future sessions: deployment flow, the never-delete-data rule, parity discipline, decisions already made, open issues.

## June/July 2026 - Rebuild Phase 1 (Next.js)

### Rebuild kickoff — engine port + public site
Decisions locked: cycle anchor **2008** everywhere, **Next.js** (live site), property_data kept but Location Analysis deferred. See [[Rebuild Proposal]].

- **`supabase/migrations/`** created (apply in order; manual pg_dump first):
  - `0001` history dedupe + `UNIQUE(date, indicator_name)` + `is_known_indicator()` CHECK on both tables
  - `0002` combined view recreated as SECURITY INVOKER with deterministic tiebreaker (owned by migrations, not app code)
  - `0003` `market_score_history` table (daily score snapshots)
  - `0004` RLS everywhere — anon read-only on 4 public tables, deny-all elsewhere (pre-flight: verify n8n credential first)
- **`web/`** — Next.js 15 App Router + Tailwind 4 + Recharts:
  - `src/lib/scoring/` — pure-TS port of v2/v3 engine, Anderson tracker, auto-commentary, UI helpers. Bug-for-bug (unreachable danger/crisis elif branches preserved; v3 seeds from rounded base score) so numbers match the old app exactly
  - **24/24 golden-master parity tests** — fixtures generated by running the real Python functions against synthetic SQLite scenarios (baseline, perfect storm, crisis, supply squeeze, oversupply, sparse-like-production, regions, empty)
  - Pages: `/` (score+gauge+pillars+tiles+commentary+sparkline), `/indicators` (config-driven cards, honest single-point handling), `/cycle` (2008 anchor → Year 18/Crash in 2026), `/audit` (freshness flags >90d)
  - `/api/snapshot` + Vercel Cron (daily 21:00 UTC) writes score history — the signal-over-time series the old app never had
  - One data query per page render instead of ~40; ISR revalidate hourly
- **`.github/workflows/web-ci.yml`** — vitest + typecheck + build on every web/ change
- Anderson consequence of the 2008 anchor: tracker now reads **Year 18 — Crash/Correction** (was Year 15/Winner's Curse under 2011)
- Not yet: admin data entry (Supabase Studio in the interim), migrations not applied, Vercel not deployed

### Mortgage stress graphing fix (22df685)
- Registered psycopg2 DEC2FLOAT caster: the combined view's NUMERIC column returned Decimals that Plotly can't serialise — history-fed indicators (mortgage stress especially) now chart correctly in the legacy app
- `pd.to_numeric` + re-sort guard on the trend chart

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
