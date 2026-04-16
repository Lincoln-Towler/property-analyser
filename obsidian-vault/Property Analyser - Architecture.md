# Property Analyser - Architecture

## Overview

Australian Property Investment Analysis Dashboard built with **Streamlit** (Python). Single-file application (`property_analysis_dashboard.py`, ~4,935 lines) deployed on **Streamlit Cloud** with **Supabase** (PostgreSQL) as the production database and **SQLite** as local fallback.

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Frontend | Streamlit |
| Charts | Plotly (go + express) |
| Production DB | Supabase (PostgreSQL) |
| Local DB | SQLite |
| Automation | n8n (feeds economic_indicators_history) |
| Deployment | Streamlit Cloud |

## Navigation Pages

1. **Dashboard** - Quick stats, key indicators at a glance, market score
2. **Ultimate Analysis** - Full v3 score with volatility, sub-scores, confidence intervals, data audit
3. **Economic Indicators** - Current values, crash risk indicators, historical trend charts
4. **Location Analysis** - Property data comparison between locations
5. **Anderson Cycle Tracker** - 18.6-year property cycle position
6. **Leading Indicators** - 5 modules (Infrastructure, Migration, Jobs, Supply/Demand, Suburb Scorer)
7. **Data Management** - Manual entry, CSV import, view/edit/delete, export

## Database Schema

### Core Tables (created in `init_database()`)

#### `economic_indicators`
| Column | Type | Notes |
|--------|------|-------|
| id | SERIAL / INTEGER PK | Auto-increment |
| date | DATE | NOT NULL |
| indicator_name | TEXT | NOT NULL |
| value | REAL | NOT NULL |
| source | TEXT | |
| created_at | TIMESTAMP | Default CURRENT_TIMESTAMP |
| | | UNIQUE(date, indicator_name) |

#### `economic_indicators_history` (Supabase only, created externally)
| Column | Type | Notes |
|--------|------|-------|
| id | INT8 | |
| date | DATE | |
| indicator_name | TEXT | |
| value | NUMERIC | |
| recorded_at | TIMESTAMPTZ | |

> No `source` or `created_at` columns. Fed by n8n automation.

#### `economic_indicators_combined` (VIEW)
Created dynamically by `economic_indicators_view()`. Combines both tables using `UNION ALL` with deduplication - current table takes priority over history when the same `(date, indicator_name)` exists in both.

Columns: `date`, `indicator_name`, `value`, `source`

#### `property_data`
| Column | Type | Notes |
|--------|------|-------|
| id | SERIAL / INTEGER PK | |
| date | DATE | NOT NULL |
| location | TEXT | NOT NULL |
| metric_name | TEXT | NOT NULL |
| value | REAL | NOT NULL |
| source | TEXT | |
| created_at | TIMESTAMP | |
| | | UNIQUE(date, location, metric_name) |

#### `market_sentiment`
| Column | Type |
|--------|------|
| id | SERIAL / INTEGER PK |
| date | DATE |
| source | TEXT |
| sentiment_score | REAL |
| notes | TEXT |
| created_at | TIMESTAMP |

#### `market_commentary`
| Column | Type |
|--------|------|
| id | INTEGER PK |
| commentary | TEXT |
| updated_date | DATE |

### Leading Indicator Tables

#### `infrastructure_projects`
Tracks transport, hospital, school, commercial projects with location, budget, dates, status, and impact radius.

#### `migration_data`
State-level interstate/overseas migration, international students, total population. UNIQUE(date, state).

#### `employment_data`
Regional employment stats: total employed, unemployment rate, job ads, growth rate. UNIQUE(date, region).

#### `employer_events`
Major employer events (expansion, closure, relocation) with jobs impact and industry.

#### `suburb_scores`
Calculated composite scores per suburb: infrastructure, population, employment, supply/demand, credit, gentrification, total score, rank. UNIQUE(date, suburb, state).

## Helper Functions

| Function | Purpose |
|----------|---------|
| `get_ph(conn)` | Returns `%s` (Postgres) or `?` (SQLite) placeholder |
| `db_upsert(cursor, conn, table, columns, values, conflict_cols)` | Insert with ON CONFLICT UPDATE |
| `csv_import_section(label, columns, table, ...)` | Reusable CSV import UI widget |
| `is_postgres(conn)` | Type check for connection |
| `economic_indicators_view(conn)` | Returns combined view name, creates view if needed |

## Related Notes

- [[Score Calculation Pipeline]]
- [[Economic Indicators Integration]]
- [[Leading Indicators Modules]]
- [[Changelog]]
