# Property Analyser - Architecture

A personal Australian property-market analyser, plus a separate Bitcoin cycle signal that shares the database. For how to work on it, see `CLAUDE.md` in the repo root.

## How it fits together

```
Public sources ──▶ n8n on TrueNAS ──writes──▶ Supabase ──reads──▶ Next.js site on Vercel
                   (service role)             (Postgres)          (public, read-only)
```

Nothing on the website can change the data. The only writers are the n8n workflows and the site's daily snapshot job, both using the service role.

## Workflows

| Workflow | When (Adelaide) | Writes to |
|---|---|---|
| Economic Indicators | Mondays 7:00am | `economic_indicators_history` |
| Location Data (SQM) | Mondays 7:15am | `property_data` |
| Cowen Videos → BTC flags | Daily 8:30am | `btc_video_analysis`, `btc_flag_suggestions` (which applies to `btc_context_flags`) |
| BTC brief (separate, not in this repo) | Daily | calls `get_btc_signal`, which logs to `btc_signal_log` |
| `/api/snapshot` (Vercel Cron, not n8n) | Daily, 21:00 UTC | `market_score_history` |

## Site pages

| Page | Shows |
|---|---|
| `/` | Market score, four pillars, key indicators, commentary, score history |
| `/indicators` | Each indicator against its target, with a history chart |
| `/locations` | Locations ranked, and a head-to-head comparison |
| `/cycle` | The 18.6-year Anderson cycle, anchored at 2008 |
| `/btc` | BTC signal, price ladder, flags, Cowen summaries, recent flag changes |
| `/audit` | Automation health, data freshness, and readings worth a second look |

## Tables

| Table | Holds |
|---|---|
| `economic_indicators` | Manual and backfilled readings, with a `source` |
| `economic_indicators_history` | Readings from the n8n feed |
| `economic_indicators_combined` | A view merging both; a manual row wins on the same date |
| `property_data` | `(date, location, metric_name, value, source)` for 11 locations |
| `market_score_history` | One market score per day |
| `btc_key_levels` | 14 price bands, each with a base score |
| `btc_context_flags` | The flags that nudge the BTC score |
| `btc_flag_suggestions` | The log of flag changes; saving a row applies it |
| `btc_video_analysis` | One paraphrased summary per Cowen video |
| `btc_signal_log` | One BTC reading per day |

Migrations in `supabase/migrations/` own the schema. Anyone with the public key can read the market tables and nothing else, and can't write to any table.

## Scoring

- **Market score:** `web/src/lib/scoring/`, a TypeScript port of the Python in `legacy/`, tested against outputs from the original. See [[Score Calculation Pipeline]].
- **BTC score:** `get_btc_signal(price)` in Supabase. The price band's base score, plus the active flags capped at ±2, kept between 1 and 10.

## Related notes

- [[Handoff 2026-11-02 (close-out)]]
- [[Score Calculation Pipeline]]
- [[Economic Indicators Integration]]
- [[Changelog]]
