# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Lincoln's personal Australian property-market analyser, plus a separate Bitcoin cycle signal that shares the same database. Lincoln is a coding beginner: explain changes in plain language and keep PRs small.

- **Live site:** `web/` (Next.js on Vercel, https://property-analyser-three.vercel.app). The site is public.
- **Data:** the Supabase project `property-analyzer` (ref `uorrpukzqwyvzbewgudk`, ap-southeast-2).
- **Feeds:** n8n workflows on Lincoln's self-hosted n8n (TrueNAS, behind Cloudflare Access). Cloud sessions can't reach it, so workflow changes are delivered as JSON files for him to import.
- **Legacy:** `property_analysis_dashboard.py` (Streamlit). This is the reference implementation the web scoring engine is parity-tested against. Don't change scoring in it.
- **Notes:** `obsidian-vault/` holds Lincoln's notes: architecture, `Changelog.md`, dated handoff notes and the cutover checklist. Add a changelog entry for notable changes.
- **In-flight work:** read the newest `obsidian-vault/*Handoff*.md` before starting. It records decisions waiting on Lincoln that aren't visible in the code.

## Commands

The web app needs Node 22.

```bash
cd web
npm ci
npm run dev                       # needs .env.local (copy .env.example); without env vars pages show a setup notice
npm test                          # vitest: scoring parity + unit tests
npx vitest run tests/ui.test.ts   # one file
npx vitest run -t "ceiling"       # tests whose name matches
npm run typecheck
npm run build
```

CI (`.github/workflows/web-ci.yml`) runs test, typecheck and build on every change under `web/`.

The parity fixtures come from running the original Python. Regenerate them only when you change scoring on purpose: `python3 tests/generate_fixtures.py tests/fixtures/golden.json` (run from `web/`). Then commit the scoring change on its own, with the score delta recorded.

n8n workflows (Node 22 for the tests, Python 3 for the builders; neither needs dependencies):

```bash
cd n8n/src
node test.js && node test_locations.js && node test_btc.js
python3 build.py && python3 build_locations.py && python3 build_btc.py   # writes n8n/workflows/*.json
```

Edit `n8n/src/code/*.js` and `n8n/src/build_*.py`, then rebuild. Never hand-edit `n8n/workflows/*.json`. A rebuild gives every node a fresh random ID, and nothing else changes.

## Architecture

**Data flows** from n8n into Supabase, and the site reads it. Writes use the service role: the "Supabase account" credential in n8n, and `SUPABASE_SERVICE_ROLE_KEY` in Vercel, which is used by `/api/snapshot` and the server-side BTC reads. With RLS on, `anon` is read-only.

**Economic indicators**
- n8n upserts into `economic_indicators_history` (`on_conflict=date,indicator_name`).
- Manual or backfilled rows go into `economic_indicators`, with a `source` label.
- The view `economic_indicators_combined` (SECURITY INVOKER) merges the two. A manual row wins when both have the same date and indicator. The site and the snapshot cron read only this view.
- A `CHECK (is_known_indicator(...))` constraint allows only the 12 known names.
- Monthly and quarterly values are dated the first day of the period.

**Web data path**
- `web/src/lib/data.ts` makes one query per table per ISR window (`revalidate = 3600`).
- `lib/transforms.ts` then drops implausible values and annualises building approvals, since the feed delivers monthly *original*, not seasonally adjusted, readings.
- `/api/snapshot` runs on Vercel Cron at 21:00 UTC (`web/vercel.json`) and upserts the day's score into `market_score_history`.

**Scoring**
- `web/src/lib/scoring/` is a pure TypeScript port of the Python engine: `engine.ts` (v3 score, audit), `commentary.ts`, `anderson.ts` and `location.ts`.
- `config.ts` (`INDICATORS_CONFIG`) is the single source of thresholds for both the score and the UI.
- Known quirks of the Python are kept on purpose (see the PARITY NOTE in `engine.ts`). Output must match `tests/fixtures/golden.json` exactly.
- UI-only derivations live outside the engine: `scoring/ui-helpers.ts`, `lib/format.ts`, `lib/theme.ts`.

**Property data**
- `property_data` holds `(date, location, metric_name, value, source)`.
- `median_price` is SQM's 13-week average **asking** price for 3-bed houses, not a sold median.
- `days_on_market` and `sales_volume` are entered manually, by Lincoln's choice.

**BTC signal** (separate from the property score)
- `btc_key_levels` holds the price bands, each with a `base_score`.
- `btc_context_flags` holds the flags: `score_modifier` ±0.5 each, `is_active`.
- `get_btc_signal(price)` = base score of the highest band ≤ price, plus the sum of active modifiers **capped at ±2**, clamped to 1–10. It logs to `btc_signal_log`. Lincoln's separate BTC brief workflow calls it daily.
- **Cowen workflow:** writes paraphrased video summaries to `btc_video_analysis` and proposed flag changes to `btc_flag_suggestions`. Setting a suggestion's `status` to `approved` fires a trigger that applies it.
- **Limits:** a hard maximum of 20 active flags, and only one pending suggestion per flag (`supersede_older` trigger).
- **The `/btc` page** reads these tables server-side with the service role (`lib/btc.ts`). Pure view logic lives in `lib/btc-view.ts`.

## Rules

- **Production database changes:** back up first and get Lincoln's OK. Record each change as the next numbered file in `supabase/migrations/`. Some migrations were applied through the Supabase dashboard or MCP and committed afterwards as records. Their headers say "ALREADY APPLIED", so check `list_migrations` before running anything.
- **Never edit `btc_context_flags` directly.** Insert `pending` rows into `btc_flag_suggestions` (`video_id` may be NULL, with a `reason`) and let Lincoln approve them.
- **Keep these in sync** when one changes (only with Lincoln's agreement):
  - **Flag cap ±2:** `get_btc_signal`, `BTC_MODIFIER_CAP` in `web/src/lib/btc-view.ts`, and `n8n/src/code/btc_build_request.js`.
  - **20-flag limit:** `apply_btc_flag_suggestion` and `btc_build_request.js`.
  - **BTC signal thresholds:** `get_btc_signal` and `btcSignalForScore` in `btc-view.ts`.
  - **Market-score bands:** `signalForScore` in `engine.ts` and `SIGNAL_BANDS` in `lib/theme.ts` (pinned by `tests/ui.test.ts`).
- **The repo is public.** Credentials live in n8n and Vercel only. n8n credential blocks in the workflow JSON hold only an internal ID and name, which is fine to commit.
- **Cowen content:** store and display only paraphrased summaries and price levels. Never store or show transcript text.
- **n8n pitfalls:**
  - Supabase save nodes must leave Response Format on Autodetect, because JSON crashes on Supabase's empty 201 reply.
  - Use On Error → Continue, so one bad source doesn't stop the run.
  - Turn on Always Output Data for lookups that can be empty.
  - Upsert instead of using Supabase "Update row" nodes.
  - Batch rate-limited APIs one request at a time.
- Keep the site's "not financial advice" framing.
