# Data Pipelines Handoff — 2 Nov 2026

**For:** Claude Code working in `Lincoln-Towler/property-analyser`
**From:** a Claude session (2 Nov 2026) that fixed the n8n data feeds, cleaned the Supabase data, replaced Hotspotting with a free location-data feed, and added an AI review loop for the BTC score.

> **Working with Lincoln:** he's a coding beginner. Explain changes in plain language, keep PRs small, and run the web tests before pushing. **Ask before changing production data.** Back up first, and never change `btc_context_flags` directly: flag changes go through `btc_flag_suggestions` for his approval.

---

## TL;DR

- Three n8n workflows now feed Supabase reliably. Their source, tests and importable JSON are in `n8n/` (see `n8n/README.md`).
- Today's database changes are recorded as `supabase/migrations/0007`–`0010`. **They are already applied to production. Don't re-run them.**
- The BTC score was stuck at 1.0 at every price. Now the flags' combined effect is **capped at ±2**, there's a **max of 20 active flags**, and new flag changes arrive as **suggestions Lincoln approves**.
- **Top next tasks:** (1) commit this bundle, (2) make the `/btc` page show the ±2 cap, (3) show the Cowen video summaries on `/btc`, (4) fix the Locations page labels now that "median price" is an asking price. The full list is in [§5](#5-next-tasks-prioritised).

---

## 1. System map

```
 Public sources                n8n (Lincoln's TrueNAS)                  Supabase (property-analyzer)              Next.js site (web/, Vercel)
 ────────────────              ───────────────────────                  ────────────────────────────              ──────────────────────────
 RBA, ABS API, SQM,     ──▶  Economic Indicators (Mon 7:00)      ──▶  economic_indicators_history ─┐
 PropertyUpdate                                                       economic_indicators (manual) ─┴▶ economic_indicators_combined ──▶ /, /indicators, /audit
                                                                                                                                       /api/snapshot (cron 21:00 UTC) ──▶ market_score_history
 SQM postcode/city pages ──▶ Location Data (SQM) (Mon 7:15)      ──▶  property_data ─────────────────────────────────────────────────▶ /locations
 YouTube RSS + Supadata  ──▶ Cowen Videos → BTC Flag Suggestions ──▶  btc_video_analysis, btc_flag_suggestions ──(Lincoln approves)──▶ btc_context_flags
 + Claude API                 (daily 8:30)
 BTC price               ──▶ (existing BTC signal workflow, not   ──▶  rpc get_btc_signal(price) ──▶ btc_signal_log ────────────────────▶ /btc
                              changed today, ~19:30 UTC daily)        reads btc_key_levels + btc_context_flags
```

- **Supabase:** project `property-analyzer`, ref `uorrpukzqwyvzbewgudk`, region ap-southeast-2. RLS gives `anon` read-only access. Writes come from the n8n "Supabase account" credential (service role) and from Vercel's `SUPABASE_SERVICE_ROLE_KEY` (`/api/snapshot` only).
- **n8n:** self-hosted v2.46.3 on TrueNAS at `n8n.towlervideo.win`, behind Cloudflare Access. You can't reach its API from here, so n8n changes are delivered as JSON files that Lincoln imports.
- **Site:** https://property-analyser-three.vercel.app. It uses ISR and refreshes from Supabase hourly.

---

## 2. Supabase data model (the parts touched today)

### 2.1 Economic indicators

| Object | Notes |
|---|---|
| `economic_indicators` | Manual and backfilled rows **with a `source` label**. Wins over history in the combined view for the same `(date, indicator_name)`. |
| `economic_indicators_history` | n8n writes here via `POST …?on_conflict=date,indicator_name`. Has no `source` column; the view labels these rows `history`. |
| `economic_indicators_combined` | View (SECURITY INVOKER) used by the site and the snapshot cron. |
| `is_known_indicator()` | CHECK on both tables. 12 allowed names. |
| `*_backup_20261102` | Full copies of both tables taken before today's cleanup (RLS deny-all). Drop them once Lincoln confirms (see §5). |

**Series definitions (decided today; the n8n workflow writes exactly these):**

| indicator_name | Definition | Source and cadence |
|---|---|---|
| `interest_rate` | RBA cash rate target | RBA cash-rate table, snapshot every Monday (dated with the run date) |
| `unemployment_rate` | **Seasonally adjusted** headline rate | ABS Data API `LF` `M13.3.1599.20.AUS.M`, monthly |
| `building_approvals` | Total dwellings approved, **original (unadjusted)** | ABS `BA_GCCSA` `1.1.9.TOT.TOT.10.AUS.M`, monthly. The ABS API has no national seasonally adjusted series. `web/src/lib/transforms.ts` annualises using a trailing 12-month mean × 12, which absorbs the seasonality. |
| `credit_growth` | **Housing** credit, monthly % change (not total credit) | RBA Financial Aggregates page, monthly (month M is published around the end of M+1) |
| `wage_growth` | WPI annual %, excluding bonuses, private and public, seasonally adjusted | ABS `WPI` `3.THRPEB.7.TOT.20.AUS.Q`, quarterly |
| `population_growth` | Annual change in estimated resident population (persons) | ABS `ERP_COMP_Q` `10.AUS.Q` (latest quarter minus the same quarter a year earlier), quarterly |
| `rental_vacancy_rate` | SQM national vacancy % | SQM national page, monthly |
| `auction_clearance_rate` | National weekend clearance % | PropertyUpdate weekly report (fragile; see §3.1) |
| `mortgage_stress_rate`, `household_debt_gdp`, `mortgage_arrears_rate`, `dwelling_supply_deficit` | **Manual only**, with no feed | Roy Morgan (monthly), BIS/Trading Economics (quarterly), and two single values of unclear origin (weight 0) |

**Date convention:** monthly and quarterly values are dated the **first day of the period** (Sep 2026 → `2026-09-01`, Q2 → `2026-06-01`). Weekly auctions use the week-ending or published date.

### 2.2 Location data — `property_data`

`(date, location, metric_name, value, source)` with `UNIQUE(date, location, metric_name)`. There are 11 locations, named exactly as they appear in the table (see the appendix).

| metric_name | Now comes from | Meaning |
|---|---|---|
| `vacancy_rate` | SQM workflow, monthly | % |
| `rental_yield` | SQM workflow, weekly | Gross %: 13-week average **asking** rent × 52 ÷ 13-week average **asking** price, 3-bed houses (falls back to all houses) |
| `annual_growth` | SQM workflow, weekly | % change in the 13-week average asking price vs the same window a year earlier |
| `median_price` | SQM workflow, weekly | ⚠️ **13-week average ASKING price, 3-bed houses. Not a sold median.** |
| `days_on_market`, `sales_volume` | **Manual (Lincoln's decision)** | There's no free source. Latest entries are Aug and Feb 2026. |

SQM rows have a `source` starting with `SQM`. **Odd-looking legacy rows** from Feb 2026, which stay in the table: Sydney vacancy 19.83, Brisbane `annual_growth` −30.93 and vacancy 7.69, Port Adelaide vacancy 3.33, Perth vacancy 3.15, Albany/Tamworth vacancy 0. An August 2026 review already rejected deleting these, and zero readings count as real values (see the rules in `CLAUDE.md`). Newer SQM rows now supersede them, and `/audit` flags them.

### 2.3 BTC score

| Object | Notes |
|---|---|
| `btc_key_levels` | 16 price bands, each with a `base_score` (2.0 at $83K up to 9.8 at $31.75K) |
| `btc_context_flags` | Unique `flag_key`. `category` is CYCLE / MACRO / ON_CHAIN. `score_modifier` is ±0.5 max. `is_active`, `current_value`, `updated_at`. |
| `get_btc_signal(price)` | **Score = base of the highest band ≤ price (1.5 if none) + SUM(active modifiers) capped at ±2, then clamped to 1–10.** Thresholds: ≥9 MAX BUY, ≥7.5 STRONG BUY, ≥6 BUY, ≥5 LIGHT BUY, ≥4 HOLD, ≥2.5 CAUTION/REDUCE, else SELL/TAKE PROFIT. Logs to `btc_signal_log` (where `modifier_total` is the **capped** value from 2 Nov). Returns `score_breakdown.raw_modifier_total` and `modifier_cap`. |
| `btc_video_analysis` | One row per Cowen video: `video_id` PK, title, `published_at`, url, `transcript_chars`, `summary` (paraphrased), `stance` (bullish/bearish/neutral/mixed), `key_levels`, model, `status` (analysed/failed), error, `attempts` |
| `btc_flag_suggestions` | Proposed changes: `action` (add/update/retire), `flag_key`, label, category, `current_value`, `score_modifier`, `is_active`, reason, `status` (pending/approved/rejected/applied/failed/**superseded**), `apply_note` |
| Trigger `apply_on_approve` | Setting `status = 'approved'` applies the change to `btc_context_flags`, and the status becomes `applied` or `failed` with a note. **Hard limit of 20 active flags:** an add or re-activation past the limit fails with a note. |
| Trigger `supersede_older` | Keeps only **one pending suggestion per flag**, from the newest video. Older ones become `superseded`. |

**State at handoff:** 36 active flags (raw total −6.7, capped to −2). Today's score at $77.7K is 1.5 (SELL), versus 5.5 at $60K and 7.7 at $38K. New flags are blocked until fewer than 20 are active. Approvals are currently done in Supabase Studio (Table editor → `btc_flag_suggestions` → set `status`).

---

## 3. n8n workflows

Source of truth: `n8n/src/code/*.js` (the Code-node scripts) and `n8n/src/build_*.py` (node layout). Tests: `node test*.js`. See `n8n/README.md`.

### 3.1 Economic Indicators (Mondays 7:00am)

There are 8 independent branches, each **fetch → parse (range-checked) → upsert into `economic_indicators_history`**. A failing branch saves nothing and the others carry on.

- **Cash rate:** `https://www.rba.gov.au/statistics/cash-rate/`. Reads the *rate* column of the newest row. (The old code read the first `<td>`, which would have saved 0.00 on a hold.)
- **Housing credit:** tries `fin-agg-MMYY.html` for last month, then the month before, and uses whichever page exists.
- **Unemployment, approvals, wages, population:** `https://data.api.abs.gov.au/rest/data/ABS,{flow}/{key}?lastNObservations=N&dimensionAtObservation=AllDimensions&format=jsondata`. This is the path the ABS documents since its Nov 2024 API change. The old workflow used older forms (`api.data.abs.gov.au/data/…`), so don't go back to them.
- **Vacancy:** `https://sqmresearch.com.au/property/vacancy-rates?national=1&t=1`. Parses the embedded `var data = [...]`.
- **Clearance:** `https://propertyupdate.com.au/this-weekends-auction-results-what-happened-in-sydney-melbourne-brisbane-adelaide-canberra/`. The page can hold two weekly reports; the parser picks the one the other doesn't call "the previous week".

### 3.2 Location Data (SQM) (Mondays 7:15am)

Locations → fetch 33 SQM pages (11 locations × vacancy, asking rents, asking prices; **one every 1.5 seconds**) → calculate → upsert into `property_data?on_conflict=date,location,metric_name`. The first run on 2 Nov wrote 44 rows. Sanity ranges: vacancy 0–15%, price $100k–$5M, growth −40% to +60%, yield 1–12%. **It never touches `days_on_market` or `sales_volume`.**

### 3.3 Cowen Videos → BTC Flag Suggestions (daily 8:30am)

1. RSS `https://www.youtube.com/feeds/videos.xml?channel_id=UCRvqjQPSeaWn-uEx-w0XOIg`. This is the official @benjaminjcowen channel. **Several impersonator channels use his name, so always use this ID.**
2. Skips Shorts, and keeps videos aged 6 hours to 7 days. Skips videos already analysed. A failed video is retried up to 3 attempts. **Max 3 videos per run**, oldest first.
3. Supadata `GET https://api.supadata.ai/v1/transcript?url=…&lang=en&text=true&mode=native`, using the "Supadata" Header Auth credential (`x-api-key`). **Requests go one at a time, 3 seconds apart.** Sending them in parallel caused HTTP 429. The free plan allows 100 requests a month, and usage is roughly 30 (one video a day plus retries).
4. Claude `POST https://api.anthropic.com/v1/messages` using the "Anthropic" credential, model `claude-sonnet-5-5`, structured output via `output_config.format` (JSON schema), `max_tokens` 16000. The transcript is truncated at 150k characters. Cost is about 5–10c per video. The prompt is in `btc_build_request.js`. It tells Claude about the ±2 cap, the 20-flag limit, and the pending suggestions, to prefer retire/update over add, and to paraphrase only.
5. `btc_prepare_rows.js` clamps modifiers to ±0.5, converts keys to snake_case, turns an "add" of an existing flag into an update, drops unknown or already-retired flags, and keeps one suggestion per flag per run (from the newest video).
6. Upserts into `btc_video_analysis`, then inserts into `btc_flag_suggestions`. The `supersede_older` trigger handles duplicates across runs.

**First run (2 Nov):** analysed 3 videos and produced 19 suggestions. Lincoln approved 11 (retired `june_fed_boj_catalyst`, `day_count_capitulation_window`, `gold_summer_weakness`, `btc_summer_relief_rally_trap`, `bear_market_stage_3_confirmed`, and refreshed 6 others) and rejected 8 (7 duplicates plus a double-counting "three lows" flag). Three newer videos are waiting for the next run.

### 3.4 n8n lessons learned (read before editing any workflow)

1. **HTTP Request + Response Format "JSON" crashes on Supabase's empty 201** (`Cannot read properties of undefined (reading 'data')`). Leave saves on Autodetect.
2. **Any node error stops the whole workflow.** The credit branch usually hit an RBA 404, which stopped the later branches, so most feeds stopped updating between May and July. Use On Error → Continue and have parsers return `[]` on bad data.
3. An empty JSON array from an HTTP node means **zero items, so downstream nodes don't run**. Set *Always Output Data* on lookups that can be empty.
4. A node after a multi-item step runs **once per item** unless *Execute Once* is set.
5. **Don't use Supabase "Update row" nodes.** The old cash-rate node filtered on `value = 4.6`, overwrote the unemployment row every week and re-dated it. Upsert into history tables instead.
6. Supadata, and probably other APIs, rate-limit parallel requests. Use HTTP node batching (`batchSize: 1`).
7. YouTube blocks transcript requests from cloud IPs. That's why the workflow uses Supadata.

---

## 4. What changed today (for `Changelog.md`)

- **Diagnosis:** only the cash rate was still updating. The credit branch asked for an RBA page that usually isn't published yet, and the resulting 404 stopped the later branches. Several parsers and addresses were also outdated, and the cash-rate "update" node was hijacking the unemployment row.
- **Economic Indicators workflow fixed:** new ABS host, correct series (unemployment seasonally adjusted, housing credit), cash-rate parser, credit page fallback, isolated failures, range checks, broken "Update row" nodes removed.
- **Data backfilled and cleaned** (after a full backup): about 45 sourced rows added, mostly for 2026. Unemployment, approvals, credit, wages, population, cash-rate decisions and mortgage stress were aligned to official figures, and every series was moved to first-of-period dates, with one point per period.
- **Location data:** new SQM workflow replaces Hotspotting for vacancy, yield, growth and asking price on all 11 locations.
- **BTC:** new `btc_video_analysis` and `btc_flag_suggestions` tables with approve-to-apply, supersede and 20-flag-limit triggers. `get_btc_signal` now caps the flags' combined effect at ±2. First review cycle done.

---

## 5. Next tasks (prioritised)

1. **Commit this bundle (no production changes).** Add `n8n/`, `supabase/migrations/0007`–`0010` (marked already-applied) and this note, and add a `Changelog.md` entry from §4. Optionally link this note from `Property Analyser - Architecture.md` and update its schema section.

2. **`/btc` page: reflect the ±2 cap.** In `web/src/app/btc/page.tsx` (around lines 58–120), `modifierSum` is summed from the active flags, and the page says "Band base 3.5 −7.5 from 41 active flags" and "Score ceiling … SELL at every price". `scoreCeiling()` in `web/src/lib/btc-view.ts` (line ~22) also uses the uncapped sum.
   - Export `BTC_MODIFIER_CAP = 2` (mirroring `get_btc_signal`) and use the capped value for the score explanation and the ceiling. Prefer `latest.modifier_total` from `btc_signal_log` when it's present (capped from 2 Nov; older rows are uncapped).
   - Wording example: "Band base 3.5, flags net −6.7 (capped at −2.0) → 1.5". Keep the per-category "net" figures raw.
   - Add vitest coverage for `scoreCeiling` with the cap. **Done when** the ceiling message no longer claims SELL at every price and the tests pass in web-ci.

3. **`/btc` page: show the latest Cowen analysis.** Read `btc_video_analysis` (status `analysed`, newest 3–5): title (linked), date, stance, `summary` and `key_levels`. Optionally show the count of pending suggestions. **Never display transcript text.** Summaries are paraphrased on purpose, for copyright reasons.

4. **Locations page labels and freshness.**
   - `median_price` is now an SQM asking price. Relabel it, e.g. "Asking price (3-bed)", in `web/src/lib/scoring/location.ts` (line ~117, `median_price: 'Median Price'`) and in the chart heading "Median price trend" in `web/src/components/LocationCompare.tsx` (line ~208).
   - Metrics now have different dates (vacancy monthly, SQM weekly, days on market and sales volume manual), so the shared "latest readings as of" logic (page.tsx ~line 98) will fall back to per-row dates. Show per-metric dates, e.g. as a tooltip.
   - Tell Lincoln, without retuning silently: the location score thresholds (growth >10% → +15 and so on) now receive *asking-price* growth, which is smoother than sold-price growth. Stale manual metrics (days on market from Aug, sales volume from Feb) still count toward the score, so consider ignoring readings older than about 180 days.

5. **Get BTC active flags under 20.** Draft a retire list (stale items, and duplicates of the same thesis, e.g. several flags encoding "Q4 low after the midterms") and present it to Lincoln. On his OK, insert them as `pending` rows in `btc_flag_suggestions` (`video_id` can be NULL, with reason "manual cleanup") so he approves them the normal way.

6. ~~Clean the legacy `property_data` rows.~~ **Not doing this.** It contradicted the standing rule against deleting anomalies. They stay, flagged on `/audit`.

7. **Monitoring.** Extend `/audit` to flag feeds older than 2× their `expected_frequency` (already in `web/src/lib/scoring/config.ts`), SQM location freshness, and `btc_video_analysis` rows with `status = 'failed'`. Lincoln could also add an n8n Error Trigger workflow that emails on failure.

8. **Approval UI (discuss with Lincoln first).** Today he approves in Supabase Studio. An in-site UI must be authenticated, because the site is public and the anon key is read-only. Options: Supabase Auth plus an RLS UPDATE policy for his user on `btc_flag_suggestions.status`, or a server action using the service role behind auth. Never expose the service role key to the browser.

9. **Drop the backup tables** (`economic_indicators_backup_20261102`, `economic_indicators_history_backup_20261102`) in migration `0011` once Lincoln confirms.

10. **Backlog:** automate the remaining manual indicators (household debt to GDP from RBA/BIS quarterly; mortgage stress from Roy Morgan monthly). The leading-indicator tables (`infrastructure_projects`, `migration_data`, `employment_data`, `employer_events`, `suburb_scores`) exist but are empty (see `Leading Indicators Modules.md`).

---

## 6. Constraints and decisions to respect

- **Lincoln approves every BTC flag change.** Automation only suggests.
- **The ±2 cap and the 20-flag limit are deliberate.** Commentary points mustn't overwhelm the price-based score. Change the constants only with Lincoln's agreement, and in all three places: `get_btc_signal`, `apply_btc_flag_suggestion`, and `btc_build_request.js`.
- **Days on market and sales volume stay manual.**
- Prefer **free or self-hosted** sources. An open question: check SQM's terms of use before increasing the scraping load (it's about 35 page loads a week now).
- **Cowen content:** store and display only paraphrased summaries and levels. This is for personal use (YouTube's terms).
- Keep the site's "not financial advice" framing.
- **The repo is public.** Never commit keys, tokens or n8n credential exports.
- Production DB changes: back up first, get Lincoln's OK, and record each one as a numbered migration in `supabase/migrations/`.

---

## 7. Handy SQL

```sql
-- Freshness of every indicator the site uses
select indicator_name, max(date) as latest, count(*) as points
from economic_indicators_combined group by 1 order by 2;

-- BTC: active flag count and totals (raw vs what the score uses)
select count(*) as active, sum(score_modifier) as raw_total,
       greatest(-2, least(2, sum(score_modifier))) as capped_total
from btc_context_flags where is_active;

-- BTC: suggestions waiting for review
select s.id, v.title, s.action, s.flag_key, s.score_modifier, s.current_value, s.reason
from btc_flag_suggestions s left join btc_video_analysis v using (video_id)
where s.status = 'pending' order by s.created_at;

-- Approve or reject (the trigger applies approvals and sets 'applied' or 'failed')
update btc_flag_suggestions set status = 'approved' where id in (...);
update btc_flag_suggestions set status = 'rejected' where id in (...);

-- Latest SQM location figures
select location, metric_name, date, value from property_data
where source like 'SQM%' and date >= now() - interval '45 days' order by location, metric_name;
```

---

## Appendix: locations ↔ SQM

| `property_data.location` | SQM query |
|---|---|
| Sydney NSW | `region=nsw-Sydney&type=c` |
| Brisbane QLD | `region=qld-Brisbane&type=c` |
| Perth WA | `region=wa-Perth&type=c` |
| Adelaide SA | `region=sa-Adelaide&type=c` |
| Tamworth NSW | `postcode=2340` |
| Newcastle NSW | `postcode=2300` |
| Albany WA | `postcode=6330` |
| Geelong VIC | `postcode=3220` |
| Sale VIC | `postcode=3850` |
| Port-Adelaide SA | `postcode=5015` |
| Wodonga VIC | `postcode=3690` |

Page pattern: `https://sqmresearch.com.au/property/{vacancy-rates|weekly-rents|asking-property-prices}?{query}&t=1`. Each page embeds `var data = [...]`. To add a location, edit `n8n/src/code/locations_list.js`, rebuild, and re-import.
