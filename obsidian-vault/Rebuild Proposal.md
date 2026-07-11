# Rebuild Proposal — Streamlit → Own Website

*June 2026. Full review of the codebase + live Supabase database, three candidate architectures designed independently and scored by a three-judge panel (maintainability / pragmatism / product lenses).*

## What the review found

### The app vs the data (live DB, verified June 2026)

| Table | Rows | State | Verdict |
|---|---|---|---|
| `economic_indicators` | 12 | manual entries, ~1/month | **KEEP** |
| `economic_indicators_history` | 114 | n8n feed, actively growing | **KEEP** (needs dedupe) |
| `property_data` | 108 | stale since 2026-04-18 | DEFER (archive until revived) |
| `market_commentary` | 0 | never used (auto-commentary needs no table) | DROP |
| `market_sentiment` | 0 | dead dropdown option, no form, no reader | DROP |
| `infrastructure_projects`, `migration_data`, `employment_data`, `employer_events`, `suburb_scores` | 0 each | ~1,060 lines of UI, never held a row | DROP UI, keep formulas documented in [[Leading Indicators Modules]] |
| `btc_key_levels` / `btc_context_flags` / `btc_signal_log` | 15 / 30 / 100 | separate live BTC system sharing the DB | isolate, don't touch |

**Headline insight: roughly 1,000+ lines of the 5,258-line app serve features whose tables have never held a row.** The actually-used product is: the market score engine (v2/v3), the Anderson cycle tracker, the economic-indicator charts, auto-commentary, and a once-a-month manual entry path.

### Structural pain points (why Streamlit is fighting you)

1. **Connection-per-render, no pooling** — 33 call sites open fresh psycopg2 connections to Sydney per rerun.
2. **Full-script rerun model** — the sidebar recomputes a ~40-query market score on every widget interaction; `init_database()` re-runs `CREATE TABLE IF NOT EXISTS` for 9 tables on every page load.
3. **DDL at read time** — the app DROPs/CREATEs `economic_indicators_combined` while rendering, which is why it needs the owner-level connection string and why the view is SECURITY DEFINER.
4. **Zero caching, zero tests, no migrations** — schema ownership is split between app code, n8n, and hand-run SQL; the duplicate rows are the direct result.
5. **Broken SQLite fallback** — 35 hard-coded `%s` placeholders mean the "fallback" crashes on most pages anyway.
6. **Known UI bugs** — the commentary Save button is unreachable (nested `st.button`), CSV import can silently roll back while reporting success.
7. **Score is ephemeral** — recomputed every render, never persisted; there is no "how has the signal moved over 6 months" view.

### Security findings (⚠️ act on these regardless of the rebuild)

- **RLS is disabled on all 13 tables.** Anyone holding the project's anon key can read *and write* every row via the Supabase REST API. ([Supabase RLS docs](https://supabase.com/docs/guides/database/postgres/row-level-security))
- **The raw owner-level Postgres connection string lives in Streamlit Cloud secrets** — it must be treated as burned and rotated at cutover.
- `economic_indicators_combined` is a **SECURITY DEFINER** view (created by app code at runtime).
- **Supabase free tier has no automated backups** — take a manual `pg_dump` before any migration work.
- **Do not enable RLS blindly**: first inventory what credential n8n (and whatever reads/writes the BTC tables) actually uses. If n8n uses the anon key or the same connection string, a naive RLS flip or password rotation silently kills the only live data feed.

### Data-quality findings

- `economic_indicators_history` has **no unique constraint** → n8n re-runs appended pure duplicates (e.g. 18 identical `wage_growth` rows dated 2025-09-01). Verified: all duplicate groups have exactly one distinct value, so dedupe is safe.
- **Four indicators have exactly one data point ever** (`mortgage_stress_rate`, `household_debt_gdp`, `mortgage_arrears_rate`, `dwelling_supply_deficit`) — lines can't render, trends read "stable", volatility reads "insufficient data", yet their weights (15+25) silently anchor the score forever. The engine needs a staleness guard.
- **The 18.6-year cycle uses two different anchors in the same file** — scoring uses 2008 (`Year 18, ×0.90 peak multiplier`), the Anderson page and commentary use 2011 (`Year 15, Winner's Curse`). Unifying to 2011 changes the headline score — a deliberate, signed-off decision, not a silent fix.
- Two unreachable `elif` branches (`danger_above` shadowed by `warning_above`; `crisis_below` shadowed by `deficit_below`) mean danger scores can never fire. Fixing them changes scores — same discipline: parity first, then fix as reviewed commits.

## The recommendation

**A static site (Astro + TypeScript) with build-time scoring, deployed free on Cloudflare/GitHub Pages, rebuilt nightly by GitHub Actions after the n8n run — no servers, no auth stack, no custom CRUD, no Supabase client in the browser.**

Two of three judges ranked this first (8.5/10 on both maintainability and pragmatism). It is the only design sized to the actual data: ~130 rows, one daily feed, one manual entry per month.

### Architecture

```
n8n (unchanged, switched to upsert)
        │
        ▼
Supabase Postgres  ◄── Supabase Studio = the admin UI (replaces ~1,100 lines of CRUD)
  - migrations own the schema (incl. the combined view, SECURITY INVOKER)
  - UNIQUE(date, indicator_name) on history
  - CHECK constraint on indicator_name (no phantom series from typos)
  - new score_history table (one row per build: score, sub-scores, breakdown, audit jsonb)
  - RLS: deny-all (no anon policies — nothing browser-side ever connects)
        │
        ▼  (nightly GitHub Action, ~07:00 AEST + manual "Run workflow" button)
astro build
  - ONE query pulls the combined view (~130 rows)
  - scoring.ts (pure functions, unit-tested): v2/v3 ladders, trend bonus,
    composite overrides, volatility penalty, sub-scores, confidence,
    Anderson cycle, auto-commentary — all from one INDICATORS_CONFIG
  - writes today's score_history row
  - emits /report.md (the "export for Claude" report, every build)
        │
        ▼
Static HTML + tiny vendored uPlot charts → Cloudflare Pages (free)
Pages: / (score, gauge, sub-scores, tiles, commentary, score-history sparkline)
       /indicators (config-driven cards; 1-point indicators shown honestly as value cards)
       /cycle (Anderson tracker)
       /audit (per-indicator freshness, point counts, n8n health — the ops window)
```

### Why this over the alternatives

| | **Astro static (recommended)** | Next.js + Vercel SSR | Python cron + vanilla JS |
|---|---|---|---|
| Judge scores | **8.5 / 8.5 / 7.5** | 6 / 6 / 8.5 | 7 / 7 / 6.5 |
| Moving parts | build only; nothing runs between builds | RSC, Server Actions, middleware auth, ISR, 2-key discipline | 2 languages forever, hand-rolled admin page |
| Security | **no key in any client; RLS deny-all** | anon key public (read-only via RLS) | write-capable session in the browser |
| Data freshness | nightly + manual button (data changes ~daily anyway) | live | nightly + manual button |
| Effort | **15–25 h** | 25–35 h | 25–35 h |
| Failure mode | broken build → last good site keeps serving | framework churn, free-tier ToS | rot-prone bespoke JS CRUD |

If you decide you want a *live* site with an in-browser admin form (instant score refresh after edits), Next.js + Supabase + Vercel is the fallback — it scored highest on pure product experience. The Python-cron option is dominated: its one advantage (no scoring port) is neutralised by golden-master parity tests, and it spends the savings on the worst frontend.

### Grafted improvements the judges demanded (all adopted)

- **Golden-master parity**: before writing any TypeScript, run the *existing Python* scoring functions against live data and freeze the outputs as fixtures. The TS port must reproduce them exactly; the 2011-anchor and elif fixes then land as separate reviewed commits with before/after score deltas.
- **Staleness guard in the engine**: any indicator whose latest point is >90 days old is flagged in the breakdown and downgrades confidence — no more one-point-forever anchoring.
- **`npm run add-indicator` CLI** ships in v1: validated manual entry (name checked against config) without any hosted CRUD.
- **Deterministic view tiebreaker** (`ORDER BY priority, id DESC`) so any duplicate that ever sneaks past the constraint dedupes predictably.
- **Loud pipeline death**: the nightly workflow fails if the newest indicator date is >45 days old (catches silent n8n death) and notifies on failure; note GitHub auto-disables cron on 60 days of repo inactivity.
- **"Data as of <build time>" stamp** on every page footer.
- **BTC isolation**: v1 = RLS with no policies (zero risk to the live signal system); moving it to its own schema/project is an optional later step — *after* verifying what reads it today.

## Migration plan (each phase leaves everything working)

- **Phase 0 — Inventory & backup (1 evening).** Manual `pg_dump`. Find what credential n8n uses (open the n8n Postgres/Supabase node and look) and what, if anything, reads the BTC tables. Nothing is rotated or locked until this is known.
- **Phase 1 — DB repair (1 evening).** Init `supabase/migrations/` (baseline via `supabase db pull`). Migrations: dedupe history (safe — duplicates verified identical) → `UNIQUE(date, indicator_name)` → CHECK constraint on `indicator_name` → recreate combined view `WITH (security_invoker = true)` + deterministic tiebreaker → create `score_history`. Switch n8n INSERT → upsert. Streamlit keeps working throughout; re-run Supabase advisors until clean.
- **Phase 2 — RLS lockdown (short).** Enable RLS on all tables, no anon policies (adjust only if Phase 0 found an anon-key consumer). Streamlit (owner connection) and n8n (service credential) bypass RLS, so nothing breaks.
- **Phase 3 — Scoring port (1–2 evenings).** Generate golden-master fixtures from the Python engine against live data. Port to `scoring.ts` + Vitest until parity is exact. Then the three deliberate fixes (2011 anchor, two elif bugs) as separate commits with recorded score deltas — sign off on the new headline number.
- **Phase 4 — Site + CI (1–2 weekends).** Astro pages, uPlot charts, `/audit`, `/report.md`, nightly GitHub Action, deploy to Cloudflare Pages.
- **Phase 5 — Parallel run & cutover (2 weeks passive).** Compare the daily static score against the still-running Streamlit app. Then: rotate the DB password, decommission Streamlit Cloud, archive `property_analysis_dashboard.py` in the repo, rename `property_data` → archive (or recommit to feeding it).

**Total: ~15–25 hours across 2–3 weekends. Running cost: $0/month.**

## Decisions — RESOLVED June 2026

1. **Cycle anchor: 2008** (Lincoln's call). Score multiplier unchanged (×0.90); the Anderson tracker moves from Year 15/Winner's Curse to **Year 18/Crash phase**.
2. **Unreachable danger/crisis branches**: kept bug-for-bug in the v1 port to preserve exact score parity. Fixing them is a flagged follow-up (changes household_debt at 125 from 35 → 15 points).
3. **`property_data`**: kept as-is, Location Analysis deferred until the feed is revived.
4. **Freshness: live site → Next.js chosen** (the product-lens judge's pick). ISR-cached RSC pages, hourly revalidate, daily score snapshot via Vercel Cron.
5. **BTC system**: left in place behind deny-all RLS (v1); schema/project split deferred. (Free tier allows only 2 projects — Lardr occupies the second slot.)

**Build status**: see [[Changelog]] — scoring engine ported with 24/24 golden-master parity tests, 4 public pages built, migrations written. Remaining: apply migrations, deploy to Vercel, admin section, parallel run, cutover.

## Related Notes

- [[Property Analyser - Architecture]]
- [[Score Calculation Pipeline]]
- [[Leading Indicators Modules]]
- [[Changelog]]
