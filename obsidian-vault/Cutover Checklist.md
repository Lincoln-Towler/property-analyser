# Cutover Checklist — Lincoln's Steps

Plain-English steps to take the new Next.js site live. ~45 minutes of active work. Full technical detail lives in `web/README.md`; the background is in [[Rebuild Proposal]].

## 1. Check what credential n8n uses ⚠️ do first (5 min)
- [ ] Open the n8n workflow that writes indicator data
- [ ] Open the DB node and look at its credential:
  - **Connection string / host+password** (`db.uorrpukzqwyvzbewgudk.supabase.co`, user `postgres`) → it uses the database password. **Do not rotate the password until this is changed** — flag it before cutover.
  - **Supabase API credential with a `service_role` key** → safe, nothing we do affects it.

## 2. Backup (10 min)
Supabase free tier has NO automatic backups.
- [ ] Supabase Dashboard → Table Editor → Export as CSV for: `economic_indicators`, `economic_indicators_history`, `property_data`, `btc_key_levels`, `btc_context_flags`, `btc_signal_log`
- [ ] Save the files somewhere safe

## 3. Apply migrations (10 min)
- [ ] Merge the rebuild branch on GitHub (usual PR flow)
- [ ] Supabase Dashboard → SQL Editor → paste & Run, in order:
  - [ ] `supabase/migrations/0001_history_dedupe_and_constraints.sql`
  - [ ] `supabase/migrations/0002_combined_view_security_invoker.sql`
  - [ ] `supabase/migrations/0003_market_score_history.sql`
  - [ ] `0004_enable_rls.sql` — **only after step 1**; if n8n uses the anon key, stop and flag it
- [ ] Optional sanity check: Supabase Dashboard → Advisors → Security — the RLS errors should be gone after 0004

## 4. Switch n8n to upsert (5 min)
- [ ] Change the insert node's SQL to:
```sql
INSERT INTO economic_indicators_history (date, indicator_name, value)
VALUES (...)
ON CONFLICT (date, indicator_name) DO UPDATE SET value = EXCLUDED.value;
```
- [ ] If the node uses the point-and-click Insert operation, switch it to "Execute Query"

## 5. Deploy to Vercel (15 min)
- [ ] vercel.com → log in with GitHub → Add New → Project → import `Lincoln-Towler/property-analyser`
- [ ] Set **Root Directory** = `web`
- [ ] Add 4 environment variables:

| Name | Value |
|---|---|
| `NEXT_PUBLIC_SUPABASE_URL` | `https://uorrpukzqwyvzbewgudk.supabase.co` |
| `NEXT_PUBLIC_SUPABASE_ANON_KEY` | Project Settings → API Keys → `anon` / publishable |
| `SUPABASE_SERVICE_ROLE_KEY` | same page → `service_role` secret |
| `CRON_SECRET` | any long random string |

- [ ] Deploy, then check `/`, `/indicators`, `/cycle`, `/audit` load with real data

## 6. Parallel run (1–2 weeks, passive)
- [ ] Keep Streamlit running; compare its score against the new site every few days — they should match exactly
- [ ] The daily cron (21:00 UTC) starts building the score-history chart automatically

## 7. Cutover (with Claude)
- [ ] Rotate the database password (ONLY once n8n's credential from step 1 is known/updated)
- [ ] Delete the Streamlit Cloud deployment
- [ ] Optionally drop the empty legacy tables

## In the meantime
- **Monthly data entry** (until the `/admin` page is built): Supabase Dashboard → Table Editor → `economic_indicators` → Insert row. Use exact snake_case names (`mortgage_stress_rate`) — typos are now rejected by the database.
- **Force a score snapshot / fresh data now**: redeploy on Vercel, or wait for the hourly refresh.

## Related Notes
- [[Rebuild Proposal]]
- [[Changelog]]
