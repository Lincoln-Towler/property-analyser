# Cutover Checklist

The move from Streamlit to the Next.js site. Only the last steps are left. Background is in [[Rebuild Proposal]].

## Done

- [x] Checked what credential n8n uses: a Supabase API credential (service role), not the database password
- [x] Backed up the data
- [x] Applied the migrations, up to `0015`
- [x] Switched n8n to upsert
- [x] Deployed to Vercel, with the daily score snapshot running since 14 July
- [x] Parallel run, which is over. The Streamlit code moved to `legacy/` on 2 Nov, so the old app should no longer start

## Left to do (about 5 minutes)

Claude can't do these, because they need your logins and a new password.

### 1. Rotate the database password

The old password is stored in Streamlit Cloud's secrets, so it should be treated as exposed. Nothing that's running uses it: n8n and Vercel use API keys, which this doesn't change.

- [ ] Open [Supabase](https://supabase.com/dashboard) → project `property-analyzer` → **Project Settings** → **Database**
- [ ] Click **Reset database password**, then **Generate a password**, and save it in your password manager
- [ ] The next morning, check that `/audit` still says everything ran on schedule

### 2. Delete the Streamlit app

- [ ] Open [share.streamlit.io](https://share.streamlit.io) and sign in
- [ ] Find `property-analyser`, click **⋮**, then **Delete**. This also removes the old secrets

### 3. Optional: drop the seven empty tables

These tables have never held a row, so they're harmless clutter. Do this after step 2, because the old app recreates them if it's opened.

- [ ] In Supabase, open **SQL Editor**, paste `supabase/migrations/0016_drop_empty_legacy_tables.sql` and click **Run**. It refuses to run if any of them has data

## Entering data by hand

The site has no admin page, by decision. For the few values that are still manual (household debt, mortgage stress, days on market, sales volume), use Supabase → **Table Editor** → insert a row. Use the exact snake_case names (such as `mortgage_stress_rate`); typos are rejected by the database.

## Related notes

- [[Handoff 2026-11-02 (close-out)]]
- [[Rebuild Proposal]]
- [[Changelog]]
