# Handoff — 2 Nov 2026 (close-out)

**Read this first.** It's the current state. [[Data Pipelines Handoff 2026-11-02]] is still the reference for the system map, data definitions and n8n lessons.

## Where things stand

The system runs by itself. n8n collects the data, Supabase stores it, and the public site only reads it.

- **BTC flags change automatically.** A row saved to `btc_flag_suggestions` applies straight away, and that table is the log of what changed and why.
- **The website can't change anything.** The public key has no write permission on any table, and it can't call `get_btc_signal`. Writing with the public key was tested and refused.
- **19 active flags,** raw total −3.4, which counts as −2 in the score.
- **14 price bands,** in order: a deeper price never scores lower.

Everything above was applied to production on 2 Nov (migrations `0011`–`0015`). The old flag and band rows are in `btc_context_flags_backup_20261102`, `btc_flag_suggestions_backup_20261102` and `btc_key_levels_backup_20261102`.

## What replaced the approval step

These are the limits on a change nobody reviews:

- A single flag can move the score at most ±0.5, and all flags together at most ±2.
- Each video can make at most 8 changes, and each run covers at most 3 videos.
- Only the official channel ID is read, and the transcript is passed to Claude as data, not instructions.
- Every change is logged with a reason and shown on `/btc`.

**Worst case:** a badly misread run could still swing the score by up to 4 points (from −2 to +2 from flags), until a later video corrects it. One video alone can't go that far, because it's limited to 8 changes. The price band sets the rest, and no workflow can change the bands.

## What's left for Lincoln

See [[Cutover Checklist]] for the steps.

1. Rotate the database password, and delete the Streamlit app.
2. Optional: paste `supabase/migrations/0016_drop_empty_legacy_tables.sql` into the SQL Editor. Claude's attempt was stopped by a safety check on bulk deletes.
3. Copy the updated launcher `index.html` to the NAS (see the launcher's `CLAUDE.md`).
4. Optional: re-import `n8n/workflows/Cowen Videos - BTC Flag Suggestions.json`. It tells Claude there's no human check, so it should make fewer marginal changes, and saves retirements first. The live workflow works without it.

## Why the market score moved on 2 Nov

The score went from 44.46 to 43.02 because of that day's data cleanup, not the market. Expect that step in the score history. Re-running the engine on the cleaned data gives 43.02 for every date tested, from 25 Oct onward.

## Decisions made

- **The BTC flags stay (3 Nov).** Replacing them with one number per video was considered and declined. It's known and accepted that the ±2 cap usually absorbs a single flag change.
- **No approval process, and nothing writable on the website.** Lincoln's call. The approval screen and `/admin` page are off the table.
- **Legacy `property_data` rows stay,** flagged on `/audit`, under the rule against deleting anomalies.
- **Price bands don't come from the workflow.** The flags are capped; the bands aren't. A misread video could move the score far more through a band, and they rarely change, so they stay as deliberate edits.
- **The 2 Nov backup tables stay for now.** There are five of them, they're locked, and the free tier has no other backups.

## Open questions

- **Fix the monthly trend window?** Monthly readings are dated the first of the month, so from the 2nd until the next release, the reading from three months ago falls outside the engine's 3-month window. That's why `/audit` jumped from 22% to 70% "can't move" on 2 Nov. The score didn't change that day (43.02 either way), but a large move could be missed during those weeks. Month-aligned windows would fix it. It's a small change, but it's a scoring change, so it needs a recorded before/after.
- **Automate or drop the manual indicators?** `household_debt_gdp` (weight 25) and `mortgage_stress_rate` (weight 15) have no feed, which is 40 of the 135 points. `household_debt_gdp` was last updated for December 2025. These are the main thing left that isn't automatic. Dropping them is a scoring change, so it needs a recorded before/after score.

## Related notes

- [[Cutover Checklist]]
- [[Property Analyser - Architecture]]
- [[Changelog]]
