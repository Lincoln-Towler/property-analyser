# n8n workflows

The data feeds that fill Supabase. They run on Lincoln's self-hosted n8n (TrueNAS),
not in this repo, so this folder is the source of truth and history for them.

| File in `workflows/` | Schedule (Adelaide) | Writes to |
|---|---|---|
| `Economic Indicators.json` | Mondays 7:00am | `economic_indicators_history` |
| `Location Data (SQM).json` | Mondays 7:15am | `property_data` |
| `Cowen Videos - BTC Flag Suggestions.json` | Daily 8:30am | `btc_video_analysis`, `btc_flag_suggestions` |

## Editing a workflow

Don't hand-edit the JSON in `workflows/`. It's generated.

1. Edit the Code-node scripts in `src/code/*.js` (or the node layout in `src/build_*.py`).
2. Run the tests (Node 22, no dependencies):
   ```bash
   cd n8n/src
   node test.js            # Economic Indicators
   node test_locations.js  # Location Data
   node test_btc.js        # Cowen videos -> BTC flag suggestions
   ```
3. Rebuild the importable JSON (Python 3, no dependencies):
   ```bash
   python3 build.py            # -> workflows/Economic Indicators.json (starts from original.json)
   python3 build_locations.py  # -> workflows/Location Data (SQM).json
   python3 build_btc.py        # -> workflows/Cowen Videos - BTC Flag Suggestions.json
   ```
4. Lincoln imports it in n8n: open the workflow, select all and delete, then
   **… → Import from File…**, re-select credentials if prompted, then **Save**.

## Credentials (in n8n only, never in this repo)

- **Supabase account**: Supabase API credential (service role). Used by every Supabase step and links automatically on import.
- **Supadata**: Header Auth, header name `x-api-key`. Used by *Get transcript (Supadata)*. Must be re-selected after import.
- **Anthropic**: Claude API key. Used by *Ask Claude*. Must be re-selected after import.

This repo is public, so never commit keys, tokens or exported credentials.

## Rules every workflow follows

- Every fetch and save step uses **On Error: Continue**, and parsers return nothing when the data looks wrong (range checks). One broken source skips that figure; it doesn't stop the run.
- Supabase saves are upserts (`?on_conflict=...`) with the HTTP node's **Response Format left on Autodetect**. Forcing JSON crashes on Supabase's empty 201 reply.
- Monthly and quarterly values are dated the first day of the period (Sep 2026 → `2026-09-01`, Q2 → `2026-06-01`).
