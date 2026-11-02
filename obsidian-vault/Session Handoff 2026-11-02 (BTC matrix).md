# Session Handoff — 2 Nov 2026 (BTC scoring matrix)

**For:** a local Claude Code instance on Lincoln's computer, picking up from a cloud session.
**Read first:** `CLAUDE.md` (rules and architecture), then [[Data Pipelines Handoff 2026-11-02]] (how the n8n feeds and the BTC approval loop work).

## Where things stand

- Branch `claude/database-cutover-checklist-y7zevw` has two commits on top of `main`:
  - `d74bddc` adds `n8n/`, migrations `0007`–`0010` (already applied; records only), the data-pipelines handoff, a changelog entry and `CLAUDE.md`.
  - `245938e` updates `/btc` for the ±2 flag cap. Each ladder band shows what it scores today, and the page warns when flags exceed the 20-flag limit. All 72 web tests pass.
- **Live BTC state (2 Nov):**
  - 36 active flags against the limit of 20, so new flags are blocked.
  - The flags total −6.7 raw; the score uses −2 after the cap.
  - At $77.7K (band base 3.5) the next run scores 1.5, SELL. The last logged reading (1 Nov, 1.0) predates the cap.
- **13 pending suggestions** in `btc_flag_suggestions` (ids 30–42):
  - 5 retirements: `atm_sweep_risk`, `sp_dotcom_distribution_risk`, `btc_200w_ma_being_tested`, `itc_bull_bear_tally_sep26`, `golden_cross_bull`.
  - 8 refreshes that keep their modifiers.

## Waiting on Lincoln's decision (do not apply without his OK)

### 1. Get the flags under 20

Approve the 13 pending suggestions, then retire these 12. Add each as a `pending` row in `btc_flag_suggestions` (`action = 'retire'`, `video_id` NULL, reason "manual cleanup: …") so Lincoln approves them himself:

| Flag | Reason |
|---|---|
| `mvrv_not_extreme_low` | Already counted inside `onchain_bottom_signals_absent` (MVRV Z < 0 is one of its four checks) |
| `btc_300w_ma_cluster_54k` | Same: the realized price (~$54K) is inside `onchain_bottom_signals_absent` |
| `btc_dom_excl_stables_rising` | Covered by `stablecoin_dom_not_peaked` |
| `time_based_cap_incomplete` | July status; superseded by `jun_oct_2026_timeline` and `cycle_day_timing_confirms_bear` |
| `midterm_avg_50k_target` | June figures; same "lower Q4 low" thesis |
| `qqq_etf_analog_bear` | An analog for the same downside targets as the on-chain and BTC/gold flags |
| `diminishing_returns_cycle` | About the 2026 top, which has passed |
| `btc_leads_sp_breakdown` | April; covered by `sp_april_low_not_confirmed` (equity correction underway) |
| `btc_relative_underperformance` | May year-to-date figures; overlaps `btc_gold_declining` |
| `recession_approaching_gdp` | Built on Q4 2025 GDP; never refreshed |
| `total_mcap_50pct_below_fair_value` | April figures; never refreshed |
| `whale_activity_low` | Same "no capitulation yet" thesis as `volume_spike_absent` |

**Result:** 19 active flags. The raw total moves from −6.7 to −3.4, which still caps at −2, so **the score does not change**.

### 2. Fix the price bands in `btc_key_levels`

Back up first (`create table btc_key_levels_backup_YYYYMMDD as table btc_key_levels`), then apply as migration `0011`.

- **Bug:** `get_btc_signal` falls back to a base of 1.5 when the price is below every band (below $31,750), so the bottom of the market reads SELL. Change the fallback to the highest base score, `(select max(base_score) from btc_key_levels)`. Keep `SET search_path` and the ±2 cap exactly as in migration `0010`.
- **Duplicates:** two rows at $38K (ids 14 and 16), plus a separate $39K "balance price" row (id 13). Cowen now puts the balance price at ~$38K, so merge them into one $38K row with base 9.7 and drop id 13.
- **Ordering:** a deeper price should never score lower.
  - $57K (id 15): base 8.6 → 8.2 (it currently beats $55K at 8.5).
  - $42K (id 12): base 9.0 → 9.3 (it currently loses to $45K at 9.2).
- **Labels and notes:** refresh $83K, $75K and $57K from the latest `btc_video_analysis.key_levels`. Paraphrase only:
  - $83K: the May high, where the breakout failed after two weekly closes below it in late October.
  - $75K: inside the bull-market support band of ~$73–77K (21-week EMA ~$76.6K, 200-day MA ~$73K).
  - $57K: the summer (1 July) low.

None of these change the signal at any price BTC has traded at this year, apart from the below-$31,750 bug.

### 3. Open question for Lincoln

Should key levels also come from the Cowen workflow, as suggestions he approves (a `btc_level_suggestions` table like the flags one)? Not started.

## Other open items

- **Docs in Lincoln's local folders:** `Claude/PropertyAnalyser` and `Claude/Crypto database` on his computer need updating to match the above, the ±2 cap and the 20-flag limit. The cloud session couldn't reach them.
- **Remaining tasks from [[Data Pipelines Handoff 2026-11-02]] §5:**
  - Show Cowen video summaries on `/btc`.
  - Relabel Locations "median price" as an asking price.
  - Add monitoring to `/audit`.
  - Drop the 2 Nov backup tables once confirmed.
