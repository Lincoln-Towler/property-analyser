-- ALREADY APPLIED to production on 2026-11-02 (Supabase migration 20261102120911,
-- "btc_flag_cleanup_20261102"). Kept here as a record — do not re-run.

-- Brings the active flags from 36 to 19: applies the 13 suggestions that were
-- waiting, then retires 12 more that duplicate another flag or have gone stale.
-- The raw total moves from -6.7 to -3.4, which still caps at -2, so the score
-- at any given price does not change.

do $$
begin
  if (select count(*) from public.btc_flag_suggestions where status = 'pending') <> 13 then
    raise exception 'Expected exactly 13 pending suggestions';
  end if;
end $$;

update public.btc_flag_suggestions set status = 'approved' where status = 'pending';

insert into public.btc_flag_suggestions (action, flag_key, is_active, reason) values
  ('retire', 'mvrv_not_extreme_low', false, 'Manual cleanup: already counted inside onchain_bottom_signals_absent (MVRV Z < 0 is one of its four checks).'),
  ('retire', 'btc_300w_ma_cluster_54k', false, 'Manual cleanup: the realized price (~$54K) is already inside onchain_bottom_signals_absent.'),
  ('retire', 'btc_dom_excl_stables_rising', false, 'Manual cleanup: covered by stablecoin_dom_not_peaked.'),
  ('retire', 'time_based_cap_incomplete', false, 'Manual cleanup: July status, superseded by jun_oct_2026_timeline and cycle_day_timing_confirms_bear.'),
  ('retire', 'midterm_avg_50k_target', false, 'Manual cleanup: June figures, same "lower Q4 low" thesis as other flags.'),
  ('retire', 'qqq_etf_analog_bear', false, 'Manual cleanup: an analog for the same downside targets as the on-chain and BTC/gold flags.'),
  ('retire', 'diminishing_returns_cycle', false, 'Manual cleanup: about the 2026 top, which has passed.'),
  ('retire', 'btc_leads_sp_breakdown', false, 'Manual cleanup: April status, covered by sp_april_low_not_confirmed now the equity correction is underway.'),
  ('retire', 'btc_relative_underperformance', false, 'Manual cleanup: May year-to-date figures, overlaps btc_gold_declining.'),
  ('retire', 'recession_approaching_gdp', false, 'Manual cleanup: built on Q4 2025 GDP and never refreshed.'),
  ('retire', 'total_mcap_50pct_below_fair_value', false, 'Manual cleanup: April figures, never refreshed.'),
  ('retire', 'whale_activity_low', false, 'Manual cleanup: same "no capitulation yet" thesis as volume_spike_absent.');

update public.btc_flag_suggestions set status = 'approved' where status = 'pending';

-- Roll the whole thing back if the result isn't exactly what was planned
do $$
declare
  n integer;
  raw numeric;
begin
  select count(*), sum(score_modifier) into n, raw from public.btc_context_flags where is_active;
  if n <> 19 or raw <> -3.4 then
    raise exception 'Unexpected result: % active flags, raw total %', n, raw;
  end if;
  if exists (select 1 from public.btc_flag_suggestions where status in ('pending', 'approved', 'failed')) then
    raise exception 'Some suggestions did not apply';
  end if;
end $$;
