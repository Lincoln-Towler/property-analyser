-- ALREADY APPLIED to production on 2026-11-02 (Supabase migration 20261102120845,
-- "btc_backup_before_automation_20261102"). Kept here as a record — do not re-run.

-- Full snapshot of the BTC tables before the flag cleanup, band fixes and the
-- switch to automatic flag changes, so any of it can be undone.
create table public.btc_context_flags_backup_20261102 as table public.btc_context_flags;
create table public.btc_flag_suggestions_backup_20261102 as table public.btc_flag_suggestions;
create table public.btc_key_levels_backup_20261102 as table public.btc_key_levels;
-- Keep the backups private (no API access)
alter table public.btc_context_flags_backup_20261102 enable row level security;
alter table public.btc_flag_suggestions_backup_20261102 enable row level security;
alter table public.btc_key_levels_backup_20261102 enable row level security;
comment on table public.btc_context_flags_backup_20261102 is 'Snapshot before the BTC changes on 2026-11-02. Safe to drop once happy.';
comment on table public.btc_flag_suggestions_backup_20261102 is 'Snapshot before the BTC changes on 2026-11-02. Safe to drop once happy.';
comment on table public.btc_key_levels_backup_20261102 is 'Snapshot before the BTC changes on 2026-11-02. Safe to drop once happy.';
