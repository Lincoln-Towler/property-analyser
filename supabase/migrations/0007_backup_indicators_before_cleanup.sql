-- ALREADY APPLIED to production on 2026-11-02 (Supabase migration 20261102025631,
-- "backup_indicators_before_cleanup_20261102"). Kept here as a record — do not re-run.
--
-- Full snapshot of both indicator tables before the 2 Nov 2026 cleanup, so any change can be undone.
-- Drop both tables (in a later migration) once Lincoln is happy with the cleanup.

create table public.economic_indicators_backup_20261102 as table public.economic_indicators;
create table public.economic_indicators_history_backup_20261102 as table public.economic_indicators_history;
-- Keep the backups private (no API access)
alter table public.economic_indicators_backup_20261102 enable row level security;
alter table public.economic_indicators_history_backup_20261102 enable row level security;
comment on table public.economic_indicators_backup_20261102 is 'Snapshot before data cleanup on 2026-11-02. Safe to drop once happy with the cleanup.';
comment on table public.economic_indicators_history_backup_20261102 is 'Snapshot before data cleanup on 2026-11-02. Safe to drop once happy with the cleanup.';
