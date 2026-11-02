-- ALREADY APPLIED to production on 2026-11-02 (Supabase migration 20261102032101,
-- "btc_video_analysis_attempts"). Kept here as a record — do not re-run.

alter table public.btc_video_analysis add column attempts integer not null default 1;
comment on column public.btc_video_analysis.attempts is 'How many times the workflow has tried this video; failed videos are retried until 3 attempts.';
