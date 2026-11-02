-- ALREADY APPLIED to production on 2026-11-02 (Supabase migration 20261102031848,
-- "btc_video_analysis_and_flag_suggestions"). Kept here as a record — do not re-run.
-- NOTE: apply_btc_flag_suggestion() below was replaced in 0010 (adds the 20-flag limit),
-- and the status check was widened in 0010 (adds 'superseded').

-- One row per Benjamin Cowen video the workflow has processed
create table public.btc_video_analysis (
  video_id         text primary key,
  title            text not null,
  published_at     timestamptz,
  url              text,
  transcript_chars integer,
  summary          text,                  -- short paraphrased summary (for the website)
  stance           text check (stance in ('bullish', 'bearish', 'neutral', 'mixed')),
  key_levels       text,
  model            text,
  status           text not null default 'analysed' check (status in ('analysed', 'failed')),
  error            text,
  created_at       timestamptz not null default now()
);

-- Changes to btc_context_flags suggested from each video, waiting for approval
create table public.btc_flag_suggestions (
  id             bigserial primary key,
  video_id       text references public.btc_video_analysis(video_id) on delete cascade,
  action         text not null check (action in ('add', 'update', 'retire')),
  flag_key       text not null,
  flag_label     text,
  category       text,
  current_value  text,
  score_modifier numeric,
  is_active      boolean,
  reason         text,
  status         text not null default 'pending'
                 check (status in ('pending', 'approved', 'rejected', 'applied', 'failed')),
  apply_note     text,
  applied_at     timestamptz,
  created_at     timestamptz not null default now()
);
create index btc_flag_suggestions_status_idx on public.btc_flag_suggestions (status, created_at desc);

-- Same access pattern as the other tables: anyone can read, only the service key (n8n) / dashboard can write
alter table public.btc_video_analysis enable row level security;
alter table public.btc_flag_suggestions enable row level security;
create policy public_read on public.btc_video_analysis for select to anon, authenticated using (true);
create policy public_read on public.btc_flag_suggestions for select to anon, authenticated using (true);

-- Approving a suggestion (status pending -> approved) applies it to btc_context_flags automatically
create or replace function public.apply_btc_flag_suggestion()
returns trigger
language plpgsql
set search_path = public
as $$
declare
  hit integer;
begin
  if not (new.status = 'approved' and old.status is distinct from 'approved') then
    return new;
  end if;

  if new.action = 'add' then
    insert into btc_context_flags (flag_key, flag_label, category, is_bearish, score_modifier, current_value, is_active, notes, updated_at)
    values (new.flag_key, coalesce(new.flag_label, new.flag_key), new.category,
            coalesce(new.score_modifier, 0) < 0, coalesce(new.score_modifier, 0),
            new.current_value, coalesce(new.is_active, true),
            'Added from video ' || coalesce(new.video_id, '?'), now())
    on conflict (flag_key) do update set
      flag_label     = excluded.flag_label,
      category       = coalesce(excluded.category, btc_context_flags.category),
      is_bearish     = excluded.is_bearish,
      score_modifier = excluded.score_modifier,
      current_value  = excluded.current_value,
      is_active      = excluded.is_active,
      notes          = excluded.notes,
      updated_at     = now();
    hit := 1;

  elsif new.action = 'update' then
    update btc_context_flags set
      flag_label     = coalesce(new.flag_label, flag_label),
      current_value  = coalesce(new.current_value, current_value),
      score_modifier = coalesce(new.score_modifier, score_modifier),
      is_bearish     = case when new.score_modifier is not null then new.score_modifier < 0 else is_bearish end,
      is_active      = coalesce(new.is_active, is_active),
      notes          = 'Updated from video ' || coalesce(new.video_id, '?'),
      updated_at     = now()
    where flag_key = new.flag_key;
    get diagnostics hit = row_count;

  elsif new.action = 'retire' then
    update btc_context_flags set
      is_active     = false,
      current_value = coalesce(new.current_value, current_value),
      notes         = 'Retired from video ' || coalesce(new.video_id, '?'),
      updated_at    = now()
    where flag_key = new.flag_key;
    get diagnostics hit = row_count;
  end if;

  if hit = 0 then
    new.status := 'failed';
    new.apply_note := 'No flag called ' || new.flag_key || ' exists';
  else
    new.status := 'applied';
    new.applied_at := now();
    new.apply_note := null;
  end if;
  return new;
end;
$$;

create trigger apply_on_approve
before update of status on public.btc_flag_suggestions
for each row execute function public.apply_btc_flag_suggestion();

comment on table public.btc_flag_suggestions is 'AI-suggested changes to btc_context_flags from Benjamin Cowen videos. Set status to approved to apply, or rejected to dismiss.';
