-- ALREADY APPLIED to production on 2026-11-02 (Supabase migration 20261102121027,
-- "btc_flags_apply_automatically"). Kept here as a record — do not re-run.

-- No more approval step: a row saved to btc_flag_suggestions applies straight away.
-- The workflow needs no change, because it already saves rows here without a status.
--
-- What takes the place of a person checking each change:
--   * each flag stays within ±0.5 (also enforced here now), and flags together
--     move the score at most ±2 (get_btc_signal)
--   * when 20 flags are active, the one that's gone longest without a refresh
--     retires to make room, so the list can't jam or grow
--   * one bad row is recorded as failed instead of losing the whole batch
-- The table stays as the log of what changed and why.

DROP TRIGGER apply_on_approve ON public.btc_flag_suggestions;
DROP TRIGGER supersede_older ON public.btc_flag_suggestions;
DROP FUNCTION public.supersede_older_btc_suggestions();

ALTER TABLE public.btc_flag_suggestions ALTER COLUMN status SET DEFAULT 'applied';

CREATE OR REPLACE FUNCTION public.apply_btc_flag_suggestion()
RETURNS trigger
LANGUAGE plpgsql
SET search_path = public
AS $$
DECLARE
  max_active CONSTANT integer := 20;
  origin text := coalesce('video ' || new.video_id, 'manual cleanup');
  evicted text;
  hit integer;
BEGIN
  IF new.score_modifier IS NOT NULL THEN
    new.score_modifier := greatest(-0.5, least(0.5, new.score_modifier));
  END IF;

  BEGIN
    -- Would this make a flag active that isn't active now? If the list is full, make room.
    IF ((new.action = 'add' AND coalesce(new.is_active, true)) OR (new.action = 'update' AND new.is_active IS TRUE))
       AND NOT EXISTS (SELECT 1 FROM btc_context_flags WHERE flag_key = new.flag_key AND is_active)
       AND (SELECT count(*) FROM btc_context_flags WHERE is_active) >= max_active THEN
      UPDATE btc_context_flags SET
        is_active  = false,
        notes      = 'Retired automatically to make room for ' || new.flag_key,
        updated_at = now()
      WHERE flag_key = (
        SELECT flag_key FROM btc_context_flags WHERE is_active
        ORDER BY updated_at ASC NULLS FIRST, flag_key LIMIT 1
      )
      RETURNING flag_key INTO evicted;
    END IF;

    IF new.action = 'add' THEN
      INSERT INTO btc_context_flags (flag_key, flag_label, category, is_bearish, score_modifier, current_value, is_active, notes, updated_at)
      VALUES (new.flag_key, coalesce(new.flag_label, new.flag_key), new.category,
              coalesce(new.score_modifier, 0) < 0, coalesce(new.score_modifier, 0),
              new.current_value, coalesce(new.is_active, true),
              'Added from ' || origin, now())
      ON CONFLICT (flag_key) DO UPDATE SET
        flag_label     = excluded.flag_label,
        category       = coalesce(excluded.category, btc_context_flags.category),
        is_bearish     = excluded.is_bearish,
        score_modifier = excluded.score_modifier,
        current_value  = excluded.current_value,
        is_active      = excluded.is_active,
        notes          = excluded.notes,
        updated_at     = now();
      hit := 1;

    ELSIF new.action = 'update' THEN
      UPDATE btc_context_flags SET
        flag_label     = coalesce(new.flag_label, flag_label),
        current_value  = coalesce(new.current_value, current_value),
        score_modifier = coalesce(new.score_modifier, score_modifier),
        is_bearish     = CASE WHEN new.score_modifier IS NOT NULL THEN new.score_modifier < 0 ELSE is_bearish END,
        is_active      = coalesce(new.is_active, is_active),
        notes          = 'Updated from ' || origin,
        updated_at     = now()
      WHERE flag_key = new.flag_key;
      GET DIAGNOSTICS hit = ROW_COUNT;

    ELSIF new.action = 'retire' THEN
      UPDATE btc_context_flags SET
        is_active     = false,
        current_value = coalesce(new.current_value, current_value),
        notes         = 'Retired from ' || origin,
        updated_at    = now()
      WHERE flag_key = new.flag_key;
      GET DIAGNOSTICS hit = ROW_COUNT;
    END IF;

    -- Raising undoes this block, including any flag retired to make room above
    IF hit = 0 THEN
      RAISE EXCEPTION 'No flag called %', new.flag_key;
    END IF;

    new.status := 'applied';
    new.applied_at := now();
    new.apply_note := CASE WHEN evicted IS NOT NULL THEN 'Retired ' || evicted || ' to make room' END;
  EXCEPTION WHEN OTHERS THEN
    new.status := 'failed';
    new.applied_at := NULL;
    new.apply_note := left(SQLERRM, 300);
  END;

  RETURN new;
END;
$$;

CREATE TRIGGER apply_on_insert
BEFORE INSERT ON public.btc_flag_suggestions
FOR EACH ROW EXECUTE FUNCTION public.apply_btc_flag_suggestion();

COMMENT ON TABLE public.btc_flag_suggestions IS 'Log of changes to btc_context_flags from Benjamin Cowen videos. Saving a row applies it straight away; status records applied or failed. Older rows may show rejected or superseded, from when changes needed approval.';
