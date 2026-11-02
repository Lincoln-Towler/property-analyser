-- ALREADY APPLIED to production on 2026-11-02 (Supabase migration 20261102112126,
-- "btc_score_cap_flag_limit_and_supersede"). Kept here as a record — do not re-run.
--
-- 1) get_btc_signal(): the flags' combined effect is capped at ±2 points.
-- 2) btc_flag_suggestions.status gains 'superseded'.
-- 3) supersede_older trigger: only one pending suggestion per flag; the newest video wins.
-- 4) apply_btc_flag_suggestion(): hard limit of 20 active flags.

-- 1) Score: flags together can move the score at most ±2 points (price band does the rest)
CREATE OR REPLACE FUNCTION public.get_btc_signal(price_usd numeric)
 RETURNS json
 LANGUAGE plpgsql
 SET search_path TO 'public'
AS $function$
DECLARE
  modifier_cap     CONSTANT NUMERIC := 2.0;   -- max combined effect of all active flags (±)
  lower_lvl        RECORD;
  upper_lvl        RECORD;
  base_score       NUMERIC(3,1);
  raw_modifier     NUMERIC(5,1);
  modifier_total   NUMERIC(3,1);
  final_score      NUMERIC(3,1);
  signal_text      TEXT;
  intensity_text   TEXT;
  summary_line     TEXT;
  pct_to_resist    NUMERIC(5,1);
  pct_to_support   NUMERIC(5,1);
  active_flags     JSON;
  result           JSON;
BEGIN
  -- Band below price (support)
  SELECT * INTO lower_lvl FROM btc_key_levels WHERE level_usd <= price_usd ORDER BY level_usd DESC LIMIT 1;
  -- Band above price (resistance)
  SELECT * INTO upper_lvl FROM btc_key_levels WHERE level_usd > price_usd ORDER BY level_usd ASC LIMIT 1;

  -- Base score from the band we're sitting in
  base_score := COALESCE(lower_lvl.base_score, 1.5);

  -- Sum all active modifiers, then cap their combined effect
  SELECT COALESCE(SUM(score_modifier), 0) INTO raw_modifier FROM btc_context_flags WHERE is_active = true;
  modifier_total := GREATEST(-modifier_cap, LEAST(modifier_cap, raw_modifier));

  -- Final score clamped 1–10
  final_score := GREATEST(1.0, LEAST(10.0, base_score + modifier_total));

  signal_text := CASE
    WHEN final_score >= 9.0 THEN 'MAX BUY'
    WHEN final_score >= 7.5 THEN 'STRONG BUY'
    WHEN final_score >= 6.0 THEN 'BUY'
    WHEN final_score >= 5.0 THEN 'LIGHT BUY'
    WHEN final_score >= 4.0 THEN 'HOLD'
    WHEN final_score >= 2.5 THEN 'CAUTION / REDUCE'
    ELSE                         'SELL / TAKE PROFIT'
  END;

  intensity_text := CASE
    WHEN final_score >= 9.0 THEN 'Aggressive — deploy significant capital, high conviction zone'
    WHEN final_score >= 7.5 THEN 'Moderate — scale in steadily, strong value area'
    WHEN final_score >= 6.0 THEN 'Light — small staged positions, patience still rewarded'
    WHEN final_score >= 5.0 THEN 'Cautious light — only tiny nibbles, wait for better levels'
    WHEN final_score >= 4.0 THEN 'Hold — no new entries, protect capital'
    WHEN final_score >= 2.5 THEN 'Reduce — trim longs on any bounce, risk is elevated'
    ELSE                         'Take profit — near or above key resistance, not a buy zone'
  END;

  pct_to_resist := CASE WHEN upper_lvl.level_usd IS NOT NULL
    THEN ROUND(((upper_lvl.level_usd - price_usd) / price_usd) * 100, 1) ELSE NULL END;
  pct_to_support := CASE WHEN lower_lvl.level_usd IS NOT NULL
    THEN ROUND(((price_usd - lower_lvl.level_usd) / price_usd) * 100, 1) ELSE NULL END;

  SELECT json_agg(flag_label ORDER BY score_modifier ASC) INTO active_flags
  FROM btc_context_flags WHERE is_active = true;

  summary_line := format('%s | Score %s/10 | %s', signal_text, final_score, intensity_text);

  INSERT INTO btc_signal_log (
    price_usd, signal, intensity, score, base_score, modifier_total,
    current_band, resistance_level, support_level,
    pct_to_resistance, pct_to_support, summary_line
  ) VALUES (
    price_usd, signal_text, intensity_text, final_score, base_score, modifier_total,
    COALESCE(lower_lvl.label, 'Below all defined levels'),
    upper_lvl.level_usd, lower_lvl.level_usd,
    pct_to_resist, pct_to_support, summary_line
  );

  result := json_build_object(
    'price_usd',       price_usd,
    'signal',          signal_text,
    'intensity',       intensity_text,
    'score',           final_score,
    'score_breakdown', json_build_object(
                         'base_score',         base_score,
                         'modifier_total',     modifier_total,
                         'raw_modifier_total', raw_modifier,
                         'modifier_cap',       modifier_cap,
                         'active_flags',       active_flags),
    'current_band',    COALESCE(lower_lvl.label, 'Below all defined levels'),
    'resistance',      json_build_object('level', upper_lvl.level_usd, 'label', upper_lvl.label, 'pct_away', pct_to_resist),
    'support',         json_build_object('level', lower_lvl.level_usd, 'label', lower_lvl.label, 'pct_away', pct_to_support),
    'summary_line',    summary_line
  );
  RETURN result;
END;
$function$;

-- 2) New status for suggestions replaced by a newer one
ALTER TABLE public.btc_flag_suggestions DROP CONSTRAINT btc_flag_suggestions_status_check;
ALTER TABLE public.btc_flag_suggestions ADD CONSTRAINT btc_flag_suggestions_status_check
  CHECK (status IN ('pending', 'approved', 'rejected', 'applied', 'failed', 'superseded'));

-- 3) Only one pending suggestion per flag: the one from the newest video wins
CREATE OR REPLACE FUNCTION public.supersede_older_btc_suggestions()
RETURNS trigger
LANGUAGE plpgsql
SET search_path = public
AS $$
DECLARE
  newer_exists boolean;
BEGIN
  IF new.status IS DISTINCT FROM 'pending' THEN
    RETURN new;
  END IF;

  SELECT EXISTS (
    SELECT 1
    FROM btc_flag_suggestions s
    JOIN btc_video_analysis v  ON v.video_id = s.video_id
    JOIN btc_video_analysis nv ON nv.video_id = new.video_id
    WHERE s.flag_key = new.flag_key AND s.status = 'pending' AND v.published_at > nv.published_at
  ) INTO newer_exists;

  IF newer_exists THEN
    new.status := 'superseded';
    new.apply_note := 'A newer video already has a pending suggestion for this flag';
  ELSE
    UPDATE btc_flag_suggestions
       SET status = 'superseded', apply_note = 'Replaced by a newer suggestion for the same flag'
     WHERE flag_key = new.flag_key AND status = 'pending';
  END IF;
  RETURN new;
END;
$$;

CREATE TRIGGER supersede_older
BEFORE INSERT ON public.btc_flag_suggestions
FOR EACH ROW EXECUTE FUNCTION public.supersede_older_btc_suggestions();

-- 4) Hard limit: at most 20 active flags. Adding (or re-activating) a flag past the limit fails until one is retired.
CREATE OR REPLACE FUNCTION public.apply_btc_flag_suggestion()
RETURNS trigger
LANGUAGE plpgsql
SET search_path = public
AS $$
DECLARE
  max_active CONSTANT integer := 20;
  active_now integer;
  hit integer;
BEGIN
  IF NOT (new.status = 'approved' AND old.status IS DISTINCT FROM 'approved') THEN
    RETURN new;
  END IF;

  -- Would this make a flag active that isn't active now?
  IF (new.action = 'add' OR (new.action = 'update' AND new.is_active IS TRUE))
     AND NOT EXISTS (SELECT 1 FROM btc_context_flags WHERE flag_key = new.flag_key AND is_active) THEN
    SELECT count(*) INTO active_now FROM btc_context_flags WHERE is_active;
    IF active_now >= max_active THEN
      new.status := 'failed';
      new.apply_note := format('Flag limit reached (%s active, max %s). Retire a flag, then set this back to approved.', active_now, max_active);
      RETURN new;
    END IF;
  END IF;

  IF new.action = 'add' THEN
    INSERT INTO btc_context_flags (flag_key, flag_label, category, is_bearish, score_modifier, current_value, is_active, notes, updated_at)
    VALUES (new.flag_key, coalesce(new.flag_label, new.flag_key), new.category,
            coalesce(new.score_modifier, 0) < 0, coalesce(new.score_modifier, 0),
            new.current_value, coalesce(new.is_active, true),
            'Added from video ' || coalesce(new.video_id, '?'), now())
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
      notes          = 'Updated from video ' || coalesce(new.video_id, '?'),
      updated_at     = now()
    WHERE flag_key = new.flag_key;
    GET DIAGNOSTICS hit = ROW_COUNT;

  ELSIF new.action = 'retire' THEN
    UPDATE btc_context_flags SET
      is_active     = false,
      current_value = coalesce(new.current_value, current_value),
      notes         = 'Retired from video ' || coalesce(new.video_id, '?'),
      updated_at    = now()
    WHERE flag_key = new.flag_key;
    GET DIAGNOSTICS hit = ROW_COUNT;
  END IF;

  IF hit = 0 THEN
    new.status := 'failed';
    new.apply_note := 'No flag called ' || new.flag_key || ' exists';
  ELSE
    new.status := 'applied';
    new.applied_at := now();
    new.apply_note := null;
  END IF;
  RETURN new;
END;
$$;

COMMENT ON FUNCTION public.get_btc_signal(numeric) IS 'BTC score = price band base score + active flag modifiers, with the flags'' combined effect capped at ±2, clamped 1–10.';
