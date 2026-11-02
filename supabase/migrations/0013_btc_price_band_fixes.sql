-- ALREADY APPLIED to production on 2026-11-02 (Supabase migration 20261102120948,
-- "btc_price_band_fixes"). Kept here as a record — do not re-run.

-- 1) get_btc_signal(): a price below every band now gets the deepest band's score.
--    It used to fall back to 1.5, so the bottom of the market read SELL.
--    The rest of the function, including the ±2 cap, is unchanged from 0010.
-- 2) Merges the duplicate $38K and $39K rows into one $38K row.
-- 3) Fixes two bands that scored out of order, and refreshes three labels.
-- The previous rows are in btc_key_levels_backup_20261102.

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

  -- Base score from the band we're sitting in; below every band, the deepest band's score
  base_score := COALESCE(lower_lvl.base_score, (SELECT max(k.base_score) FROM btc_key_levels k));

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

-- Merge the balance-price rows: Cowen now puts it at ~$38K, the same level as the 70% drop target
UPDATE public.btc_key_levels SET
  label = 'Balance Price / 70% drop from $126K ATH — Cowen base case',
  source = 'Cowen Late Apr 26 + ITC Jul-2026',
  notes = '70% from $126K = ~$37.8K. The realistic bear case. Not doomer — just what Bitcoin does every cycle. Balance price now ~$38K (was ~$39K in April; ~$37-38K confirmed in Sep-2026). All major bottoms 2011/2015/2018/2020/2022 went below it, and it has been visited in midterm bear markets. Cowen notes this is "greedy" but possible if fear overtakes the market late 2026. Below realized price ($53K) and above the 2022 cycle low floor. Sits between the $40K cycle low zone and the $31.75K all-time floor.',
  updated_at = now()
WHERE level_usd = 38000 AND signal = 'MAX_BUY';
DELETE FROM public.btc_key_levels WHERE level_usd = 38000 AND signal = 'STRONG_BUY';
DELETE FROM public.btc_key_levels WHERE level_usd = 39000;

-- A deeper price should never score lower
UPDATE public.btc_key_levels SET
  base_score = 8.2,
  label = 'Summer low (1 July) — 10x the 2018 sweep low',
  updated_at = now()
WHERE level_usd = 57000;
UPDATE public.btc_key_levels SET base_score = 9.3, updated_at = now() WHERE level_usd = 42000;

-- Refresh from the latest videos (paraphrased)
UPDATE public.btc_key_levels SET
  label = 'May high — breakout failed, two weekly closes below (late Oct)',
  notes = '[Nov-2026] The May high; the breakout failed after two weekly closes below it in late October. ' || coalesce(notes, ''),
  source = coalesce(source || '; ', '') || 'Cowen Nov 2026',
  updated_at = now()
WHERE level_usd = 83000;
UPDATE public.btc_key_levels SET
  label = 'Inside bull-market support band (~$73–77K)',
  notes = '[Nov-2026] Inside the bull-market support band of ~$73–77K (21-week EMA ~$76.6K, 200-day MA ~$73K). ' || coalesce(notes, ''),
  source = coalesce(source || '; ', '') || 'Cowen Nov 2026',
  updated_at = now()
WHERE level_usd = 75000;

-- One row per price level from here on
ALTER TABLE public.btc_key_levels ADD CONSTRAINT btc_key_levels_level_usd_key UNIQUE (level_usd);

-- Roll the whole thing back if the ladder isn't in order
DO $$
BEGIN
  IF (SELECT count(*) FROM public.btc_key_levels) <> 14 THEN
    RAISE EXCEPTION 'Expected 14 bands';
  END IF;
  IF EXISTS (
    SELECT 1 FROM (
      SELECT base_score, lag(base_score) OVER (ORDER BY level_usd DESC) AS higher_band
      FROM public.btc_key_levels
    ) s WHERE base_score < higher_band
  ) THEN
    RAISE EXCEPTION 'A deeper band scores lower than the one above it';
  END IF;
END $$;
