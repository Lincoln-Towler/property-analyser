-- ALREADY APPLIED to production on 2026-04-19 (Supabase migration 20260419005505,
-- "create_get_btc_signal_function"). Added as a record on 2026-11-02 — do not re-run.
-- NOTE: replaced in 0010 (the ±2 cap) and again in 0013 (the below-every-band fix).

CREATE OR REPLACE FUNCTION get_btc_signal(price_usd NUMERIC)
RETURNS JSON
LANGUAGE plpgsql
AS $$
DECLARE
  lower_lvl        RECORD;
  upper_lvl        RECORD;
  base_score       NUMERIC(3,1);
  modifier_total   NUMERIC(3,1);
  final_score      NUMERIC(3,1);
  signal_text      TEXT;
  intensity_text   TEXT;
  action_text      TEXT;
  summary_line     TEXT;
  pct_to_resist    NUMERIC(5,1);
  pct_to_support   NUMERIC(5,1);
  active_flags     JSON;
  result           JSON;
BEGIN
  -- Band below price (support)
  SELECT * INTO lower_lvl
  FROM btc_key_levels
  WHERE level_usd <= price_usd
  ORDER BY level_usd DESC
  LIMIT 1;

  -- Band above price (resistance)
  SELECT * INTO upper_lvl
  FROM btc_key_levels
  WHERE level_usd > price_usd
  ORDER BY level_usd ASC
  LIMIT 1;

  -- Base score from the band we're sitting in
  base_score := COALESCE(lower_lvl.base_score, 1.5);

  -- Sum all active modifiers
  SELECT COALESCE(SUM(score_modifier), 0)
  INTO modifier_total
  FROM btc_context_flags
  WHERE is_active = true;

  -- Final score clamped 1–10
  final_score := GREATEST(1.0, LEAST(10.0, base_score + modifier_total));

  -- Signal text
  signal_text := CASE
    WHEN final_score >= 9.0 THEN 'MAX BUY'
    WHEN final_score >= 7.5 THEN 'STRONG BUY'
    WHEN final_score >= 6.0 THEN 'BUY'
    WHEN final_score >= 5.0 THEN 'LIGHT BUY'
    WHEN final_score >= 4.0 THEN 'HOLD'
    WHEN final_score >= 2.5 THEN 'CAUTION / REDUCE'
    ELSE                         'SELL / TAKE PROFIT'
  END;

  -- Intensity / action guidance
  intensity_text := CASE
    WHEN final_score >= 9.0 THEN 'Aggressive — deploy significant capital, high conviction zone'
    WHEN final_score >= 7.5 THEN 'Moderate — scale in steadily, strong value area'
    WHEN final_score >= 6.0 THEN 'Light — small staged positions, patience still rewarded'
    WHEN final_score >= 5.0 THEN 'Cautious light — only tiny nibbles, wait for better levels'
    WHEN final_score >= 4.0 THEN 'Hold — no new entries, protect capital'
    WHEN final_score >= 2.5 THEN 'Reduce — trim longs on any bounce, risk is elevated'
    ELSE                         'Take profit — near or above key resistance, not a buy zone'
  END;

  -- Distance calculations
  pct_to_resist := CASE
    WHEN upper_lvl.level_usd IS NOT NULL
    THEN ROUND(((upper_lvl.level_usd - price_usd) / price_usd) * 100, 1)
    ELSE NULL
  END;

  pct_to_support := CASE
    WHEN lower_lvl.level_usd IS NOT NULL
    THEN ROUND(((price_usd - lower_lvl.level_usd) / price_usd) * 100, 1)
    ELSE NULL
  END;

  -- Collect active flag labels for transparency
  SELECT json_agg(flag_label ORDER BY score_modifier ASC)
  INTO active_flags
  FROM btc_context_flags
  WHERE is_active = true;

  -- One-line summary for email
  summary_line := format(
    '%s | Score %s/10 | %s',
    signal_text,
    final_score,
    intensity_text
  );

  -- Log to history table
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
    'price_usd',           price_usd,
    'signal',              signal_text,
    'intensity',           intensity_text,
    'score',               final_score,
    'score_breakdown',     json_build_object(
                             'base_score',     base_score,
                             'modifier_total', modifier_total,
                             'active_flags',   active_flags
                           ),
    'current_band',        COALESCE(lower_lvl.label, 'Below all defined levels'),
    'resistance',          json_build_object(
                             'level',   upper_lvl.level_usd,
                             'label',   upper_lvl.label,
                             'pct_away', pct_to_resist
                           ),
    'support',             json_build_object(
                             'level',   lower_lvl.level_usd,
                             'label',   lower_lvl.label,
                             'pct_away', pct_to_support
                           ),
    'summary_line',        summary_line
  );

  RETURN result;
END;
$$;
