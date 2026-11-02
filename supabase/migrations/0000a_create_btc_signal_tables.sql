-- ALREADY APPLIED to production on 2026-04-19 (Supabase migration 20260419005349,
-- "create_btc_signal_tables"). Added as a record on 2026-11-02 — do not re-run.

-- ── BTC Key Price Levels ─────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS btc_key_levels (
  id            SERIAL PRIMARY KEY,
  level_usd     INTEGER       NOT NULL,
  label         TEXT          NOT NULL,
  signal        TEXT          NOT NULL,  -- SELL | HOLD | LIGHT_BUY | BUY | STRONG_BUY | AGGRESSIVE_BUY | MAX_BUY
  base_score    NUMERIC(3,1)  NOT NULL,  -- 1–10, higher = more bullish
  emoji         TEXT,
  source        TEXT,
  notes         TEXT,
  updated_at    TIMESTAMPTZ   DEFAULT NOW()
);

-- ── Context / Modifier Flags ─────────────────────────────────────────────────
-- Each active flag applies a score_modifier to the base price-band score.
-- Negative = bearish drag (reduces buy conviction), Positive = bullish boost.
CREATE TABLE IF NOT EXISTS btc_context_flags (
  id              SERIAL PRIMARY KEY,
  flag_key        TEXT          UNIQUE NOT NULL,
  flag_label      TEXT          NOT NULL,
  category        TEXT,                    -- CYCLE | MACRO | ON_CHAIN | SENTIMENT
  is_bearish      BOOLEAN       NOT NULL,
  score_modifier  NUMERIC(3,1)  NOT NULL,
  current_value   TEXT,
  is_active       BOOLEAN       DEFAULT true,
  notes           TEXT,
  updated_at      TIMESTAMPTZ   DEFAULT NOW()
);

-- ── Signal History (auto-logged by function) ─────────────────────────────────
CREATE TABLE IF NOT EXISTS btc_signal_log (
  id              SERIAL PRIMARY KEY,
  logged_at       TIMESTAMPTZ   DEFAULT NOW(),
  price_usd       NUMERIC(12,2) NOT NULL,
  signal          TEXT,
  intensity       TEXT,
  score           NUMERIC(3,1),
  base_score      NUMERIC(3,1),
  modifier_total  NUMERIC(3,1),
  current_band    TEXT,
  resistance_level INTEGER,
  support_level    INTEGER,
  pct_to_resistance NUMERIC(5,1),
  pct_to_support    NUMERIC(5,1),
  summary_line    TEXT
);
