-- ALREADY APPLIED to production on 2026-11-02 (Supabase migration 20261102121103,
-- "public_access_read_only"). Kept here as a record — do not re-run.

-- Anyone using the public (anon) key, or a signed-up user, can only read.
-- Row Level Security already blocked their writes; this removes the underlying
-- permission as well, including for tables created later.
-- Automation is unaffected: n8n and the site's server code use the service role.

REVOKE INSERT, UPDATE, DELETE, TRUNCATE, REFERENCES, TRIGGER ON ALL TABLES IN SCHEMA public FROM anon, authenticated;
REVOKE ALL ON ALL SEQUENCES IN SCHEMA public FROM anon, authenticated;

ALTER DEFAULT PRIVILEGES FOR ROLE postgres IN SCHEMA public
  REVOKE INSERT, UPDATE, DELETE, TRUNCATE, REFERENCES, TRIGGER ON TABLES FROM anon, authenticated;
ALTER DEFAULT PRIVILEGES FOR ROLE postgres IN SCHEMA public
  REVOKE ALL ON SEQUENCES FROM anon, authenticated;

-- This function writes to the signal log, so only automation may call it
REVOKE EXECUTE ON FUNCTION public.get_btc_signal(numeric) FROM PUBLIC, anon, authenticated;
GRANT EXECUTE ON FUNCTION public.get_btc_signal(numeric) TO service_role;
