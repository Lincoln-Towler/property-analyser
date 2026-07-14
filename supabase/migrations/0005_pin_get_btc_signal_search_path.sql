-- 0005: Pin get_btc_signal's search_path (clears the security advisor WARN
-- "function_search_path_mutable").
--
-- Pinned to 'public' rather than '' because the function body references
-- btc_key_levels / btc_context_flags / btc_signal_log unqualified.
-- Applied to production 2026-07-14, immediately after 0004.

ALTER FUNCTION public.get_btc_signal(numeric) SET search_path = public;
