export function SetupNotice() {
  return (
    <div className="rounded-xl border border-slate-800 bg-slate-900/60 p-8 text-sm text-slate-300">
      <h1 className="mb-2 text-lg font-semibold text-white">Supabase not configured</h1>
      <p>
        Set <code className="rounded bg-slate-800 px-1">NEXT_PUBLIC_SUPABASE_URL</code> and{' '}
        <code className="rounded bg-slate-800 px-1">NEXT_PUBLIC_SUPABASE_ANON_KEY</code> (see{' '}
        <code className="rounded bg-slate-800 px-1">web/.env.example</code>), then reload.
      </p>
    </div>
  );
}
