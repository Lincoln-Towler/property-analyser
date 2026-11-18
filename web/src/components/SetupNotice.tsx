export function SetupNotice() {
  return (
    <div className="rounded-xl bg-surface p-8 text-sm text-secondary">
      <h1 className="mb-2 text-lg font-semibold text-primary">Supabase not configured</h1>
      <p>
        Set <code className="rounded bg-tile px-1">NEXT_PUBLIC_SUPABASE_URL</code> and{' '}
        <code className="rounded bg-tile px-1">NEXT_PUBLIC_SUPABASE_ANON_KEY</code> (see{' '}
        <code className="rounded bg-tile px-1">web/.env.example</code>), then reload.
      </p>
    </div>
  );
}
