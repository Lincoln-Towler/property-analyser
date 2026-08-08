import { getSiteData } from '@/lib/data';
import { calculateRegionalDivergence } from '@/lib/scoring/engine';
import {
  scoreAllLocations,
  formatMetric,
  METRIC_LABELS,
} from '@/lib/scoring/location';
import { LocationCompare, type PriceSeriesPoint } from '@/components/LocationCompare';
import { SetupNotice } from '@/components/SetupNotice';

export const revalidate = 3600;

const TABLE_METRICS = [
  'median_price',
  'annual_growth',
  'rental_yield',
  'vacancy_rate',
  'days_on_market',
];

export default async function LocationsPage() {
  const data = await getSiteData();
  if (!data) return <SetupNotice />;

  const rows = data.propertyData;
  if (!rows.length) {
    const queryFailed = data.errors.propertyData;
    return (
      <div className="space-y-4">
        <h1 className="text-xl font-semibold">Location Analysis</h1>
        {queryFailed ? (
          <div className="rounded-xl border border-red-700/60 bg-red-950/30 p-6 text-sm text-slate-200">
            <p className="mb-2 font-medium">
              The property_data query failed — this is not an empty table.
            </p>
            <p className="mb-3 font-mono text-xs text-red-300">{queryFailed}</p>
            <p className="text-slate-300">
              Most likely Row Level Security: the site reads with the anon key, which needs a
              SELECT policy on <code className="rounded bg-slate-800 px-1">property_data</code>.
              Migration{' '}
              <code className="rounded bg-slate-800 px-1">0004_enable_rls.sql</code> creates one —
              check it was applied, then confirm in Supabase → Authentication → Policies.
            </p>
          </div>
        ) : (
          <p className="rounded-xl border border-slate-800 bg-slate-900/60 p-6 text-sm text-slate-300">
            No property data in the last 12 months. Add rows to{' '}
            <code className="rounded bg-slate-800 px-1">property_data</code> (location,
            metric_name, value, date) in Supabase to populate this page.
          </p>
        )}
      </div>
    );
  }

  const scored = scoreAllLocations(rows);
  const divergence = calculateRegionalDivergence(rows, new Date());

  // Wide-format median price series for the comparison chart
  const priceRows = rows.filter((r) => r.metric_name === 'median_price');
  const dates = [...new Set(priceRows.map((r) => r.date))].sort();
  const priceSeries: PriceSeriesPoint[] = dates.map((date) => {
    const point: PriceSeriesPoint = { date };
    for (const row of priceRows.filter((r) => r.date === date)) {
      point[row.location] = row.value;
    }
    return point;
  });

  const best = scored[0];
  const worst = scored[scored.length - 1];

  return (
    <div className="space-y-8">
      <header>
        <h1 className="text-xl font-semibold">Location Analysis</h1>
        <p className="mt-1 text-sm text-slate-400">
          {scored.length} locations scored on their latest readings. The investment score starts at
          50 and adjusts for annual growth, rental yield, vacancy rate and days on market —
          the same formula the Streamlit app used.
        </p>
      </header>

      {divergence && (
        <section className="rounded-xl border border-slate-800 bg-slate-900/60 p-5">
          <h2 className="mb-2 text-sm font-medium uppercase tracking-wide text-slate-400">
            Regional divergence
          </h2>
          <p className="text-sm text-slate-300">
            Spread of <strong>{divergence.divergence}</strong> points between the strongest and
            weakest markets ({best.location} {best.score} → {worst.location} {worst.score}).{' '}
            {divergence.recommendation === 'selective' ? (
              <>
                Divergence is high — this is a <strong>selective</strong> market where location
                choice matters more than timing.
              </>
            ) : (
              <>
                Divergence is low — markets are moving together, so a <strong>broad</strong>{' '}
                approach is reasonable.
              </>
            )}
          </p>
        </section>
      )}

      <section>
        <h2 className="mb-3 text-sm font-medium uppercase tracking-wide text-slate-400">
          All locations, ranked
        </h2>
        <div className="overflow-x-auto rounded-xl border border-slate-800">
          <table className="w-full min-w-[760px] text-sm">
            <thead className="bg-slate-900 text-left text-xs uppercase tracking-wide text-slate-400">
              <tr>
                <th className="px-4 py-3">#</th>
                <th className="px-4 py-3">Location</th>
                <th className="px-4 py-3">Score</th>
                {TABLE_METRICS.map((m) => (
                  <th key={m} className="px-4 py-3">
                    {METRIC_LABELS[m]}
                  </th>
                ))}
                <th className="px-4 py-3">As of</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800">
              {scored.map((loc, i) => (
                <tr key={loc.location}>
                  <td className="px-4 py-3 tabular-nums text-slate-500">{i + 1}</td>
                  <td className="px-4 py-3 font-medium">{loc.location}</td>
                  <td className="px-4 py-3">
                    <span
                      className={`tabular-nums font-semibold ${
                        loc.score >= 70
                          ? 'text-emerald-400'
                          : loc.score >= 40
                            ? 'text-amber-400'
                            : 'text-red-400'
                      }`}
                    >
                      {loc.score}
                    </span>
                  </td>
                  {TABLE_METRICS.map((m) => (
                    <td key={m} className="px-4 py-3 tabular-nums text-slate-300">
                      {formatMetric(m, loc.metrics[m]?.value)}
                    </td>
                  ))}
                  <td className="px-4 py-3 tabular-nums text-xs text-slate-500">
                    {loc.latestDate ?? '—'}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      <section>
        <h2 className="mb-3 text-sm font-medium uppercase tracking-wide text-slate-400">
          Head to head
        </h2>
        <LocationCompare scored={scored} priceSeries={priceSeries} />
      </section>

      <p className="text-xs text-slate-500">
        Data as of {data.fetchedAt.slice(0, 16).replace('T', ' ')} UTC
      </p>
    </div>
  );
}
