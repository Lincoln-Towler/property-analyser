import { getSiteData } from '@/lib/data';
import { calculateRegionalDivergence } from '@/lib/scoring/engine';
import {
  scoreAllLocations,
  formatMetric,
  METRIC_LABELS,
} from '@/lib/scoring/location';
import { LocationCompare, type PriceSeriesPoint } from '@/components/LocationCompare';
import { SetupNotice } from '@/components/SetupNotice';
import { Card, DataStamp, PageHeader, SectionTitle } from '@/components/ui';
import { LOCATION_GOOD, scoreTone } from '@/lib/theme';
import { fmtDate } from '@/lib/format';

export const revalidate = 3600;

function metricTone(metric: string, value: number | undefined): string {
  if (value === undefined) return 'text-slate-500';
  if (metric === 'annual_growth') return value > 0 ? 'text-emerald-400' : value < 0 ? 'text-red-400' : 'text-slate-300';
  return 'text-slate-300';
}

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
        <PageHeader title="Location analysis" />
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
  const asOfDates = [...new Set(scored.map((s) => s.latestDate ?? ''))];
  const sharedAsOf = asOfDates.length === 1 ? asOfDates[0] : null;

  return (
    <div className="space-y-8">
      <PageHeader title="Location analysis">
        {scored.length} locations scored on their latest readings. The investment score starts at 50 and
        adjusts for annual growth, rental yield, vacancy rate and days on market — the same formula the
        Streamlit app used.
      </PageHeader>

      {divergence && (
        <Card>
          <SectionTitle>Regional divergence</SectionTitle>
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
        </Card>
      )}

      <section>
        <SectionTitle
          right={sharedAsOf ? <span className="text-xs text-slate-500">latest readings as of {fmtDate(sharedAsOf)}</span> : undefined}
        >
          All locations, ranked
        </SectionTitle>
        <div className="overflow-x-auto rounded-xl border border-slate-800">
          <table className="w-full min-w-[760px] text-sm">
            <thead className="bg-slate-900 text-left text-xs uppercase tracking-wider text-slate-400">
              <tr>
                <th className="px-4 py-3">#</th>
                <th className="px-4 py-3">Location</th>
                <th className="px-4 py-3">Score</th>
                {TABLE_METRICS.map((m) => (
                  <th key={m} className="px-4 py-3 text-right">
                    {METRIC_LABELS[m]}
                  </th>
                ))}
                {!sharedAsOf && <th className="px-4 py-3">As of</th>}
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800">
              {scored.map((loc, i) => {
                const tone = scoreTone(loc.score, LOCATION_GOOD);
                return (
                  <tr key={loc.location} className="hover:bg-slate-900/60">
                    <td className="px-4 py-3 tabular-nums text-slate-500">{i + 1}</td>
                    <td className="whitespace-nowrap px-4 py-3 font-medium">{loc.location}</td>
                    <td className="px-4 py-3">
                      <div className="flex items-center gap-2">
                        <span className={`w-7 tabular-nums font-semibold ${tone.text}`}>{loc.score}</span>
                        <div className="h-1.5 w-16 rounded-full bg-slate-800" aria-hidden>
                          <div className={`h-1.5 rounded-full ${tone.bar}`} style={{ width: `${loc.score}%` }} />
                        </div>
                      </div>
                    </td>
                    {TABLE_METRICS.map((m) => (
                      <td key={m} className={`whitespace-nowrap px-4 py-3 text-right tabular-nums ${metricTone(m, loc.metrics[m]?.value)}`}>
                        {formatMetric(m, loc.metrics[m]?.value)}
                      </td>
                    ))}
                    {!sharedAsOf && (
                      <td className="whitespace-nowrap px-4 py-3 text-xs tabular-nums text-slate-500">
                        {fmtDate(loc.latestDate)}
                      </td>
                    )}
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </section>

      <section>
        <SectionTitle>Head to head</SectionTitle>
        <LocationCompare scored={scored} priceSeries={priceSeries} />
      </section>

      <DataStamp fetchedAt={data.fetchedAt} />
    </div>
  );
}
