import { getSiteData } from '@/lib/data';
import { calculateRegionalDivergence } from '@/lib/scoring/engine';
import {
  scoreAllLocations,
  formatMetric,
  isAskingPriceReading,
  METRIC_LABELS,
} from '@/lib/scoring/location';
import { LocationCompare, type PriceSeriesPoint } from '@/components/LocationCompare';
import { SetupNotice } from '@/components/SetupNotice';
import { Bar, Card, DataStamp, Notice, PageHeader, SectionTitle, StaleDate } from '@/components/ui';
import { LOCATION_GOOD, scoreTone } from '@/lib/theme';
import { fmtDate } from '@/lib/format';

export const revalidate = 3600;

function metricTone(metric: string, value: number | undefined): string {
  if (value === undefined) return 'text-tertiary';
  if (metric === 'annual_growth') return value > 0 ? 'text-go' : value < 0 ? 'text-danger' : 'text-secondary';
  return 'text-primary';
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
      <div className="space-y-2.5">
        <PageHeader title="Locations" />
        {queryFailed ? (
          <>
            <Notice status="danger">The property_data query failed. This isn&apos;t an empty table.</Notice>
            <Card as="div">
              <p className="mb-3 font-mono text-xs text-danger">{queryFailed}</p>
              <p className="text-sm text-secondary">
                Most likely Row Level Security: the site reads with the anon key, which needs a SELECT policy on{' '}
                <code className="rounded bg-tile px-1">property_data</code>. Migration{' '}
                <code className="rounded bg-tile px-1">0004_enable_rls.sql</code> creates one. Check it was applied, then confirm in
                Supabase → Authentication → Policies.
              </p>
            </Card>
          </>
        ) : (
          <Card as="div">
            <p className="text-sm text-secondary">
              No property data in the last 12 months. Add rows to <code className="rounded bg-tile px-1">property_data</code> (location,
              metric_name, value, date) in Supabase to populate this page.
            </p>
          </Card>
        )}
      </div>
    );
  }

  const scored = scoreAllLocations(rows);
  const divergence = calculateRegionalDivergence(rows, new Date());

  // Wide-format asking price series for the comparison chart (stored as median_price)
  const priceRows = rows.filter(isAskingPriceReading);
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
    <div className="space-y-2.5">
      <PageHeader title="Locations">
        {scored.length} locations, scored on their latest readings. The score starts at 50 and adjusts for growth, yield, vacancy and days
        on market. Asking price is SQM&apos;s 13-week average for 3-bed houses, not a sold median, and growth is measured on it. Days on
        market is entered by hand, so it can be older; any reading older than the newest shows its month.
      </PageHeader>

      {divergence && (
        <Card>
          <SectionTitle>Regional divergence</SectionTitle>
          <p className="text-sm text-secondary">
            A spread of <strong className="font-semibold text-primary">{divergence.divergence}</strong> points between the strongest and
            weakest markets ({best.location} {best.score}, {worst.location} {worst.score}).{' '}
            {divergence.recommendation === 'selective'
              ? 'Divergence is high: a selective market, where location choice matters more than timing.'
              : 'Divergence is low: markets are moving together, so a broad approach is reasonable.'}
          </p>
        </Card>
      )}

      <Card>
        <SectionTitle right={sharedAsOf ? `Latest readings as of ${fmtDate(sharedAsOf)}` : undefined}>All locations, ranked</SectionTitle>
        <div className="no-scrollbar -mx-2 overflow-x-auto">
          <table className="w-full min-w-[720px] text-sm">
            <thead className="text-left text-[11px] font-medium text-tertiary">
              <tr>
                <th className="px-2 pb-1.5 font-medium">#</th>
                <th className="px-2 pb-1.5 font-medium">Location</th>
                <th className="px-2 pb-1.5 font-medium">Score</th>
                {TABLE_METRICS.map((m) => (
                  <th key={m} className="px-2 pb-1.5 text-right font-medium">
                    {METRIC_LABELS[m]}
                  </th>
                ))}
                {!sharedAsOf && <th className="px-2 pb-1.5 font-medium">As of</th>}
              </tr>
            </thead>
            <tbody>
              {scored.map((loc, i) => {
                const tone = scoreTone(loc.score, LOCATION_GOOD);
                return (
                  <tr key={loc.location} className="transition-colors hover:bg-tile">
                    <td className="rounded-l-lg px-2 py-1.5 text-tertiary">{i + 1}</td>
                    <td className="whitespace-nowrap px-2 py-1.5">{loc.location}</td>
                    <td className="px-2 py-1.5">
                      <div className="flex items-center gap-2">
                        <span className={`w-7 font-medium ${tone.text}`}>{loc.score}</span>
                        <div className="w-16">
                          <Bar value={loc.score} fill={tone.bar} />
                        </div>
                      </div>
                    </td>
                    {TABLE_METRICS.map((m) => (
                      <td key={m} className={`whitespace-nowrap px-2 py-1.5 text-right last:rounded-r-lg ${metricTone(m, loc.metrics[m]?.value)}`}>
                        {formatMetric(m, loc.metrics[m]?.value)}
                        <StaleDate date={loc.metrics[m]?.date} latest={loc.latestDate} />
                      </td>
                    ))}
                    {!sharedAsOf && (
                      <td className="whitespace-nowrap rounded-r-lg px-2 py-1.5 text-xs text-tertiary">{fmtDate(loc.latestDate)}</td>
                    )}
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </Card>

      <Card>
        <SectionTitle>Head to head</SectionTitle>
        <LocationCompare scored={scored} priceSeries={priceSeries} />
      </Card>

      <DataStamp fetchedAt={data.fetchedAt} />
    </div>
  );
}
