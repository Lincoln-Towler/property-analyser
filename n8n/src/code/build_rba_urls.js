// The RBA publishes each month's credit figures at the END of the following month,
// so last month's page usually doesn't exist yet. Ask for last month AND the month
// before; the next step uses whichever page is actually published.
const now = new Date();
const items = [];
for (const monthsBack of [1, 2]) {
  const d = new Date(now.getFullYear(), now.getMonth() - monthsBack, 1);
  const mm = String(d.getMonth() + 1).padStart(2, '0');
  const yy = String(d.getFullYear()).slice(-2);
  items.push({ json: { url: `https://www.rba.gov.au/statistics/frequency/fin-agg/${d.getFullYear()}/fin-agg-${mm}${yy}.html` } });
}
return items;
