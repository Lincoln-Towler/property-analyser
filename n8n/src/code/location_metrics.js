// Works out vacancy, rental yield, 12-month growth and asking price for each location
// from the SQM pages fetched in the previous step, ready to save to property_data.
// Days on market and sales volume are NOT touched (they stay manual entries).

const requests = $('Locations').all(); // what was asked for, in the same order as the pages
const pages = {};                      // pages[location][page] = SQM's data list

$input.all().forEach((item, i) => {
  const req = requests[i] && requests[i].json;
  if (!req) return;
  const match = String(item.json.data || '').match(/var data = (\[.*?\]);/s);
  if (!match) return; // page failed or its layout changed: skip it
  try {
    (pages[req.location] = pages[req.location] || {})[req.page] = JSON.parse(match[1]);
  } catch (e) { /* unreadable data: skip it */ }
});

const DAY = 86400000;
const pad = n => String(n).padStart(2, '0');

// Average one column over the 13 weeks up to `end`, ignoring blanks. Smooths out week-to-week noise.
function avg13(series, field, end) {
  const to = end.getTime(), from = to - 13 * 7 * DAY;
  const vals = series
    .filter(r => { const t = Date.parse(r.date); return t > from && t <= to; })
    .map(r => Number(r[field]))
    .filter(v => v > 0);
  return vals.length >= 4 ? vals.reduce((a, b) => a + b, 0) / vals.length : null;
}

const rows = [];
const skipped = [];
const locations = [...new Set(requests.map(r => r.json.location))];

for (const location of locations) {
  const p = pages[location] || {};
  const before = rows.length;

  // 1. Vacancy rate: newest month
  const vacancy = (p['vacancy-rates'] || []).slice()
    .sort((a, b) => (a.year - b.year) || (a.month - b.month)).at(-1);
  if (vacancy) {
    const value = Math.round(Number(vacancy.vr) * 10000) / 100; // 0.0078 -> 0.78 (%)
    if (value >= 0 && value < 15) {
      rows.push({ date: `${vacancy.year}-${pad(vacancy.month)}-01`, location, metric_name: 'vacancy_rate', value, source: 'SQM vacancy rate (monthly)' });
    }
  } else skipped.push(`${location}: vacancy`);

  // 2-4. Asking prices and rents (weekly)
  const prices = (p['asking-property-prices'] || []).slice().sort((a, b) => a.date.localeCompare(b.date));
  const rents = (p['weekly-rents'] || []).slice().sort((a, b) => a.date.localeCompare(b.date));
  if (!prices.length) { skipped.push(`${location}: prices`); continue; }

  const latest = new Date(prices.at(-1).date);
  const yearAgo = new Date(latest.getTime() - 365 * DAY);
  // Use 3-bedroom houses (like-for-like); fall back to all houses for very small markets
  const field = avg13(prices, 'houses_3', latest) ? 'houses_3' : 'houses_all';
  const basis = field === 'houses_3' ? '3-bed houses' : 'all houses';
  const date = prices.at(-1).date;

  const priceNow = avg13(prices, field, latest);
  const priceThen = avg13(prices, field, yearAgo);
  const rentNow = rents.length ? avg13(rents, field, new Date(rents.at(-1).date)) : null;

  if (priceNow > 100000 && priceNow < 5000000) {
    rows.push({ date, location, metric_name: 'median_price', value: Math.round(priceNow / 1000) * 1000, source: `SQM asking price, ${basis}, 13-wk avg` });
  }
  if (priceNow && priceThen) {
    const growth = Math.round((priceNow / priceThen - 1) * 1000) / 10;
    if (growth > -40 && growth < 60) {
      rows.push({ date, location, metric_name: 'annual_growth', value: growth, source: `SQM asking price change vs year earlier, ${basis}, 13-wk avg` });
    }
  }
  if (priceNow && rentNow) {
    const gross = Math.round((rentNow * 52 / priceNow) * 10000) / 100;
    if (gross > 1 && gross < 12) {
      rows.push({ date, location, metric_name: 'rental_yield', value: gross, source: `SQM gross yield (asking rent / asking price), ${basis}, 13-wk avg` });
    }
  } else skipped.push(`${location}: yield`);

  if (rows.length === before) skipped.push(`${location}: nothing usable`);
}

if (!rows.length) {
  console.log('Location data: nothing to save', skipped);
  return [];
}
return [{ json: { rows, saved: rows.length, skipped } }];
