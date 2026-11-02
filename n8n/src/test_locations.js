// Tests the Location Data workflow's two Code nodes with pages shaped like SQM's live ones.
const fs = require('fs');
const path = require('path');
const AsyncFunction = Object.getPrototypeOf(async function () {}).constructor;
const read = f => fs.readFileSync(path.join(__dirname, 'code', f), 'utf8');

async function runNode(file, inputItems, nodeOutputs = {}) {
  const $input = { first: () => inputItems[0], all: () => inputItems };
  const $ = name => ({ all: () => nodeOutputs[name] });
  return new AsyncFunction('$input', '$', read(file))($input, $);
}

let failures = 0;
const check = (name, ok, detail = '') => { if (!ok) failures++; console.log(`${ok ? 'PASS' : 'FAIL'}  ${name}${ok ? '' : '  ' + detail}`); };

// ---- build SQM-style pages ----
const iso = d => d.toISOString().slice(0, 10);
function weekly(endDate, weeks, fn) {
  const out = [];
  for (let w = weeks - 1; w >= 0; w--) {
    const d = new Date(Date.parse(endDate) - w * 7 * 86400000);
    out.push({ date: iso(d), ...fn(w) });
  }
  return out;
}
const page = data => `<html><script>\nvar data = ${JSON.stringify(data)};\nrender(data);</script></html>`;

// Year-ago window: 3-bed $600k; latest 13 weeks: $660k -> growth +10.0%. Rent $550/wk -> yield 550*52/660000 = 4.33%
const pricesNormal = weekly('2026-10-23', 70, w => ({ houses_all: w < 13 ? 760000 : 700000, houses_3: w < 13 ? 660000 : (w >= 52 ? 600000 : 640000), units_all: 0, units_2: 0, combined: 0 }));
const rentsNormal = weekly('2026-10-22', 70, () => ({ houses_all: 560, houses_3: 550, units_all: 0, units_2: 0, combined: 0 }));
const vacancyNormal = [{ year: 2026, month: 7, vr: '0.0090' }, { year: 2026, month: 9, vr: '0.0078' }, { year: 2026, month: 8, vr: '0.0072' }];

// Small market: no 3-bed figures -> falls back to all houses ($500k now, $400k a year ago -> +25%)
const pricesSmall = weekly('2026-10-23', 70, w => ({ houses_all: w < 13 ? 500000 : (w >= 52 ? 400000 : 450000), houses_3: 0 }));
const rentsSmall = weekly('2026-10-22', 70, () => ({ houses_all: 480, houses_3: 0 }));

(async () => {
  // Locations node
  const reqs = await runNode('locations_list.js', [{ json: {} }]);
  check('locations: 11 places x 3 pages = 33 requests', reqs.length === 33, reqs.length);
  check('locations: city URL format', reqs[0].json.url === 'https://sqmresearch.com.au/property/vacancy-rates?region=nsw-Sydney&type=c&t=1', reqs[0].json.url);
  check('locations: postcode URL format', reqs.some(r => r.json.url === 'https://sqmresearch.com.au/property/asking-property-prices?postcode=3690&t=1'));

  // Build fetched pages, in the same order as the requests
  const fetched = reqs.map(r => {
    const { location, page: p } = r.json;
    if (location === 'Sale VIC') return { json: { data: page(p === 'vacancy-rates' ? vacancyNormal : p === 'weekly-rents' ? rentsSmall : pricesSmall) } };
    if (location === 'Geelong VIC' && p === 'vacancy-rates') return { json: { error: { message: 'timeout' } } }; // one failed page
    if (location === 'Albany WA') return { json: { data: '<html>Access denied</html>' } };                       // whole location blocked
    return { json: { data: page(p === 'vacancy-rates' ? vacancyNormal : p === 'weekly-rents' ? rentsNormal : pricesNormal) } };
  });

  const out = await runNode('location_metrics.js', fetched, { Locations: reqs });
  const rows = out[0].json.rows;
  const get = (loc, m) => rows.find(r => r.location === loc && r.metric_name === m);

  const w = m => get('Wodonga VIC', m);
  check('vacancy: newest month, as %', w('vacancy_rate') && w('vacancy_rate').value === 0.78 && w('vacancy_rate').date === '2026-09-01', JSON.stringify(w('vacancy_rate')));
  check('price: 13-wk avg of 3-bed, rounded to $1k', w('median_price') && w('median_price').value === 660000 && w('median_price').date === '2026-10-23', JSON.stringify(w('median_price')));
  check('growth: 13-wk avg vs year earlier = +10.0%', w('annual_growth') && w('annual_growth').value === 10, JSON.stringify(w('annual_growth')));
  check('yield: rent x 52 / price = 4.33%', w('rental_yield') && w('rental_yield').value === 4.33, JSON.stringify(w('rental_yield')));
  check('small market falls back to all houses (+25%)', get('Sale VIC', 'annual_growth') && get('Sale VIC', 'annual_growth').value === 25 && /all houses/.test(get('Sale VIC', 'annual_growth').source), JSON.stringify(get('Sale VIC', 'annual_growth')));
  check('one failed page only skips that metric', !get('Geelong VIC', 'vacancy_rate') && !!get('Geelong VIC', 'rental_yield'));
  check('blocked location writes nothing for it', !rows.some(r => r.location === 'Albany WA') && out[0].json.skipped.some(s => s.startsWith('Albany WA')));
  check('never touches days on market or sales volume', !rows.some(r => ['days_on_market', 'sales_volume'].includes(r.metric_name)));
  check('one row per location+metric (safe to upsert)', new Set(rows.map(r => r.location + r.metric_name)).size === rows.length);
  console.log(`      rows to save: ${rows.length}   skipped: ${out[0].json.skipped.join(' | ')}`);

  // Everything blocked -> nothing saved, Save step doesn't run
  const none = await runNode('location_metrics.js', reqs.map(() => ({ json: { error: {} } })), { Locations: reqs });
  check('all pages failed -> no output', Array.isArray(none) && none.length === 0);

  console.log(failures ? `\n${failures} FAILED` : '\nAll tests passed');
  process.exit(failures ? 1 : 0);
})();
