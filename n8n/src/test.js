// Runs each n8n Code-node script against sample responses shaped like the live pages.
const fs = require('fs');
const path = require('path');
const AsyncFunction = Object.getPrototypeOf(async function () {}).constructor;
const read = f => fs.readFileSync(path.join(__dirname, 'code', f), 'utf8');
const helpers = read('abs_helpers.js');

async function run(file, items, withHelpers = false) {
  const code = (withHelpers ? helpers + '\n' : '') + read(file);
  const $input = { first: () => items[0], all: () => items };
  const logs = [];
  const origLog = console.log; console.log = m => logs.push(m);
  try { return { out: await new AsyncFunction('$input', code)($input), logs }; }
  finally { console.log = origLog; }
}

let failures = 0;
function check(name, got, expected) {
  const ok = JSON.stringify(got) === JSON.stringify(expected);
  if (!ok) failures++;
  console.log(`${ok ? 'PASS' : 'FAIL'}  ${name}${ok ? '' : `\n      got:      ${JSON.stringify(got)}\n      expected: ${JSON.stringify(expected)}`}`);
}
const today = new Date().toISOString().split('T')[0];
const adelaideToday = new Intl.DateTimeFormat('en-CA', { timeZone: 'Australia/Adelaide' }).format(new Date());

// ---------- sample responses (structure copied from the live sites on 2 Nov 2026) ----------
const cashHtml = `<table><thead><tr><th>Effective date</th><th>Change</th><th>Rate</th></tr></thead><tbody>\r\n\t\t\t<!-- year 2026 -->\r\n\t\t\t<tr>\r\n\t\t\t\t<th scope="row">30 Sep 2026</th>\r\n\t\t\t\t<td>+0.25</td>\r\n\t\t\t\t<td>4.60</td>\r\n\t\t\t\t<td class="links"><a href="#">Statement</a></td>\r\n\t\t\t</tr>\r\n\t\t\t<tr>\r\n\t\t\t\t<th scope="row">12 Aug 2026</th>\r\n\t\t\t\t<td>0.00</td>\r\n\t\t\t\t<td>4.35</td></tr></tbody></table>`;
const cashHoldHtml = cashHtml.replace('<!-- year 2026 -->', '<!-- year 2026 -->\r\n<tr>\r\n<th scope="row">4 Nov 2026</th>\r\n<td>0.00</td>\r\n<td>4.60</td>\r\n<td class="links"></td></tr>');

const finAggSep = `<html><head><title>Financial Aggregates September 2026 | RBA</title></head><body><table><tbody>\r\n<tr>\r\n\t<th scope="row">Total credit</th>\r\n\t<td>0.6</td>\r\n\t<td>0.6</td>\r\n\t<td></td>\r\n\t<td>7.2</td>\r\n\t<td>8.4</td>\r\n</tr>\r\n<tr>\r\n\t\t\t\t\t\t\t\t\t\t<th class="sub" scope="row"> &ndash; Housing</th>\r\n\t\t\t\t\t\t\t\t\t\t<td>0.4</td>\r\n\t\t\t\t\t\t\t\t\t\t<td>0.4</td>\r\n\t\t\t\t\t\t\t\t\t\t<td></td>\r\n\t\t\t\t\t\t\t\t\t\t<td>6.3</td>\r\n\t\t\t\t\t\t\t\t\t\t<td>7.0</td>\r\n</tr></tbody></table></body></html>`;
const finAgg404 = `<html><head><title>404 &ndash; Page Not Found | RBA</title></head><body>Sorry</body></html>`;
const finAggNegative = finAggSep.replace('<td>0.4</td>\r\n\t\t\t\t\t\t\t\t\t\t<td>0.4</td>', '<td>0.1</td>\r\n\t\t\t\t\t\t\t\t\t\t<td>&minus;0.2</td>');

function sdmx(timeValues, obsValues, extra = {}) {
  const observations = {};
  obsValues.forEach((v, i) => { observations[`0:0:0:0:0:0:${i}`] = [v, 0, 0, null, null]; });
  return {
    meta: { schema: 'sdmx-json-data-schema 2.0.0' },
    data: {
      dataSets: [{ structure: 0, observations }],
      structures: [{ dimensions: { observation: [
        { id: 'MEASURE', values: [{ id: 'M13' }] }, { id: 'SEX', values: [{ id: '3' }] }, { id: 'AGE', values: [{ id: '1599' }] },
        { id: 'TSEST', values: [{ id: '20', name: 'Seasonally Adjusted' }] }, { id: 'REGION', values: [{ id: 'AUS' }] }, { id: 'FREQ', values: [{ id: 'M' }] },
        { id: 'TIME_PERIOD', values: timeValues.map(t => ({ id: t })) }] } }],
      ...extra
    }
  };
}

const sqmHtml = `<script>\nvar data = [{"year":2005,"month":1,"properties":"2273042","listings":"42675","vr":"0.0188"},\n{"year":2026,"month":8,"properties":"3043211","listings":"41039","vr":"0.0135"},\n{"year":2026,"month":9,"properties":"3042828","listings":"42290","vr":"0.0139"},\n{"year":2026,"month":7,"properties":"3043411","listings":"40771","vr":"0.0134"}];\nchart(data);</script>`;

const auctionHtml = `<script type="application/ld+json">{"@type":"Article","datePublished":"2026-10-31T09:00:00+10:30","dateModified":"2026-10-31T09:00:00+10:30"}</script>
<h2>Key takeaways</h2><p>Home auction markets have returned another week of generally underwhelming results.</p>
<p>The national weekend auction market reported an average clearance rate of just 43.4% over the past week which was lower than the 46.5% reported over the previous week and again well below the 67.2% reported over the same week last year.</p>
<table><tr><td>Sydney</td><td>985</td><td>56.5%</td></tr></table><p><strong>Source:</strong> My Housing Market</p>
<p>The national weekend auction market reported an average clearance rate of 44.8% over the past week which was higher than the 43.4% reported over the previous week but again well below the 69.1% reported over the same week last year.</p>`;

(async () => {
  // Cash rate
  check('cash rate: normal page', (await run('parse_cash_rate.js', [{ json: { data: cashHtml } }])).out, [{ json: { value: 4.6, date: adelaideToday } }]);
  check('cash rate: RBA HOLD row on top (old code would save 0.00)', (await run('parse_cash_rate.js', [{ json: { data: cashHoldHtml } }])).out, [{ json: { value: 4.6, date: adelaideToday } }]);
  check('cash rate: failed request is skipped', (await run('parse_cash_rate.js', [{ json: { error: { message: 'timeout' } } }])).out, []);

  // RBA URLs
  const urls = (await run('build_rba_urls.js', [{ json: {} }])).out.map(i => i.json.url);
  console.log('      URLs built today:', urls.join('  '));
  check('rba urls: two months, newest first', urls.length === 2 && /fin-agg-\d{4}\.html$/.test(urls[0]), true);

  // Housing credit
  check('credit: Oct 404 then Sep page -> uses Sep', (await run('parse_housing_credit.js', [{ json: { data: finAgg404 } }, { json: { data: finAggSep } }])).out, [{ json: { value: 0.4, date: '2026-09-01' } }]);
  check('credit: negative figure with &minus;', (await run('parse_housing_credit.js', [{ json: { data: finAggNegative } }])).out, [{ json: { value: -0.2, date: '2026-09-01' } }]);
  check('credit: both pages missing -> skipped', (await run('parse_housing_credit.js', [{ json: { data: finAgg404 } }, { json: { error: {} } }])).out, []);

  // Unemployment (live sample: Sep 4.71834873, Aug 4.65160785; ABS lists newest first)
  const lf = sdmx(['2026-09', '2026-08'], [4.71834873, 4.65160785]);
  check('unemployment: parsed JSON', (await run('parse_unemployment.js', [{ json: lf }], true)).out, [{ json: { value: 4.7, date: '2026-09-01' } }]);
  check('unemployment: response arrived as text', (await run('parse_unemployment.js', [{ json: { data: JSON.stringify(lf) } }], true)).out, [{ json: { value: 4.7, date: '2026-09-01' } }]);
  check('unemployment: failed request skipped', (await run('parse_unemployment.js', [{ json: { error: { message: '404' } } }], true)).out, []);

  // Building approvals (live: Sep 18318 original)
  check('approvals', (await run('parse_building_approvals.js', [{ json: sdmx(['2026-09'], [18318]) }], true)).out, [{ json: { value: 18318, date: '2026-09-01' } }]);

  // Wages (live: 2026-Q2 3.2)
  check('wage growth: quarter -> date', (await run('parse_wage_growth.js', [{ json: sdmx(['2026-Q2'], [3.2]) }], true)).out, [{ json: { value: 3.2, date: '2026-06-01' } }]);

  // Population (live: thousands, as strings)
  const erp = sdmx(['2025-Q1', '2025-Q2', '2025-Q3', '2025-Q4', '2026-Q1'], ['27528.4', '27601.7', '27712.1', '27790.3', '27921.2']);
  check('population: Mar-26 vs Mar-25', (await run('parse_population.js', [{ json: erp }], true)).out, [{ json: { value: 392800, date: '2026-03-01' } }]);

  // Vacancy
  check('vacancy: newest month even if out of order', (await run('parse_vacancy.js', [{ json: { data: sqmHtml } }])).out, [{ json: { value: 1.39, date: '2026-09-01' } }]);

  // Auction
  check('clearance: picks newest of two reports', (await run('parse_clearance.js', [{ json: { data: auctionHtml } }])).out, [{ json: { value: 44.8, date: '2026-10-31' } }]);
  const reversed = auctionHtml.replace(/(<p>The national[\s\S]*?<\/p>)([\s\S]*?)(<p>The national[\s\S]*?<\/p>)$/, '$3$2$1');
  check('clearance: same answer if report order flips', (await run('parse_clearance.js', [{ json: { data: reversed } }])).out, [{ json: { value: 44.8, date: '2026-10-31' } }]);
  check('clearance: page blocked -> skipped', (await run('parse_clearance.js', [{ json: { data: '<html>Access denied</html>' } }])).out, []);

  console.log(failures ? `\n${failures} FAILED` : '\nAll tests passed');
  process.exit(failures ? 1 : 0);
})();
