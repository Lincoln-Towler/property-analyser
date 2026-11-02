// Reads the RBA Financial Aggregates page and pulls monthly HOUSING credit growth
// (the newest month in the "Housing" row). Pages that aren't published yet are skipped.
const MONTHS = ['January', 'February', 'March', 'April', 'May', 'June', 'July',
  'August', 'September', 'October', 'November', 'December'];

for (const item of $input.all()) {
  // RBA writes negative numbers with a special minus sign; turn it into a normal "-"
  const html = String(item.json.data || '').replace(/&minus;|&#8722;|−/g, '-');

  const title = html.match(/<title>\s*Financial Aggregates\s+([A-Za-z]+)\s+(\d{4})/);
  const row = html.match(/Housing\s*<\/th>\s*<td[^>]*>\s*(-?[\d.]+)\s*<\/td>\s*<td[^>]*>\s*(-?[\d.]+)\s*<\/td>/);
  if (!title || !row) continue; // e.g. a "page not found" for a month not published yet

  const month = MONTHS.indexOf(title[1]);
  const value = parseFloat(row[2]);
  if (month < 0 || !(value > -5 && value < 5)) continue;

  return [{ json: { value, date: `${title[2]}-${String(month + 1).padStart(2, '0')}-01` } }];
}

console.log('Housing credit skipped: no published RBA page found');
return [];
