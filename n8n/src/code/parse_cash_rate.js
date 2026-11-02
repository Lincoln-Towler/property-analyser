// Reads the RBA cash rate page and pulls the current cash rate target
// from the newest row of the table: | date | change | cash rate |
try {
  const html = String($input.first().json.data || '');
  const row = html.match(/<tbody[\s\S]*?<tr>\s*<th[^>]*>([^<]+)<\/th>\s*<td[^>]*>[^<]*<\/td>\s*<td[^>]*>\s*([\d.]+)\s*<\/td>/);
  if (!row) throw new Error('could not find the cash rate table');

  const value = parseFloat(row[2]);
  // Safety check: never save a number that can't be a cash rate
  if (!(value >= 0.1 && value <= 20)) throw new Error('unexpected value ' + row[2]);

  return [{ json: { value, date: new Date().toISOString().split('T')[0] } }];
} catch (e) {
  console.log('Cash rate skipped: ' + e.message);
  return []; // nothing saved, the other feeds keep running
}
