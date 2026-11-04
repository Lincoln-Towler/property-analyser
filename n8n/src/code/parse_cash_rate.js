// Reads the RBA cash rate page and pulls the current cash rate target
// from the newest row of the table: | date | change | cash rate |
// Saves a row on every run, even when the rate hasn't changed: that's how the site's
// audit can tell this feed is alive. The score only counts rows where the rate changed.
try {
  const html = String($input.first().json.data || '');
  const row = html.match(/<tbody[\s\S]*?<tr>\s*<th[^>]*>([^<]+)<\/th>\s*<td[^>]*>[^<]*<\/td>\s*<td[^>]*>\s*([\d.]+)\s*<\/td>/);
  if (!row) throw new Error('could not find the cash rate table');

  const value = parseFloat(row[2]);
  // Safety check: never save a number that can't be a cash rate
  if (!(value >= 0.1 && value <= 20)) throw new Error('unexpected value ' + row[2]);

  // Adelaide's date: a 7am run is still yesterday in UTC
  const date = new Intl.DateTimeFormat('en-CA', { timeZone: 'Australia/Adelaide' }).format(new Date());
  return [{ json: { value, date } }];
} catch (e) {
  console.log('Cash rate skipped: ' + e.message);
  return []; // nothing saved, the other feeds keep running
}
