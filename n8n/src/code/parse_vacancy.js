// SQM national rental vacancy rate: the page holds the full monthly history
// in a "var data = [...]" block; take the newest month
try {
  const html = String($input.first().json.data || '');
  const match = html.match(/var data = (\[.*?\]);/s);
  if (!match) throw new Error('could not find the data block');

  const data = JSON.parse(match[1])
    .sort((a, b) => (a.year - b.year) || (a.month - b.month));
  const latest = data[data.length - 1];
  const value = Math.round(Number(latest.vr) * 10000) / 100; // 0.0139 -> 1.39 (%)
  if (!(value > 0.1 && value < 10)) throw new Error('unexpected value ' + latest.vr);

  return [{ json: { value, date: `${latest.year}-${String(latest.month).padStart(2, '0')}-01` } }];
} catch (e) {
  console.log('Vacancy rate skipped: ' + e.message);
  return [];
}
