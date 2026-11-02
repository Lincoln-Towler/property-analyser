// Population growth over the last 12 months (number of people).
// ABS gives the population in thousands each quarter, so compare the latest
// quarter with the same quarter a year earlier.
try {
  const rows = readAbs($input.first().json);
  const latest = rows[rows.length - 1];
  const [year, quarter] = latest.period.split('-');
  const yearAgo = rows.find(r => r.period === `${Number(year) - 1}-${quarter}`);
  if (!yearAgo) throw new Error('no figure for a year earlier');

  const value = Math.round((latest.value - yearAgo.value) * 1000);
  if (!(value > 50000 && value < 1500000)) throw new Error('unexpected value ' + value);
  return [{ json: { value, date: periodToDate(latest.period) } }];
} catch (e) {
  console.log('Population growth skipped: ' + e.message);
  return [];
}
