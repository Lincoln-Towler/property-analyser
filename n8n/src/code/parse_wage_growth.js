// Headline Wage Price Index: annual % change, excluding bonuses,
// private and public sectors, all industries, seasonally adjusted
try {
  const rows = readAbs($input.first().json);
  const latest = rows[rows.length - 1];
  const value = Math.round(latest.value * 10) / 10;
  if (!(value > -2 && value < 10)) throw new Error('unexpected value ' + latest.value);
  return [{ json: { value, date: periodToDate(latest.period) } }];
} catch (e) {
  console.log('Wage growth skipped: ' + e.message);
  return [];
}
