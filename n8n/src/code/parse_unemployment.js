// Seasonally adjusted unemployment rate (the headline figure ABS reports)
try {
  const rows = readAbs($input.first().json);
  const latest = rows[rows.length - 1];
  const value = Math.round(latest.value * 10) / 10;
  if (!(value > 1 && value < 20)) throw new Error('unexpected value ' + latest.value);
  return [{ json: { value, date: periodToDate(latest.period) } }];
} catch (e) {
  console.log('Unemployment skipped: ' + e.message);
  return [];
}
