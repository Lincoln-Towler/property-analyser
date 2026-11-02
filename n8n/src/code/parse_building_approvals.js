// Total dwellings approved in Australia for the latest month.
// Note: the ABS API only has the unadjusted ("original") monthly series for this.
try {
  const rows = readAbs($input.first().json);
  const latest = rows[rows.length - 1];
  const value = Math.round(latest.value);
  if (!(value > 5000 && value < 40000)) throw new Error('unexpected value ' + latest.value);
  return [{ json: { value, date: periodToDate(latest.period) } }];
} catch (e) {
  console.log('Building approvals skipped: ' + e.message);
  return [];
}
