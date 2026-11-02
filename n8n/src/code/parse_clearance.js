// National weekend auction clearance rate from the weekly PropertyUpdate report.
// The page can contain more than one week's report, each saying
// "average clearance rate of X% ... than the Y% reported over the previous week".
// The newest is the one that no other report refers to as the previous week.
try {
  const html = String($input.first().json.data || '');
  const text = html.replace(/<[^>]+>/g, ' ').replace(/&nbsp;/g, ' ').replace(/\s+/g, ' ');

  const reports = [...text.matchAll(/average clearance rate of (?:just |only )?([\d.]+)%[^%]{0,150}?the ([\d.]+)% reported over the previous week/gi)]
    .map(m => ({ current: parseFloat(m[1]), previous: parseFloat(m[2]) }));
  let pick = reports.find(r => !reports.some(o => o !== r && o.previous === r.current));

  // Fallback if the wording changes: last "average clearance rate of X%" on the page
  if (!pick) {
    const any = [...text.matchAll(/average clearance rate of (?:just |only )?([\d.]+)%/gi)];
    if (any.length) pick = { current: parseFloat(any[any.length - 1][1]) };
  }
  if (!pick || !(pick.current > 10 && pick.current < 100)) throw new Error('no clearance rate found');

  const published = html.match(/"datePublished":"(\d{4}-\d{2}-\d{2})/);
  const date = published ? published[1] : new Date().toISOString().split('T')[0];
  return [{ json: { value: pick.current, date } }];
} catch (e) {
  console.log('Clearance rate skipped: ' + e.message);
  return [];
}
