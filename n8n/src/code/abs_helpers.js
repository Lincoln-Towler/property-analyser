// ---- Helpers for the ABS Data API (same in each ABS step) ----
// Turns an ABS response into a list of { period, value }, oldest first.
function readAbs(raw) {
  if (!raw || raw.error) throw new Error('ABS request failed');
  let root = typeof raw.data === 'string' ? JSON.parse(raw.data) : raw;
  if (root.data && root.data.dataSets) root = root.data; // current ABS format wraps everything in "data"
  const dims = (root.structure || root.structures[0]).dimensions.observation;
  const timePos = dims.findIndex(d => d.id === 'TIME_PERIOD');
  const rows = [];
  for (const [key, obs] of Object.entries(root.dataSets[0].observations)) {
    const value = Number(obs[0]);
    if (obs[0] === null || Number.isNaN(value)) continue;
    rows.push({ period: dims[timePos].values[Number(key.split(':')[timePos])].id, value });
  }
  if (!rows.length) throw new Error('ABS returned no figures');
  return rows.sort((a, b) => a.period.localeCompare(b.period));
}
// "2026-09" -> "2026-09-01"   "2026-Q2" -> "2026-06-01"
function periodToDate(p) {
  const q = p.match(/^(\d{4})-Q([1-4])$/);
  return q ? `${q[1]}-${String(q[2] * 3).padStart(2, '0')}-01` : `${p.slice(0, 7)}-01`;
}
