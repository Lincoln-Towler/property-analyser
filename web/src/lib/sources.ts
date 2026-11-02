// Source labels in the indicator tables are free text ("Roy Morgan Mortgage
// Stress Jun 2026", "Cotality final, combined capitals wk ending 9 Aug 2026"),
// so every row reads as a different source. For display, roll them up to
// the publisher.

const KNOWN_PUBLISHERS = [
  'SQM Research',
  'Roy Morgan',
  'Trading Economics',
  'Cotality',
  'CoreLogic',
  'PropTrack',
  'Domain',
  'ABS',
  'RBA',
  'BIS',
];

export function sourcePublisher(label: string): string {
  const s = label.trim();
  if (!s) return 'Unknown';
  if (s.toLowerCase() === 'history') return 'n8n feed';
  for (const name of KNOWN_PUBLISHERS) {
    if (new RegExp(`^${name}\\b`, 'i').test(s)) return name;
  }
  // Unknown publisher: keep the leading words, dropping dates/periods.
  const month = /^(jan(uary)?|feb(ruary)?|mar(ch)?|apr(il)?|may|june?|july?|aug(ust)?|sep(t(ember)?)?|oct(ober)?|nov(ember)?|dec(ember)?)$/i;
  const words = s.split(/[\s,()]+/).filter((w) => w && !/^\d/.test(w) && !month.test(w));
  return words.slice(0, 2).join(' ') || s;
}

/** { label: count } → [[publisher, count]] sorted by count, then name. */
export function groupByPublisher(sources: Record<string, number>): Array<[string, number]> {
  const totals = new Map<string, number>();
  for (const [label, n] of Object.entries(sources)) {
    const pub = sourcePublisher(label);
    totals.set(pub, (totals.get(pub) ?? 0) + n);
  }
  return [...totals.entries()].sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0]));
}
