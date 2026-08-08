import { describe, it, expect } from 'vitest';
import { auditPropertyData } from '../src/lib/property-audit';
import type { PropertyPoint } from '../src/lib/scoring/engine';

const p = (
  location: string,
  metric_name: string,
  date: string,
  value: number,
): PropertyPoint => ({ location, metric_name, date, value });

describe('auditPropertyData', () => {
  it('flags the extreme readings without removing them', () => {
    const rows = [
      p('Sydney NSW', 'vacancy_rate', '2026-02-15', 19.83),
      p('Sydney NSW', 'vacancy_rate', '2026-08-01', 1.5),
      p('Albany WA', 'vacancy_rate', '2026-02-17', 0),
      p('Tamworth NSW', 'annual_growth', '2026-02-15', 43.34),
    ];
    const audit = auditPropertyData(rows);
    const targets = audit.flags.map((f) => `${f.location}|${f.metric}|${f.value}`);
    expect(targets).toContain('Sydney NSW|vacancy_rate|19.83');
    expect(targets).toContain('Albany WA|vacancy_rate|0');
    expect(targets).toContain('Tamworth NSW|annual_growth|43.34');
    // the healthy August reading is not flagged
    expect(targets).not.toContain('Sydney NSW|vacancy_rate|1.5');
  });

  it('describes a zero as ambiguous rather than asserting it is broken', () => {
    const audit = auditPropertyData([p('Albany WA', 'vacancy_rate', '2026-02-17', 0)]);
    expect(audit.flags[0].note).toMatch(/genuine thin-market reading or a failed scrape/);
  });

  it('detects the double-entered February snapshot', () => {
    const rows = [
      p('Adelaide SA', 'median_price', '2026-02-15', 1020000),
      p('Adelaide SA', 'median_price', '2026-02-27', 1020000),
      p('Adelaide SA', 'median_price', '2026-08-01', 955000),
    ];
    const audit = auditPropertyData(rows);
    expect(audit.duplicates).toHaveLength(1);
    expect(audit.duplicates[0].dates).toEqual(['2026-02-15', '2026-02-27']);
  });

  it('does not treat a genuine repeat months apart as a duplicate', () => {
    const rows = [
      p('Geelong VIC', 'vacancy_rate', '2026-02-17', 1.52),
      p('Geelong VIC', 'vacancy_rate', '2026-08-01', 1.52),
    ];
    expect(auditPropertyData(rows).duplicates).toHaveLength(0);
  });

  it('reports metrics absent from the newest snapshot', () => {
    const rows = [
      p('Sale VIC', 'sales_volume', '2026-02-17', 379),
      p('Sale VIC', 'median_price', '2026-02-17', 520000),
      p('Sale VIC', 'median_price', '2026-08-01', 592000),
    ];
    const audit = auditPropertyData(rows);
    expect(audit.missingFromLatest).toEqual(['sales_volume']);
    expect(audit.latestDate).toBe('2026-08-01');
  });
});
