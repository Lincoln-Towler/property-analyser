// The design rules from web/DESIGN.md, as checks. Each threshold was measured
// in Lincoln's other projects; a value that fails here looks wrong or can't be
// read, so re-solve the tone rather than loosening the check.

import { describe, it, expect } from 'vitest';
import fs from 'node:fs';
import path from 'node:path';
import { COLORS, SIGNAL_BANDS } from '../src/lib/theme';

const src = path.join(__dirname, '../src');
const css = fs.readFileSync(path.join(src, 'app/globals.css'), 'utf8');

function token(name: string): string {
  const m = css.match(new RegExp(`--color-${name}:\\s*([^;]+);`));
  if (!m) throw new Error(`no --color-${name} in globals.css`);
  return m[1].trim();
}

type RGB = [number, number, number];
const hex = (h: string): RGB => [1, 3, 5].map((i) => parseInt(h.slice(i, i + 2), 16)) as RGB;
const alpha = (name: string) => Number(token(name).match(/\/\s*([\d.]+)\s*\)/)?.[1]);
const overWhite = (a: number, on: RGB): RGB => on.map((c) => 255 * a + c * (1 - a)) as RGB;

const lin = (c: number) => {
  const v = c / 255;
  return v <= 0.04045 ? v / 12.92 : ((v + 0.055) / 1.055) ** 2.4;
};
const luminance = ([r, g, b]: RGB) => 0.2126 * lin(r) + 0.7152 * lin(g) + 0.0722 * lin(b);
const contrast = (a: RGB, b: RGB) => {
  const [hi, lo] = [luminance(a), luminance(b)].sort((x, y) => y - x);
  return (hi + 0.05) / (lo + 0.05);
};
const oklabL = ([r, g, b]: RGB) => {
  const [R, G, B] = [lin(r), lin(g), lin(b)];
  const l = Math.cbrt(0.4122214708 * R + 0.5363325363 * G + 0.0514459929 * B);
  const m = Math.cbrt(0.2119034982 * R + 0.6806995451 * G + 0.1073969566 * B);
  const s = Math.cbrt(0.0883024619 * R + 0.2817188376 * G + 0.6299787005 * B);
  return 0.2104542553 * l + 0.793617785 * m - 0.0040720468 * s;
};

const canvas = hex(token('canvas'));
const surface = hex(token('surface'));
const tile = hex(token('tile'));
const ink = hex(token('ink'));
const signals = ['danger', 'reduce', 'caution', 'light-buy', 'buy', 'go', 'max-buy'];

describe('tone ladder', () => {
  it('steps evenly from canvas to surface to tile, and popups one more step up', () => {
    expect(oklabL(surface) - oklabL(canvas)).toBeCloseTo(0.052, 2);
    expect(oklabL(tile) - oklabL(surface)).toBeCloseTo(0.052, 2);
    expect(oklabL(hex(token('raised'))) - oklabL(tile)).toBeCloseTo(0.052, 2);
  });
});

describe('text contrast', () => {
  it('keeps every text step readable on the lightest surface it sits on', () => {
    expect(contrast(hex(token('primary')), tile)).toBeGreaterThan(7);
    expect(contrast(overWhite(alpha('secondary'), tile), tile)).toBeGreaterThan(4.5);
    expect(contrast(overWhite(alpha('tertiary'), tile), tile)).toBeGreaterThan(4.5);
  });

  it('keeps popup text readable on the raised tone', () => {
    const raised = hex(token('raised'));
    expect(contrast(overWhite(alpha('secondary'), raised), raised)).toBeGreaterThan(4.5);
  });

  it('lets ink carry every status fill', () => {
    for (const name of signals) expect(contrast(ink, hex(token(name))), name).toBeGreaterThanOrEqual(4.5);
  });

  it('lets each status colour work as text on a card, and on a tile inside one', () => {
    // Open and hovered rows sit on tile, with their status text on top.
    for (const name of signals) {
      expect(contrast(hex(token(name)), surface), name).toBeGreaterThanOrEqual(4.5);
      expect(contrast(hex(token(name)), tile), name).toBeGreaterThanOrEqual(4.5);
    }
    expect(contrast(hex(token('accent')), surface)).toBeGreaterThanOrEqual(4.5);
  });
});

describe('charts use the same tokens', () => {
  it('matches theme.ts to globals.css', () => {
    expect(COLORS.accent).toBe(token('accent'));
    expect(COLORS.primary).toBe(token('primary'));
    expect(COLORS.tile).toBe(token('tile'));
    expect(COLORS.surface).toBe(token('surface'));
    expect(COLORS.good).toBe(token('go'));
    expect(COLORS.warn).toBe(token('caution'));
    expect(COLORS.bad).toBe(token('danger'));
  });

  it('gives each market band a fill and chart colour that agree', () => {
    for (const band of SIGNAL_BANDS) {
      expect(band.color).toBe(token(band.fill.replace('bg-', '')));
    }
  });
});

function sourceFiles(dir: string): string[] {
  return fs.readdirSync(dir, { withFileTypes: true }).flatMap((e) => {
    const full = path.join(dir, e.name);
    if (e.isDirectory()) return sourceFiles(full);
    return /\.tsx?$/.test(e.name) ? [full] : [];
  });
}

describe('the "AI look" stays out', () => {
  const ui = [...sourceFiles(path.join(src, 'app')), ...sourceFiles(path.join(src, 'components'))];
  const styled = [...ui, path.join(src, 'lib/theme.ts'), path.join(src, 'lib/btc-view.ts')];

  // Matched inside class strings only, so ordinary words in comments and copy pass.
  const inClasses = (file: string) =>
    [...fs.readFileSync(file, 'utf8').matchAll(/className=(?:"([^"]*)"|\{`([^`]*)`\})|'((?:bg|text)-[^']*)'/g)]
      .map((m) => m[1] ?? m[2] ?? m[3])
      .join(' ');

  const banned: Array<[string, RegExp]> = [
    ['an outline on a container', /(^|\s)border(-[a-z0-9/-]+)?(\s|$)/],
    ['a ring', /(^|\s)ring(-[a-z0-9/-]+)?(\s|$)/],
    ['a rule between rows', /(^|\s)divide-/],
    ['an all-caps label', /(^|\s)uppercase(\s|$)/],
    ['tracked-out text', /(^|\s)tracking-wid/],
    ['a shadow', /(^|\s)shadow(-|\s|$)/],
    ['a gradient', /(^|\s)bg-gradient/],
    ['a raw palette colour instead of a token', /(^|\s)(bg|text)-(slate|sky|amber|red|emerald|green|orange|yellow|lime|pink|white|gray)(-|\/|\s|$)/],
    // The nav bar's frosted canvas is the one translucent surface: glass is for chrome.
    ['a tinted wash', /(^|\s)bg-(?!canvas\/)[a-z-]+\/\d/],
  ];

  it('has no hover-only hints, which never show on a phone (use InfoTip)', () => {
    // A title attribute on an HTML element (lowercase tag). Component props named title are fine.
    const hint = /<[a-z][\w-]*\b[^<>]*\stitle=/;
    const offenders = ui.filter((f) => hint.test(fs.readFileSync(f, 'utf8'))).map((f) => path.relative(src, f));
    expect(offenders).toEqual([]);
  });

  it('gives each page at most one solid status fill: its verdict', () => {
    const solid = /<Fill\b|<Notice\b|levelSignalClass\(|text-ink/g;
    for (const f of ui) {
      const rel = path.relative(src, f);
      const count = (fs.readFileSync(f, 'utf8').match(solid) ?? []).length;
      if (rel === 'components/ui.tsx') continue; // where Fill and Notice are defined
      // Pages pick their one standout; shared components never add another.
      expect(count, rel).toBeLessThanOrEqual(rel.endsWith('page.tsx') ? 1 : 0);
    }
  });

  for (const [what, pattern] of banned) {
    it(`has no ${what}`, () => {
      const offenders = styled.filter((f) => pattern.test(inClasses(f))).map((f) => path.relative(src, f));
      expect(offenders).toEqual([]);
    });
  }
});
