# Property Analyser — Design

The site uses the borderless system from `WeatherApp/DESIGN.md` (adopted there
in October 2026), adapted to the web in November 2026. The goal is the one
Lincoln described: full, filled cards and none of the "AI look", meaning no
outlines on containers, no glow, no gradients and no shouting labels.

`tests/design.test.ts` checks most of what follows. If a check fails, fix the
design, not the check.

## Principles

1. **Tone separates, lines never do.** Surfaces step up in lightness: canvas,
   then surface (cards), then tile (things inside cards), then raised (popups).
   No borders, rings, dividers, shadows or gradients. The one outline on the
   site is the keyboard focus ring.
2. **Rows, not tiles.** A value is a label/value row inside a card. Group rows
   under a heading that sits inside its card. Only use a tile when the thing is
   a unit of its own, like a location's score in Head to head or a flag
   category on `/btc`.
3. **One standout per page.** Each page gets at most one solid status fill: its
   verdict. On the dashboard that's the meter and the signal chip under it, on
   `/btc` it's the verdict card, and on `/audit` it's the health chip.
   Everything else shows status as coloured text. Warnings go inside the card
   they qualify, not in a block above it. When every status shouts, none of
   them stands out.
4. **Hints have to work on a phone.** A `title` tooltip only shows on hover, so
   phone users never see it. Use `InfoTip` (tap to read) when an explanation
   adds something the page doesn't already say, and use it sparingly. If every
   cell in a table needs the same note, put the note in the page intro instead.
   That's why stale dates on `/locations` are plain text.
5. **Sentence case and plain words.** Write "Light buy", not "LIGHT_BUY" or
   "LIGHT BUY". No letter-spaced capitals. Numbers use tabular figures so
   columns line up.

## Tokens

They're defined once in `src/app/globals.css` (`@theme`), and charts read them
from `src/lib/theme.ts`, which a test keeps in sync.

| Token | Value | Use |
|---|---|---|
| canvas | `#0b0e13` | page background |
| surface | `#171a1e` | cards |
| tile | `#24272a` | inside a card: open rows, the price marker, selects |
| raised | `#313437` | popups |
| primary | `#f2f5f7` | headings and values |
| secondary | white at 62% | labels and body text |
| tertiary | white at 48% | captions and units |
| ink | `#0b0e13` | text on a solid status fill |
| accent | `#7aa6f0` | links, chart lines, focus ring |

Each step from canvas to raised adds 0.052 OKLab lightness, so every step
looks the same size.

The status colours run from sell to maximum buy: danger `#e96660`, reduce
`#e08850`, caution `#e0a847`, light-buy `#c8bc55`, buy `#8fc060`, go
`#47c280`, max-buy `#3fd0a6`.

### Measured contrast

- Ink on every status fill is at least 6:1 (ink on danger, the lowest, is 6.01:1).
- Every status colour as text is at least 4.5:1 on a card and on a tile.
  Danger was `#de5c57` until 18 Nov 2026 and read only 4.13:1 on a tile, the
  background of an open row. It was lightened in OKLCH (L 0.64 → 0.67, same hue
  and chroma) to 4.67:1.
- Secondary and tertiary text are at least 4.5:1 on a tile.
- On raised (popups): primary is 11.44:1 and secondary 5.90:1, but tertiary is
  only 4.18:1. So popups use secondary text.
- **Tone alone can't mark a control.** A tile is only 1.16:1 against a card,
  well under the 3:1 that WCAG 1.4.11 asks of a control's edge. That's the
  same reason the glass style fails elsewhere (see the top-level `CLAUDE.md`).
  So a control has to say what it is through its text: a label, a chevron, an
  underline. The select in Head to head has a label above and a chevron. The
  `InfoTip` triggers have a dotted underline.

## Type

The system font stack (SF Pro on Apple devices). Sizes in use:

- **56px light:** the dashboard score, the page's one big number.
- **40px light:** the cycle year.
- **15px semibold:** card headings (`SectionTitle`).
- **13–14px:** rows and status text.
- **11–12px:** captions, axis labels and popups.

## Components (`src/components/`)

- **`ui.tsx`:** `Card`, `PageHeader`, `SectionTitle`, `Row`, `Bar` (a thin
  0–100 bar), `Fill` (the solid verdict chip), `Notice` (a solid block, kept
  for error states), `StatusText`, `StaleDate` and `DataStamp`. These are
  server-safe, with no hooks and no Base UI.
- **`ScoreGauge.tsx`:** the market meter, which Lincoln asked to bring back on
  18 Nov 2026. It's a half dial with the six signal bands, separated by small
  gaps. The band holding the score is lit and the rest are dimmed to 28%. It
  has tick marks, with the band boundaries labelled outside the arc so the
  needle never covers them. A wide needle has a blunt tip and a short tail. A
  faint trail runs back to the score 30 days ago. The numbers sit under the
  dial in HTML: the score, the chip, the 30-day change and the distance to the
  bands either side. The needle sweeps up from zero with a small overshoot,
  then the band lights up. It's pure CSS, off for reduced motion, and adds no
  JavaScript.
- **`InfoTip.tsx`:** a tap-to-read explanation, built on Base UI's Popover. The
  trigger has a dotted underline, and the popup sits on raised with secondary
  text and a width capped to fit a phone. It closes on Escape or an outside
  tap and returns focus to the trigger.
- **Charts (Recharts):** no axis lines, a faint grid (white at 7%), the line in
  accent or primary, and target or signal zones filled at 8% with their names
  in the gutter.
- **Nav:** the only see-through surface, canvas at 80% with blur, so content
  can be seen scrolling under it.

## Libraries

These follow the lessons of *UI Libraries for Claude Code (Ban the AI Look)*:
let a library handle behaviour and accessibility, and apply your own look on
top.

- **Base UI (`@base-ui/react`, pinned to 1.9.0)** is the foundation for
  anything that has behaviour: popovers, menus, dialogs, selects and tabs. It's
  unstyled, so it takes these tokens with nothing to undo. Don't hand-build
  focus trapping, outside-click handling or positioning.
- **Import Base UI only in the client components that use it.** If a shared
  file like `ui.tsx` imports it, every page that imports that file loads it.
  Measured: `/audit` went from 102 kB to 142 kB first-load JS when `ui.tsx`
  imported `InfoTip`.
- **Not shadcn/ui.** Its default look (borders, rings, shadows) is exactly what
  this system removes. Base UI gives the same behaviour without that look.
- **Specialist libraries were considered and skipped.** The site is read-only
  and cached hourly, so none of them has a job to do yet:
  - Sonner (toasts): there's nothing to confirm.
  - Vaul (drawers): there's nothing to open.
  - cmdk (command palette): six pages don't need one.
  - NumberFlow (animated numbers): numbers don't change while a page is open.

  Revisit them only if the site gains the problem they solve.
- **When asking Claude for UI work,** name the library and part ("use Base UI's
  Popover") and point at this file.

## Don'ts

These are all things that were removed.

- Border, ring, divide and shadow classes, and gradient backgrounds.
- Tinted washes behind content, like `bg-red-500/10`. Use a solid fill or
  coloured text.
- Solid chips on every row. One verdict per page.
- `title` tooltips.
- All-caps or letter-spaced labels.
- Raw Tailwind palette colours like `text-green-400`. Use the tokens.
