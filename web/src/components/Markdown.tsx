// Minimal renderer for the auto-commentary markdown (bold, italics, bullets).
// The commentary generator only emits these constructs, so a full markdown
// dependency isn't warranted.

import type { ReactNode } from 'react';

function renderInline(text: string, keyBase: string): ReactNode[] {
  const nodes: ReactNode[] = [];
  // split on **bold** first, then _italic_ within the remainder
  const parts = text.split(/(\*\*[^*]+\*\*)/g);
  parts.forEach((part, i) => {
    if (part.startsWith('**') && part.endsWith('**')) {
      nodes.push(<strong key={`${keyBase}-b${i}`}>{part.slice(2, -2)}</strong>);
    } else {
      const italics = part.split(/(_[^_]+_)/g);
      italics.forEach((seg, j) => {
        if (seg.startsWith('_') && seg.endsWith('_') && seg.length > 2) {
          nodes.push(
            <em key={`${keyBase}-i${i}-${j}`} className="text-slate-400">
              {seg.slice(1, -1)}
            </em>,
          );
        } else if (seg) {
          nodes.push(seg);
        }
      });
    }
  });
  return nodes;
}

export function Markdown({ text }: { text: string }) {
  const lines = text.split('\n');
  const blocks: ReactNode[] = [];
  let bullets: string[] = [];

  const flushBullets = (key: string) => {
    if (bullets.length) {
      blocks.push(
        <ul key={key} className="list-disc space-y-1 pl-5">
          {bullets.map((b, i) => (
            <li key={i}>{renderInline(b, `${key}-${i}`)}</li>
          ))}
        </ul>,
      );
      bullets = [];
    }
  };

  lines.forEach((line, idx) => {
    if (line.startsWith('- ')) {
      bullets.push(line.slice(2));
    } else {
      flushBullets(`ul-${idx}`);
      if (line.trim()) {
        blocks.push(<p key={`p-${idx}`}>{renderInline(line, `p-${idx}`)}</p>);
      }
    }
  });
  flushBullets('ul-end');

  return <div className="space-y-3 text-sm leading-relaxed text-slate-200">{blocks}</div>;
}
