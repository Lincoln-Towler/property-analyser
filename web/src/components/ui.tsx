// Shared layout primitives, so every page uses the same card, heading and
// status styles. Server-safe (no hooks). No outlines anywhere: tone separates
// surfaces. Status is coloured text, except each page's one verdict, which is a
// solid fill with ink on it (web/DESIGN.md).

import type { ReactNode } from 'react';
import { STATUS_META, type StatusKind } from '@/lib/theme';
import { fmtAdelaideTime, fmtMonthYear } from '@/lib/format';

export function PageHeader({ title, children }: { title: string; children?: ReactNode }) {
  return (
    <header className="space-y-1 pb-2">
      <h1 className="text-2xl font-semibold tracking-tight">{title}</h1>
      {children && <div className="max-w-3xl text-[13px] leading-relaxed text-secondary">{children}</div>}
    </header>
  );
}

export function Card({
  children,
  className = '',
  as: Tag = 'section',
}: {
  children: ReactNode;
  className?: string;
  as?: 'section' | 'div' | 'article';
}) {
  return <Tag className={`rounded-xl bg-surface p-4 sm:p-5 ${className}`}>{children}</Tag>;
}

/** A container's heading goes inside it: the container is the group. */
export function SectionTitle({ children, right }: { children: ReactNode; right?: ReactNode }) {
  return (
    <div className="mb-3 flex flex-wrap items-baseline justify-between gap-x-4 gap-y-0.5">
      <h2 className="text-[15px] font-semibold">{children}</h2>
      {right && <div className="text-xs text-tertiary">{right}</div>}
    </div>
  );
}

/** Status as a solid fill with ink on it. `fill` is a bg-* token class. */
export function Fill({ fill, children, className = '' }: { fill: string; children: ReactNode; className?: string }) {
  return (
    <span className={`inline-block whitespace-nowrap rounded-lg px-2.5 py-0.5 text-[13px] font-medium text-ink ${fill} ${className}`}>
      {children}
    </span>
  );
}

/** A status message as a solid block. */
export function Notice({ status, children }: { status: StatusKind; children: ReactNode }) {
  return <p className={`rounded-xl px-4 py-3 text-sm font-medium text-ink ${STATUS_META[status].fill}`}>{children}</p>;
}

/** Status as coloured text, for inside a row. */
export function StatusText({ status }: { status: StatusKind }) {
  const meta = STATUS_META[status];
  return <span className={`whitespace-nowrap text-[13px] font-medium ${meta.text}`}>{meta.label}</span>;
}

/** One label/value row. Use inside a <dl>; spacing separates rows, not rules. */
export function Row({ label, children }: { label: string; children: ReactNode }) {
  return (
    <div className="flex items-baseline justify-between gap-4 py-1">
      <dt className="shrink-0 text-[13px] text-tertiary">{label}</dt>
      <dd className="min-w-0 text-right text-sm">{children}</dd>
    </div>
  );
}

/** A thin 0–100 bar. `track` is one tone step up from whatever it sits on. */
export function Bar({ value, fill, track = 'bg-tile' }: { value: number; fill: string; track?: string }) {
  return (
    <div className={`h-1.5 rounded-full ${track}`} aria-hidden>
      <div className={`h-1.5 rounded-full ${fill}`} style={{ width: `${Math.max(0, Math.min(100, value))}%` }} />
    </div>
  );
}

/** A reading's month, shown only when it is older than the location's
 *  newest snapshot — otherwise every cell would repeat the same date. Plain
 *  text on purpose: the /locations intro explains it, and a popover in every
 *  cell would underline half the table. */
export function StaleDate({ date, latest }: { date: string | undefined; latest: string | null }) {
  if (!date || date === latest) return null;
  return (
    <time dateTime={date} className="ml-2 whitespace-nowrap text-xs text-caution">
      ({fmtMonthYear(date)})
    </time>
  );
}

export function DataStamp({ fetchedAt }: { fetchedAt: string }) {
  return <p className="px-1 pt-1 text-xs text-tertiary">Data as of {fmtAdelaideTime(fetchedAt)}</p>;
}
