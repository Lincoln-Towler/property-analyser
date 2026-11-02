// Shared layout primitives so every page uses the same card, heading and
// badge styles. Server-safe (no hooks).

import type { ReactNode } from 'react';
import { STATUS_META, type StatusKind } from '@/lib/theme';
import { fmtDateTimeUtc } from '@/lib/format';

export function PageHeader({ title, children }: { title: string; children?: ReactNode }) {
  return (
    <header className="space-y-1">
      <h1 className="text-2xl font-semibold tracking-tight">{title}</h1>
      {children && <div className="max-w-3xl text-sm leading-relaxed text-slate-400">{children}</div>}
    </header>
  );
}

export function Card({
  children,
  className = '',
  as: Tag = 'section',
  tight = false,
}: {
  children: ReactNode;
  className?: string;
  as?: 'section' | 'div' | 'article';
  /** Uniform p-5 padding (for cards in dense grids). */
  tight?: boolean;
}) {
  return (
    <Tag className={`rounded-xl border border-slate-800 bg-slate-900/60 ${tight ? 'p-5' : 'p-5 sm:p-6'} ${className}`}>
      {children}
    </Tag>
  );
}

export function SectionTitle({ children, right }: { children: ReactNode; right?: ReactNode }) {
  return (
    <div className="mb-3 flex flex-wrap items-baseline justify-between gap-x-4 gap-y-1">
      <h2 className="text-xs font-semibold uppercase tracking-wider text-slate-400">{children}</h2>
      {right}
    </div>
  );
}

export function StatusPill({ status }: { status: StatusKind }) {
  const meta = STATUS_META[status];
  return (
    <span className={`inline-flex items-center gap-1.5 whitespace-nowrap text-xs font-medium ${meta.text}`}>
      <span className={`h-2 w-2 rounded-full ${meta.dot}`} aria-hidden />
      {meta.label}
    </span>
  );
}

export function DataStamp({ fetchedAt }: { fetchedAt: string }) {
  return <p className="text-xs text-slate-500">Data as of {fmtDateTimeUtc(fetchedAt)}</p>;
}
