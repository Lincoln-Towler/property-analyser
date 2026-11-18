'use client';

// A tap-to-read explanation. Hover-only `title` hints never show on a phone, and
// a popover has to handle focus, Escape, outside taps and positioning, so this
// uses Base UI's unstyled Popover rather than a hand-built one (see web/DESIGN.md).

import type { ReactNode } from 'react';
import { Popover } from '@base-ui/react/popover';

export function InfoTip({ label, children, className = '' }: { label: ReactNode; children: ReactNode; className?: string }) {
  return (
    <Popover.Root>
      <Popover.Trigger className={`cursor-pointer text-left underline decoration-dotted underline-offset-[3px] ${className}`}>{label}</Popover.Trigger>
      <Popover.Portal>
        <Popover.Positioner sideOffset={6} className="z-30">
          <Popover.Popup className="max-w-[min(18rem,calc(100vw-2rem))] rounded-lg bg-raised px-3 py-2 text-xs leading-relaxed text-secondary outline-none">
            {children}
          </Popover.Popup>
        </Popover.Positioner>
      </Popover.Portal>
    </Popover.Root>
  );
}
