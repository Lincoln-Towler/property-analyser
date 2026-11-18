'use client';

// Site navigation. On narrow screens the links are one row that scrolls
// sideways inside itself, so the header stays short and the page never shifts,
// and the current page is scrolled into view.

import { useEffect, useRef } from 'react';
import Link from 'next/link';
import { usePathname } from 'next/navigation';

export interface NavItem {
  href: string;
  label: string;
}

export function Nav({ items }: { items: NavItem[] }) {
  const pathname = usePathname();
  const navRef = useRef<HTMLElement>(null);
  const activeRef = useRef<HTMLAnchorElement>(null);

  // Scrolls this row only. scrollIntoView could also move the page itself.
  useEffect(() => {
    const nav = navRef.current;
    const link = activeRef.current;
    if (nav && link) nav.scrollLeft = link.offsetLeft - (nav.clientWidth - link.offsetWidth) / 2;
  }, [pathname]);

  return (
    <nav ref={navRef} aria-label="Main" className="no-scrollbar relative min-w-0 flex-1 overflow-x-auto sm:flex-none">
      <ul className="flex gap-1 text-sm">
        {items.map((item) => {
          const active = item.href === '/' ? pathname === '/' : pathname.startsWith(item.href);
          return (
            <li key={item.href}>
              <Link
                ref={active ? activeRef : undefined}
                href={item.href}
                aria-current={active ? 'page' : undefined}
                className={`block whitespace-nowrap rounded-lg px-2.5 py-1.5 transition-colors ${
                  active ? 'bg-tile font-medium text-primary' : 'text-secondary hover:text-primary'
                }`}
              >
                {item.label}
              </Link>
            </li>
          );
        })}
      </ul>
    </nav>
  );
}
