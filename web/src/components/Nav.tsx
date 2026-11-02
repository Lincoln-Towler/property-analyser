'use client';

// Site navigation. Client component only for usePathname (active state).
// On narrow screens the links sit under the brand and wrap onto a second
// row, so every page stays reachable and the page never scrolls sideways.

import Link from 'next/link';
import { usePathname } from 'next/navigation';

export interface NavItem {
  href: string;
  label: string;
}

export function Nav({ items }: { items: NavItem[] }) {
  const pathname = usePathname();
  return (
    <nav aria-label="Main">
      <ul className="-ml-3 flex flex-wrap gap-1 text-sm sm:ml-0 sm:flex-nowrap">
        {items.map((item) => {
          const active = item.href === '/' ? pathname === '/' : pathname.startsWith(item.href);
          return (
            <li key={item.href}>
              <Link
                href={item.href}
                aria-current={active ? 'page' : undefined}
                className={`block whitespace-nowrap rounded-md px-3 py-1.5 transition ${
                  active
                    ? 'bg-sky-500/15 font-medium text-sky-300'
                    : 'text-slate-300 hover:bg-slate-800 hover:text-white'
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
