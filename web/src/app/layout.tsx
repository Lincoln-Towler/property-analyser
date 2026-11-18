import type { Metadata } from 'next';
import Link from 'next/link';
import { Nav, type NavItem } from '@/components/Nav';
import './globals.css';

export const metadata: Metadata = {
  title: 'Property Analyser',
  description: 'Australian property market conditions, scored.',
};

const NAV: NavItem[] = [
  { href: '/', label: 'Dashboard' },
  { href: '/indicators', label: 'Indicators' },
  { href: '/locations', label: 'Locations' },
  { href: '/cycle', label: 'Cycle' },
  { href: '/btc', label: 'BTC' },
  { href: '/audit', label: 'Audit' },
];

function BrandMark() {
  return (
    <svg viewBox="0 0 24 24" className="h-6 w-6 text-accent" fill="none" aria-hidden>
      <path d="M3 11.5 12 4l9 7.5" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
      <path d="M5.5 10v9.5h13V10" stroke="currentColor" strokeWidth="2" strokeLinejoin="round" />
      <path d="M9 19.5v-4l2.2-2 1.8 1.5 2-2.5" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className="flex min-h-screen flex-col">
        <header className="sticky top-0 z-20 bg-canvas/80 backdrop-blur-md">
          <div className="mx-auto flex max-w-5xl items-center gap-3 px-4 py-2.5 sm:justify-between">
            <Link href="/" aria-label="Property Analyser" className="flex shrink-0 items-center gap-2 text-[15px] font-semibold">
              <BrandMark />
              <span className="hidden sm:inline">Property Analyser</span>
            </Link>
            <Nav items={NAV} />
          </div>
        </header>
        <main className="mx-auto w-full max-w-5xl flex-1 px-4 pb-6 pt-2">{children}</main>
        <footer>
          <div className="mx-auto max-w-5xl px-5 pb-6 text-xs text-tertiary">
            One input among many — not financial advice. Data refreshes hourly from Supabase.
          </div>
        </footer>
      </body>
    </html>
  );
}
