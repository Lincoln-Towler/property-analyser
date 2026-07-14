import type { Metadata } from 'next';
import Link from 'next/link';
import './globals.css';

export const metadata: Metadata = {
  title: 'Property Analyser',
  description: 'Australian property market conditions, scored.',
};

const NAV = [
  { href: '/', label: 'Dashboard' },
  { href: '/indicators', label: 'Indicators' },
  { href: '/cycle', label: 'Cycle' },
  { href: '/btc', label: 'BTC' },
  { href: '/audit', label: 'Data Audit' },
];

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className="min-h-screen">
        <header className="border-b border-slate-800">
          <div className="mx-auto flex max-w-5xl items-center justify-between px-4 py-4">
            <Link href="/" className="text-lg font-semibold tracking-tight">
              🏠 Property Analyser
            </Link>
            <nav className="flex gap-1 text-sm">
              {NAV.map((item) => (
                <Link
                  key={item.href}
                  href={item.href}
                  className="rounded-md px-3 py-1.5 text-slate-300 transition hover:bg-slate-800 hover:text-white"
                >
                  {item.label}
                </Link>
              ))}
            </nav>
          </div>
        </header>
        <main className="mx-auto max-w-5xl px-4 py-8">{children}</main>
        <footer className="mx-auto max-w-5xl px-4 pb-8 text-xs text-slate-500">
          One input among many — not financial advice. Data refreshes hourly from Supabase.
        </footer>
      </body>
    </html>
  );
}
