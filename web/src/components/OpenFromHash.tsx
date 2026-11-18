'use client';

// Opens the collapsible row a link points at, so the dashboard's
// /indicators#interest_rate lands on that indicator with its chart showing.

import { useEffect } from 'react';

export function OpenFromHash() {
  useEffect(() => {
    const id = decodeURIComponent(window.location.hash.slice(1));
    const target = id ? document.getElementById(id) : null;
    if (target instanceof HTMLDetailsElement) {
      target.open = true;
      target.scrollIntoView({ block: 'start' });
    }
  }, []);
  return null;
}
