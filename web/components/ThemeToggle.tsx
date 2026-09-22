"use client";

import { useEffect, useState } from "react";

type Theme = "dark" | "light";

const STORAGE_KEY = "groundwork-theme";

function applyTheme(theme: Theme): void {
  document.documentElement.dataset.theme = theme;
}

// Dark is this whole series' default (see globals.css's own comment), so
// the only thing worth persisting per visitor is an explicit switch away
// from it. Read once on mount rather than during render, since server
// rendered HTML (the static export's own prerendered shell) cannot know
// a visitor's earlier choice, and rendering something different on the
// client than the server sent is exactly what causes a hydration
// mismatch.
export function ThemeToggle(): React.JSX.Element {
  const [theme, setTheme] = useState<Theme>("dark");

  useEffect(() => {
    let stored: string | null = null;
    try {
      stored = window.localStorage.getItem(STORAGE_KEY);
    } catch {
      // a private window or blocked site data throws here; dark stays
      // the default for this visit, nothing more to do
    }
    if (stored === "light" || stored === "dark") {
      setTheme(stored);
      applyTheme(stored);
    }
  }, []);

  function toggle(): void {
    const next: Theme = theme === "dark" ? "light" : "dark";
    setTheme(next);
    applyTheme(next);
    try {
      window.localStorage.setItem(STORAGE_KEY, next);
    } catch {
      // per visitor convenience only, never required for the app to work
    }
  }

  return (
    <button
      type="button"
      onClick={toggle}
      className="rounded-md border border-(--color-border-subtle) px-3 py-1.5 text-sm text-(--color-text-secondary) transition-colors hover:border-(--color-border-strong) hover:text-(--color-text-primary)"
      aria-label={`Switch to ${theme === "dark" ? "light" : "dark"} theme`}
    >
      {theme === "dark" ? "Light" : "Dark"}
    </button>
  );
}
