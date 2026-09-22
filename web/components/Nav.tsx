import Link from "next/link";
import { ThemeToggle } from "./ThemeToggle";

const LINKS = [
  { href: "/chat", label: "Chat" },
  { href: "/trace", label: "Trace" },
  { href: "/eval", label: "Eval" },
];

export function Nav(): React.JSX.Element {
  return (
    <header className="border-b border-[var(--border-subtle)]">
      <div className="mx-auto flex max-w-5xl items-center justify-between gap-4 px-4 py-4 sm:px-6">
        <Link href="/" className="flex items-center gap-2 text-sm font-semibold tracking-tight">
          <span className="inline-block h-2 w-2 rounded-full bg-accent" aria-hidden="true" />
          groundwork
        </Link>
        <nav className="flex items-center gap-1 text-sm">
          {LINKS.map((link) => (
            <Link
              key={link.href}
              href={link.href}
              className="rounded-md px-3 py-1.5 text-[var(--text-secondary)] transition-colors hover:bg-[var(--bg-raised)] hover:text-[var(--text-primary)]"
            >
              {link.label}
            </Link>
          ))}
        </nav>
        <div className="flex items-center gap-3">
          <a
            href="https://github.com/mekala27-45/groundwork"
            target="_blank"
            rel="noreferrer"
            className="hidden text-sm text-[var(--text-secondary)] hover:text-[var(--text-primary)] sm:inline"
          >
            GitHub
          </a>
          <ThemeToggle />
        </div>
      </div>
    </header>
  );
}
