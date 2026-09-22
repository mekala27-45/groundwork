type Status = "pass" | "fail" | "warn" | "neutral";

const STATUS_CLASSES: Record<Status, string> = {
  pass: "status-pass border-[var(--color-success)]/40 bg-[var(--color-success)]/10",
  fail: "status-fail border-[var(--color-danger)]/40 bg-[var(--color-danger)]/10",
  warn: "status-warn border-[var(--color-warning)]/40 bg-[var(--color-warning)]/10",
  neutral: "text-[var(--text-secondary)] border-[var(--border-subtle)] bg-[var(--bg-raised)]",
};

export function StatusPill({
  status,
  children,
}: {
  status: Status;
  children: React.ReactNode;
}): React.JSX.Element {
  return (
    <span
      className={`inline-flex items-center rounded-full border px-2 py-0.5 text-xs font-medium ${STATUS_CLASSES[status]}`}
    >
      {children}
    </span>
  );
}
