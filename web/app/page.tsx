import Link from "next/link";

const CARDS = [
  {
    href: "/chat",
    title: "Chat",
    description:
      "Ask a preloaded workspace anything, or upload your own PDF and get a fresh one in " +
      "under a minute. Every citation expands to its exact source passage and page number.",
  },
  {
    href: "/trace",
    title: "Trace",
    description:
      "Step through any turn: the retrieved chunks, the generated or extractive answer, " +
      "each claim's independent faithfulness label, and each citation's verification result.",
  },
  {
    href: "/eval",
    title: "Eval",
    description:
      "The retrieval metrics comparison across chunking strategy and reranking, the " +
      "faithfulness scorecard, and the injection and out of scope red team results.",
  },
];

export default function HomePage(): React.JSX.Element {
  return (
    <div className="space-y-10">
      <div className="max-w-2xl space-y-4">
        <h1 className="text-2xl font-semibold tracking-tight sm:text-3xl">groundwork</h1>
        <p className="text-[var(--text-secondary)]">
          A multi tenant platform that turns an uploaded PDF into a chatbot that answers
          questions grounded only in that document, with verifiable citations, a measured
          refusal for anything the document does not cover, and a full evaluation harness
          proving retrieval quality, chunking strategy, and answer faithfulness rather than
          asserting them. Retrieved document content is treated as data, never instructions.
        </p>
      </div>
      <div className="grid gap-4 sm:grid-cols-3">
        {CARDS.map((card) => (
          <Link
            key={card.href}
            href={card.href}
            className="rounded-lg border border-[var(--border-subtle)] bg-[var(--bg-raised)] p-5 transition-colors hover:border-[var(--color-accent)]"
          >
            <h2 className="mb-2 text-lg font-medium">{card.title}</h2>
            <p className="text-sm text-[var(--text-secondary)]">{card.description}</p>
          </Link>
        ))}
      </div>
    </div>
  );
}
