import type { Metadata } from "next";
import { Nav } from "@/components/Nav";
import "./globals.css";

export const metadata: Metadata = {
  title: "groundwork",
  description:
    "A multi tenant platform that turns an uploaded PDF into a chatbot that answers " +
    "questions grounded only in that document, with verifiable citations, a measured " +
    "refusal for anything the document does not cover, and a full evaluation harness.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>): React.JSX.Element {
  return (
    <html lang="en">
      <body className="min-h-screen">
        <Nav />
        <main className="mx-auto max-w-5xl px-4 py-8 sm:px-6">{children}</main>
      </body>
    </html>
  );
}
