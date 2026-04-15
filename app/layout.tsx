import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Mutual Fund RAG Assistant",
  description: "A fact-based AI assistant for mutual funds",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
