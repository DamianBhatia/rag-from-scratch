import type { Metadata } from "next";

import "./globals.css";

export const metadata: Metadata = {
  title: "ReAct Chat",
  description: "A minimal local chat interface for the repository ReAct agent.",
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
