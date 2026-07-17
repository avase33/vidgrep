import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "vidgrep · search inside video",
  description: "Natural-language multimodal video search across a Go / Rust / Python / TypeScript mesh.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
