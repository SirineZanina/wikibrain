import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "WikiBrain — Policy Assistant",
  description: "Answers grounded in Aster Dynamics policy documents.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
