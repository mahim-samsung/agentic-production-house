import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "Admin · Agentic Production House",
};

export default function AdminLayout({ children }: { children: React.ReactNode }) {
  return children;
}
