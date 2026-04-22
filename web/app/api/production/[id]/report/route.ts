import fs from "fs/promises";
import path from "path";
import { getJobsRoot } from "@/lib/paths";

export const dynamic = "force-dynamic";

const UUID_RE = /^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i;

export async function GET(_req: Request, { params }: { params: { id: string } }) {
  const { id } = params;
  if (!UUID_RE.test(id)) {
    return Response.json({ error: "Invalid job id." }, { status: 400 });
  }

  const reportPath = path.join(getJobsRoot(), id, "production_report.json");
  try {
    const raw = await fs.readFile(reportPath, "utf8");
    return new Response(raw, {
      headers: {
        "Content-Type": "application/json",
        "Cache-Control": "no-store",
      },
    });
  } catch {
    return new Response("Not ready", { status: 404 });
  }
}
