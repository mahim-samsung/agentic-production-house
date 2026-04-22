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

  const jobDir = path.join(getJobsRoot(), id);
  const resultPath = path.join(jobDir, "result.json");

  try {
    const raw = await fs.readFile(resultPath, "utf8");
    const result = JSON.parse(raw) as { ok?: boolean; error?: string };
    if (result.ok === false) {
      return Response.json({ status: "failed", result });
    }
    if (result.ok === true) {
      return Response.json({ status: "complete", result });
    }
    return Response.json({ status: "failed", result: { ok: false, error: "Malformed result.json" } });
  } catch {
    return Response.json({ status: "running" });
  }
}
