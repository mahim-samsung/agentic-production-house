import fs from "fs/promises";
import path from "path";
import { getJobsRoot } from "@/lib/paths";

export const dynamic = "force-dynamic";

const UUID_RE = /^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i;

export async function GET(_req: Request, { params }: { params: { id: string } }) {
  const { id } = params;
  if (!UUID_RE.test(id)) {
    return new Response("Invalid job id", { status: 400 });
  }

  const resultPath = path.join(getJobsRoot(), id, "result.json");
  let outputFilename = "final_video.mp4";
  try {
    const r = JSON.parse(await fs.readFile(resultPath, "utf8")) as { ok?: boolean; output_path?: string };
    if (!r.ok || !r.output_path) return new Response("Not ready", { status: 404 });
    outputFilename = path.basename(r.output_path);
  } catch {
    return new Response("Not ready", { status: 404 });
  }

  const videoPath = path.join(getJobsRoot(), id, "out", outputFilename);
  try {
    await fs.access(videoPath);
  } catch {
    return new Response("Video missing", { status: 404 });
  }

  const file = await fs.readFile(videoPath);
  return new Response(file, {
    headers: {
      "Content-Type": "video/mp4",
      "Content-Disposition": `attachment; filename="${outputFilename}"`,
      "Cache-Control": "no-store",
    },
  });
}
