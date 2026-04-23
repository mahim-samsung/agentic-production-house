import { randomUUID } from "crypto";
import { spawn } from "child_process";
import fs from "fs/promises";
import path from "path";
import { getJobsRoot, getRepoRoot } from "@/lib/paths";

export const dynamic = "force-dynamic";
export const maxDuration = 300;

function safeFilename(name: string): string {
  const n = name.replace(/[^a-zA-Z0-9._-]+/g, "_").replace(/^\.+/, "").slice(0, 160);
  return n || "upload.bin";
}

export async function POST(req: Request) {
  const form = await req.formData();

  const prompt = String(form.get("prompt") ?? "").trim();
  if (!prompt) {
    return Response.json({ error: "Prompt is required." }, { status: 400 });
  }

  const platform = String(form.get("platform") ?? "youtube_long");
  const outputFilename = String(form.get("outputFilename") ?? "").trim() || `web-${randomUUID().slice(0, 8)}.mp4`;
  const skipAudio = form.get("skipAudio") === "true" || form.get("skipAudio") === "on";
  const generateMusic = form.get("generateMusic") === "true" || form.get("generateMusic") === "on";
  const keepSourceAudio =
    form.get("keepSourceAudio") === "true" || form.get("keepSourceAudio") === "on";

  const writerConstrained = form.get("writerConstrained") !== "false";
  const visionBackend = String(form.get("visionBackend") ?? "inherit").trim();
  const vlmSemanticsBackend = String(form.get("vlmSemanticsBackend") ?? "inherit").trim().toLowerCase();
  const qwen25vlModelId = String(form.get("qwen25vlModelId") ?? "").trim();
  const ollamaModel = String(form.get("ollamaModel") ?? "").trim();
  const ollamaBaseUrl = String(form.get("ollamaBaseUrl") ?? "").trim();
  const configServerPath = String(form.get("configServerPath") ?? "").trim();
  const videoMomentBackend = String(form.get("videoMomentBackend") ?? "inherit").trim().toLowerCase();
  const internvideo2Enabled = form.get("internvideo2Enabled") === "true" || form.get("internvideo2Enabled") === "on";
  const internvideo2ModelId = String(form.get("internvideo2ModelId") ?? "").trim();

  if (!["inherit", "heuristic", "internvideo2"].includes(videoMomentBackend)) {
    return Response.json({ error: "Invalid videoMomentBackend." }, { status: 400 });
  }

  if (!["inherit", "none", "qwen2_5_vl"].includes(vlmSemanticsBackend)) {
    return Response.json({ error: "Invalid vlmSemanticsBackend." }, { status: 400 });
  }

  const mediaEntries = form.getAll("media").filter((v): v is File => v instanceof File);
  if (mediaEntries.length === 0) {
    return Response.json({ error: "Upload at least one video or image." }, { status: 400 });
  }

  const repoRoot = getRepoRoot();

  let configPath: string | null = null;
  if (configServerPath) {
    const resolved = path.isAbsolute(configServerPath)
      ? path.resolve(configServerPath)
      : path.resolve(repoRoot, configServerPath);
    const normalized = path.normalize(resolved);
    try {
      const st = await fs.stat(normalized);
      if (!st.isFile()) {
        return Response.json({ error: "Config path must be a file." }, { status: 400 });
      }
      const low = normalized.toLowerCase();
      if (!low.endsWith(".yaml") && !low.endsWith(".yml")) {
        return Response.json({ error: "Config must be .yaml or .yml." }, { status: 400 });
      }
      configPath = normalized;
    } catch {
      return Response.json({ error: "Config path not found on server." }, { status: 400 });
    }
  }

  const jobsRoot = getJobsRoot();
  const jobId = randomUUID();
  const jobDir = path.join(jobsRoot, jobId);
  const inputDir = path.join(jobDir, "input");
  const musicDir = path.join(jobDir, "music");

  await fs.mkdir(inputDir, { recursive: true });

  for (const file of mediaEntries) {
    const buf = Buffer.from(await file.arrayBuffer());
    const name = safeFilename(file.name || "media.bin");
    await fs.writeFile(path.join(inputDir, name), buf);
  }

  const musicEntries = form.getAll("music").filter((v): v is File => v instanceof File && v.size > 0);
  if (musicEntries.length > 0) {
    await fs.mkdir(musicDir, { recursive: true });
    for (const file of musicEntries) {
      const buf = Buffer.from(await file.arrayBuffer());
      const name = safeFilename(file.name || "music.bin");
      await fs.writeFile(path.join(musicDir, name), buf);
    }
  }

  const env: Record<string, string> = {
    LLM_PROVIDER: "ollama",
    KEEP_SOURCE_AUDIO: keepSourceAudio ? "true" : "false",
  };
  if (writerConstrained) env.WRITER_CONSTRAINED = "true";
  else env.WRITER_CONSTRAINED = "false";
  if (visionBackend && visionBackend !== "inherit") env.VISION_BACKEND = visionBackend;
  if (vlmSemanticsBackend !== "inherit") env.VLM_SEMANTICS_BACKEND = vlmSemanticsBackend;
  if (vlmSemanticsBackend === "qwen2_5_vl" && qwen25vlModelId) env.QWEN25_VL_MODEL_ID = qwen25vlModelId;
  if (ollamaModel) env.OLLAMA_MODEL = ollamaModel;
  if (ollamaBaseUrl) env.OLLAMA_BASE_URL = ollamaBaseUrl;

  if (videoMomentBackend !== "inherit") {
    env.VIDEO_MOMENT_BACKEND = videoMomentBackend;
  }
  if (videoMomentBackend === "internvideo2") {
    env.INTERNVIDEO2_ENABLED = internvideo2Enabled ? "true" : "false";
    if (internvideo2ModelId) env.INTERNVIDEO2_MODEL_ID = internvideo2ModelId;
  }

  const job = {
    repo_root: repoRoot,
    prompt,
    platform,
    output_filename: outputFilename,
    skip_audio_enhance: skipAudio,
    generate_music: generateMusic,
    config_path: configPath,
    env,
  };

  await fs.mkdir(jobDir, { recursive: true });
  await fs.writeFile(path.join(jobDir, "job.json"), JSON.stringify(job, null, 2));

  const python = process.env.PYTHON_PATH?.trim() || "python3";
  const script = path.join(repoRoot, "scripts", "web_produce.py");
  const resultPath = path.join(jobDir, "result.json");
  const workerLog = path.join(jobDir, "worker.log");

  let logFh: Awaited<ReturnType<typeof fs.open>> | null = null;
  try {
    logFh = await fs.open(workerLog, "a");
  } catch {
    logFh = null;
  }

  const stdio: ["ignore", "ignore" | number, "ignore" | number] = logFh
    ? ["ignore", logFh.fd, logFh.fd]
    : ["ignore", "ignore", "ignore"];

  const child = spawn(python, [script, jobDir], {
    cwd: repoRoot,
    detached: true,
    stdio,
    env: { ...process.env, ...env },
  });

  const finalizeLog = async () => {
    if (logFh) {
      await logFh.close().catch(() => {});
      logFh = null;
    }
  };

  const writeExitFailure = async (detail: string) => {
    try {
      await fs.readFile(resultPath, "utf8");
      return;
    } catch {
      // no result yet
    }
    try {
      await fs.writeFile(
        resultPath,
        JSON.stringify(
          {
            ok: false,
            error: detail,
            traceback: `See ${workerLog} or run: ${python} ${script} ${jobDir}`,
          },
          null,
          2
        )
      );
    } catch {
      // ignore
    }
  };

  child.on("error", async (err) => {
    await writeExitFailure(
      `Failed to start worker (${err instanceof Error ? err.message : String(err)}). Set PYTHON_PATH to your repo venv python if needed.`
    );
    await finalizeLog();
  });

  child.on("exit", async (code, signal) => {
    await finalizeLog();
    if (code === 0 || code === null) return;
    await writeExitFailure(
      `Worker exited with code ${code}${signal ? ` (${signal})` : ""}. See ${workerLog} or run: ${python} ${script} ${jobDir}`
    );
  });

  child.unref();

  return Response.json({ jobId, outputFilename });
}
