import path from "path";

/** Repository root (parent of /web when running from web/). */
export function getRepoRoot(): string {
  const fromEnv = process.env.REPO_ROOT?.trim();
  if (fromEnv) return path.resolve(fromEnv);
  return path.resolve(process.cwd(), "..");
}

export function getJobsRoot(): string {
  return path.join(getRepoRoot(), ".tmp", "web-jobs");
}
