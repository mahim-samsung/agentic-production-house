const STORAGE_KEY = "aph-admin-settings-v1";

export type AdminSettings = {
  visionBackend: "inherit" | "siglip2" | "open_clip";
  /** none | qwen2_5_vl — richer scene/frame captions (GPU + transformers). inherit = config.yaml. */
  vlmSemanticsBackend: "inherit" | "none" | "qwen2_5_vl";
  ollamaModel: string;
  ollamaBaseUrl: string;
  writerConstrained: boolean;
  /** Absolute or repo-relative path to a config.yaml on the machine running the API. */
  configServerPath: string;
  /** Use config.yaml when inherit. */
  videoMomentBackend: "inherit" | "heuristic" | "internvideo2";
  internvideo2Enabled: boolean;
  internvideo2ModelId: string;
  /** Override Qwen2.5-VL Hugging Face model id when vlmSemanticsBackend is qwen2_5_vl. */
  qwen25vlModelId: string;
};

export const defaultAdminSettings: AdminSettings = {
  visionBackend: "inherit",
  vlmSemanticsBackend: "inherit",
  ollamaModel: "",
  ollamaBaseUrl: "",
  writerConstrained: true,
  configServerPath: "",
  videoMomentBackend: "inherit",
  internvideo2Enabled: false,
  internvideo2ModelId: "",
  qwen25vlModelId: "",
};

export function loadAdminSettings(): AdminSettings {
  if (typeof window === "undefined") return { ...defaultAdminSettings };
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return { ...defaultAdminSettings };
    const parsed = JSON.parse(raw) as Partial<AdminSettings>;
    return { ...defaultAdminSettings, ...parsed };
  } catch {
    return { ...defaultAdminSettings };
  }
}

export function saveAdminSettings(s: AdminSettings): void {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(s));
}
