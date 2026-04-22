export const PLATFORMS = [
  { id: "tiktok", label: "TikTok", hint: "9:16 · ~25s" },
  { id: "youtube_long", label: "YouTube · long", hint: "16:9 · ~75s" },
  { id: "youtube_shorts", label: "YouTube · Shorts", hint: "9:16 · ~35s" },
  { id: "youtube_highlights", label: "YouTube · cuts", hint: "16:9 · ~45s" },
  { id: "vimeo_cinematic", label: "Vimeo", hint: "4K-lean · ~70s" },
  { id: "instagram_reels", label: "Reels", hint: "9:16 · ~30s" },
  { id: "instagram_stories", label: "Stories", hint: "9:16 · ~20s" },
] as const;

export type PlatformId = (typeof PLATFORMS)[number]["id"];
