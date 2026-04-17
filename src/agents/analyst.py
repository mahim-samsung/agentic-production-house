"""
Analyst Agent — analyzes all provided media and produces MediaProfiles.

Uses CLIP for visual understanding, Whisper for audio, and scene detection
for video segmentation. Then asks the LLM to generate high-level summaries.
"""

from __future__ import annotations

from pathlib import Path

from src.agents.base import BaseAgent
from src.core.media import analyze_media, get_media_files
from src.core.models import CreativeBrief, MediaProfile


SUMMARY_SYSTEM = """\
You are a media analyst for a video production house. Given technical analysis
data about a media file, write a concise 2-3 sentence summary describing:
1. What the visual content shows
2. The emotional tone
3. How it might be useful in a video production

Also assign a relevance_score (0.0 to 1.0) based on how well it matches the
creative brief provided.

Respond ONLY with valid JSON: {"summary": "...", "relevance_score": 0.X}
"""


class AnalystAgent(BaseAgent):
    name = "Analyst"

    def run(
        self,
        media_dir: Path | None = None,
        media_files: list[Path] | None = None,
        brief: CreativeBrief | None = None,
    ) -> list[MediaProfile]:
        """Analyze all provided media files."""
        if media_files is None:
            if media_dir is None:
                raise ValueError("Provide media_dir or media_files")
            media_files = get_media_files(Path(media_dir))

        if not media_files:
            raise ValueError("No media files found")

        self.log.info(f"[bold magenta]Analyst[/] analyzing {len(media_files)} media files")
        profiles = []

        for i, fpath in enumerate(media_files):
            self.log.info(f"  [{i+1}/{len(media_files)}] Analyzing {fpath.name}")
            try:
                profile = analyze_media(fpath)
                if brief:
                    profile = self._enrich_with_llm(profile, brief)
                profiles.append(profile)
            except Exception as e:
                self.log.warning(f"  Failed to analyze {fpath.name}: {e}")

        self.log.info(f"[bold magenta]Analyst[/] completed: {len(profiles)} profiles")
        return profiles

    def _enrich_with_llm(self, profile: MediaProfile, brief: CreativeBrief) -> MediaProfile:
        """Use the LLM to generate a summary and relevance score."""
        frame_desc = "; ".join(
            f"[{f.timestamp:.1f}s] {f.description} (emotion={f.emotion})"
            for f in profile.frames[:6]
        )
        scene_desc = "; ".join(
            f"[{s.start_time:.1f}–{s.end_time:.1f}s]"
            for s in profile.scenes[:10]
        )
        audio_desc = ""
        if profile.audio:
            audio_desc = f"Has speech: {profile.audio.has_speech}. "
            if profile.audio.transcript:
                audio_desc += f"Transcript excerpt: {profile.audio.transcript[:200]}"

        user_msg = (
            f"Media file: {profile.file_path}\n"
            f"Type: {profile.media_type.value}\n"
            f"Duration: {profile.duration}s\n"
            f"Resolution: {profile.width}x{profile.height}\n"
            f"Tags: {', '.join(profile.overall_tags)}\n"
            f"Frame descriptions: {frame_desc}\n"
            f"Scenes: {scene_desc}\n"
            f"Audio: {audio_desc}\n\n"
            f"Creative brief context:\n"
            f"Mood: {brief.mood.value}, Concept: {brief.concept}"
        )

        try:
            result = self.llm.chat_json(
                messages=[
                    {"role": "system", "content": SUMMARY_SYSTEM},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0.3,
            )
            profile.summary = result.get("summary", "")
            profile.relevance_score = float(result.get("relevance_score", 0.5))
        except Exception as e:
            self.log.warning(f"LLM enrichment failed for {profile.file_path}: {e}")

        return profile
