"""
Writer Agent — creates the Edit Decision List (EDL) from the creative brief
and media analysis profiles.

The Writer is the screenplay/editing planner: it decides which clips to use,
in what order, with what trimming, transitions, and pacing.
"""

from __future__ import annotations

import json

from src.agents.base import BaseAgent
from src.core.models import (
    ClipSegment,
    CreativeBrief,
    EditDecisionList,
    MediaProfile,
    TransitionType,
)


SYSTEM_PROMPT = """\
You are the Writer/Editor of an AI video production house. Your job is to create
an Edit Decision List (EDL) — a precise sequence of clip segments that, when
assembled, creates a compelling video.

You will receive:
1. A creative brief (mood, pacing, structure, target duration)
2. Profiles of available media files (descriptions, durations, scenes, relevance)

Your task:
- Select the best segments from the available media
- Arrange them to match the narrative structure from the brief
- Assign transitions, timing, and pacing
- Stay within the target duration (±10 seconds)
- Prefer higher-relevance media
- For videos, specify start_time/end_time to select the best portions
- For images, specify a duration (typically 3–6 seconds for smooth, slower pacing; shorter for reels)
- Use variety — avoid over-using any single source file
- **Smooth edits**: For every segment after the first, set **transition_in** to **crossfade** unless the brief explicitly wants hard cuts. Use **transition_duration** around **0.7–1.2s** for photos and **0.5–0.9s** for video clips (longer = softer). The first segment usually uses **cut**; use **fade_to_black** / **fade_to_white** only when the brief calls for a soft opening.

Output format (JSON):
{
  "title": "...",
  "segments": [
    {
      "source_file": "/path/to/file.mp4",
      "start_time": 0.0,
      "end_time": 5.0,
      "duration": 5.0,
      "speed": 1.0,
      "transition_in": "crossfade",
      "transition_duration": 0.85,
      "text_overlay": "",
      "narrative_role": "opening",
      "reasoning": "Selected because..."
    }
  ],
  "total_duration": 30.0,
  "narrative_summary": "...",
  "audio_plan": "..."
}

transition_in must be one of: cut, crossfade, fade_to_black, fade_to_white, slide_left, slide_right

Respond ONLY with valid JSON. No markdown, no explanation.
"""


class WriterAgent(BaseAgent):
    name = "Writer"

    def run(
        self,
        brief: CreativeBrief,
        profiles: list[MediaProfile],
    ) -> EditDecisionList:
        """Create an edit decision list from the brief and media profiles."""
        self.log.info(
            f"[bold magenta]Writer[/] planning edit for \"{brief.title}\" "
            f"({len(profiles)} assets, target {brief.target_duration}s)"
        )

        media_summary = self._format_media_summary(profiles)

        user_msg = (
            f"CREATIVE BRIEF:\n"
            f"  Title: {brief.title}\n"
            f"  Concept: {brief.concept}\n"
            f"  Mood: {brief.mood.value}\n"
            f"  Pacing: {brief.pacing.value}\n"
            f"  Target duration: {brief.target_duration} seconds\n"
            f"  Structure: {json.dumps(brief.structure)}\n"
            f"  Transition preference: {brief.transition_preference.value}\n"
            f"  Style notes: {brief.style_notes}\n"
            f"  Audio direction: {brief.audio_direction}\n"
            f"  Color tone: {brief.color_tone}\n\n"
            f"AVAILABLE MEDIA:\n{media_summary}\n\n"
            f"Create the Edit Decision List now."
        )

        edl = self.llm.chat_structured(
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            response_model=EditDecisionList,
            temperature=0.5,
        )

        edl = self._validate_and_fix(edl, profiles)

        self.log.info(
            f"[bold magenta]Writer[/] EDL ready: "
            f"{len(edl.segments)} segments, ~{edl.total_duration:.1f}s"
        )
        return edl

    def _format_media_summary(self, profiles: list[MediaProfile]) -> str:
        parts = []
        for p in profiles:
            scenes_info = ""
            if p.scenes:
                scenes_info = " | Scenes: " + ", ".join(
                    f"[{s.start_time:.1f}–{s.end_time:.1f}s]" for s in p.scenes[:8]
                )

            parts.append(
                f"- File: {p.file_path}\n"
                f"  Type: {p.media_type.value} | Duration: {p.duration}s | "
                f"Resolution: {p.width}x{p.height}\n"
                f"  Tags: {', '.join(p.overall_tags[:8])}\n"
                f"  Summary: {p.summary}\n"
                f"  Relevance: {p.relevance_score}{scenes_info}\n"
            )
        return "\n".join(parts)

    def _validate_and_fix(
        self,
        edl: EditDecisionList,
        profiles: list[MediaProfile],
    ) -> EditDecisionList:
        """Validate the EDL against actual media properties and fix issues."""
        profile_map = {p.file_path: p for p in profiles}

        valid_segments = []
        running_duration = 0.0

        for seg in edl.segments:
            profile = profile_map.get(seg.source_file)
            if profile is None:
                # Try matching by filename only
                for p in profiles:
                    if p.file_path.endswith(seg.source_file.split("/")[-1]):
                        profile = p
                        seg.source_file = p.file_path
                        break

            if profile is None:
                self.log.warning(f"  Segment references unknown file: {seg.source_file}")
                continue

            if profile.media_type.value == "video":
                if seg.end_time <= 0 or seg.end_time > profile.duration:
                    seg.end_time = profile.duration
                if seg.start_time >= seg.end_time:
                    seg.start_time = 0.0
                seg.duration = (seg.end_time - seg.start_time) / seg.speed
            elif profile.media_type.value == "image":
                if seg.duration <= 0:
                    seg.duration = 4.0
                seg.start_time = 0.0
                seg.end_time = 0.0

            running_duration += seg.duration
            valid_segments.append(seg)

        edl.segments = valid_segments
        edl.total_duration = round(running_duration, 2)
        return edl
