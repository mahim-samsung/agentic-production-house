"""
Writer Agent — creates the Edit Decision List (EDL) from the creative brief
and media analysis profiles.

Default path is **constrained editing**: the LLM may only choose from
precomputed segment candidates (real paths + valid time windows), which
reduces hallucinated clips and impossible trims. A deterministic greedy
fallback fills the target duration if the model returns nothing usable.
"""

from __future__ import annotations

import json

from src.agents.base import BaseAgent
from src.core.candidates import build_segment_candidates
from src.core.models import (
    ClipSegment,
    ConstrainedEditPlan,
    ConstrainedPick,
    CreativeBrief,
    EditDecisionList,
    MediaProfile,
    MediaType,
    SegmentCandidate,
    TransitionType,
)
from src.utils.config import get as cfg


SYSTEM_PROMPT_LEGACY = """\
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


SYSTEM_PROMPT_CONSTRAINED = """\
You are the Writer/Editor of an AI video production house.

You MUST plan the edit using ONLY the provided **segment candidates**. Each candidate
has an integer **id** (0..N-1), a **source_file**, valid **start_time/end_time** for
video (or **image_duration** for images), and a short **hint**.

Rules:
- Output ONLY valid JSON with keys: title, picks, narrative_summary, audio_plan.
- **picks** is an array of objects. Each object MUST include:
  - **candidate_index**: integer id from the candidate list (exactly one of the listed ids)
  - **speed**: number (use 1.0 unless the brief clearly needs slow/fast motion)
  - **transition_in**: one of cut, crossfade, fade_to_black, fade_to_white, slide_left, slide_right
  - **transition_duration**: number (seconds)
  - **narrative_role**: short string (e.g. opening, build, climax)
  - **text_overlay**: string (often empty)
  - **reasoning**: one short sentence tied to the hint/score
- Do NOT invent file paths or time ranges. Do NOT use candidate ids that are not listed.
- Order **picks** for story flow. Aim for total duration within ±10s of the brief target
  (sum video as (end_time-start_time)/speed, sum images as image_duration/speed).
- Prefer higher **moment_score** when choosing among candidates.
- First pick: **transition_in** = cut unless the brief asks for a soft open.
- After the first pick: prefer **crossfade** unless the brief wants hard cuts.

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

        if bool(cfg("writer.constrained", True)):
            return self._run_constrained(brief, profiles)

        return self._run_legacy(brief, profiles)

    def _run_legacy(self, brief: CreativeBrief, profiles: list[MediaProfile]) -> EditDecisionList:
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
                {"role": "system", "content": SYSTEM_PROMPT_LEGACY},
                {"role": "user", "content": user_msg},
            ],
            response_model=EditDecisionList,
            temperature=0.5,
        )
        edl = self._validate_and_fix(edl, profiles)
        self._log_edl(edl)
        return edl

    def _run_constrained(self, brief: CreativeBrief, profiles: list[MediaProfile]) -> EditDecisionList:
        candidates = build_segment_candidates(profiles)
        if not candidates:
            raise ValueError(
                "No segment candidates produced (need at least one video or image asset)."
            )

        cand_lines = self._format_candidate_list(candidates)
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
            f"SEGMENT CANDIDATES (use only candidate_index from this list):\n{cand_lines}\n\n"
            f"Return JSON: title, picks, narrative_summary, audio_plan."
        )

        try:
            plan = self.llm.chat_structured(
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT_CONSTRAINED},
                    {"role": "user", "content": user_msg},
                ],
                response_model=ConstrainedEditPlan,
                temperature=0.35,
                inject_schema=True,
            )
        except Exception as e:
            self.log.warning(f"Constrained LLM plan failed ({e}); using greedy fallback.")
            plan = None

        if plan and plan.picks:
            edl = self._plan_to_edl(plan, candidates, brief)
        else:
            edl = self._greedy_edl(brief, candidates)

        edl = self._validate_and_fix(edl, profiles)
        self._log_edl(edl)
        return edl

    def _format_candidate_list(self, candidates: list[SegmentCandidate]) -> str:
        lines = []
        for c in candidates:
            if c.media_type == MediaType.IMAGE:
                lines.append(
                    f"id={c.id} | image | file={c.source_file} | "
                    f"image_duration={c.image_duration:.2f}s | moment_score={c.moment_score:.3f} | hint={c.hint}"
                )
            else:
                lines.append(
                    f"id={c.id} | video | file={c.source_file} | "
                    f"start={c.start_time:.3f}s end={c.end_time:.3f}s | "
                    f"moment_score={c.moment_score:.3f} | hint={c.hint}"
                )
        return "\n".join(lines)

    def _plan_to_edl(
        self,
        plan: ConstrainedEditPlan,
        candidates: list[SegmentCandidate],
        brief: CreativeBrief,
    ) -> EditDecisionList:
        by_index = {c.id: c for c in candidates}
        segs: list[ClipSegment] = []
        for i, pick in enumerate(plan.picks):
            c = by_index.get(pick.candidate_index)
            if c is None:
                self.log.warning(f"  Ignoring invalid candidate_index={pick.candidate_index}")
                continue
            seg = self._candidate_to_clip(c, pick, i)
            segs.append(seg)

        if not segs:
            return self._greedy_edl(brief, candidates)

        total = sum(s.duration for s in segs)
        return EditDecisionList(
            title=plan.title or brief.title,
            segments=segs,
            total_duration=round(total, 2),
            narrative_summary=plan.narrative_summary,
            audio_plan=plan.audio_plan,
        )

    def _candidate_to_clip(self, c: SegmentCandidate, pick: ConstrainedPick, seg_index: int) -> ClipSegment:
        default_td = float(cfg("video.transition_duration", 0.85))
        t_in = pick.transition_in
        t_dur = pick.transition_duration if pick.transition_duration > 0 else default_td
        if seg_index == 0:
            t_in = TransitionType.CUT
            t_dur = min(t_dur, 0.5)

        if c.media_type == MediaType.IMAGE:
            dur = max(0.5, c.image_duration) / max(pick.speed, 0.25)
            return ClipSegment(
                source_file=c.source_file,
                start_time=0.0,
                end_time=0.0,
                duration=round(dur, 3),
                speed=pick.speed,
                transition_in=t_in,
                transition_duration=t_dur,
                text_overlay=pick.text_overlay,
                narrative_role=pick.narrative_role,
                reasoning=pick.reasoning or f"Candidate id={c.id} (image).",
            )

        span = max(0.01, c.end_time - c.start_time)
        dur = span / max(pick.speed, 0.25)
        return ClipSegment(
            source_file=c.source_file,
            start_time=c.start_time,
            end_time=c.end_time,
            duration=round(dur, 3),
            speed=pick.speed,
            transition_in=t_in,
            transition_duration=t_dur,
            text_overlay=pick.text_overlay,
            narrative_role=pick.narrative_role,
            reasoning=pick.reasoning or f"Candidate id={c.id} (video).",
        )

    def _greedy_edl(self, brief: CreativeBrief, candidates: list[SegmentCandidate]) -> EditDecisionList:
        """Deterministic fill toward target duration using moment scores."""
        target = float(brief.target_duration)
        default_td = float(cfg("video.transition_duration", 0.85))
        prefer = brief.transition_preference
        max_segments = min(80, max(12, int(target / 1.5)))

        ranked = sorted(candidates, key=lambda x: -x.moment_score)
        segs: list[ClipSegment] = []
        total = 0.0
        rounds = 0
        while total < target - 0.25 and rounds < 8 and len(segs) < max_segments:
            for c in ranked:
                if total >= target - 0.25 or len(segs) >= max_segments:
                    break
                if c.media_type == MediaType.IMAGE:
                    dur = max(0.5, c.image_duration)
                else:
                    dur = max(0.01, c.end_time - c.start_time)
                idx = len(segs)
                t_in = TransitionType.CUT if idx == 0 else prefer
                t_dur = 0.35 if idx == 0 else default_td
                segs.append(
                    ClipSegment(
                        source_file=c.source_file,
                        start_time=c.start_time,
                        end_time=c.end_time,
                        duration=round(dur, 3),
                        speed=1.0,
                        transition_in=t_in,
                        transition_duration=t_dur,
                        narrative_role="auto",
                        reasoning=f"Greedy fallback: candidate id={c.id}, score={c.moment_score:.3f}",
                    )
                )
                total += dur
            rounds += 1

        return EditDecisionList(
            title=brief.title,
            segments=segs,
            total_duration=round(total, 2),
            narrative_summary="Auto-assembled from top moment-scored candidates (LLM plan missing or invalid).",
            audio_plan=brief.audio_direction or "",
        )

    def _log_edl(self, edl: EditDecisionList) -> None:
        self.log.info(
            f"[bold magenta]Writer[/] EDL ready: "
            f"{len(edl.segments)} segments, ~{edl.total_duration:.1f}s"
        )

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
                seg.duration = (seg.end_time - seg.start_time) / max(seg.speed, 0.25)
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
