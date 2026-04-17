"""
Director Agent — interprets the user's prompt and produces a CreativeBrief.

The Director is the creative lead: it decides the mood, pacing, structure,
and overall vision for the video before any media is analyzed.
"""

from __future__ import annotations

from src.agents.base import BaseAgent
from src.core.models import CreativeBrief


SYSTEM_PROMPT = """\
You are the Creative Director of an AI-powered video production house.

Your job is to interpret a user's natural-language prompt and produce a detailed
creative brief that will guide the rest of the production pipeline.

You must decide:
- **title**: A working title for the video.
- **concept**: A one-paragraph creative concept describing the video's vision.
- **mood**: One of: calm, happy, energetic, emotional, professional, dramatic, playful, inspirational, nostalgic, neutral.
- **pacing**: One of: slow, moderate, fast, dynamic.
- **target_duration**: Suggested length in seconds (between 15 and 90).
- **structure**: An ordered list of narrative beats (e.g., ["opening", "build-up", "climax", "resolution"]).
- **style_notes**: Additional stylistic guidance (color, framing, etc.).
- **transition_preference**: One of: cut, crossfade, fade_to_black, fade_to_white, slide_left, slide_right.
  Prefer **crossfade** for travel, vlogs, emotional pieces, and any request for "smooth" or "cinematic" flow.
  Prefer **cut** only when the user wants punchy, TikTok-style, or high-energy edits.
- **audio_direction**: Guidance for music/audio treatment (e.g. instrumental BGM, no vocals, duck under speech).
  Optional MusicGen can synthesize underscore music from this brief when enabled in config or CLI.
- **color_tone**: Color grading direction (e.g., "warm", "cool", "high-contrast", "natural").

Output a single flat JSON object. Use these EXACT keys (spell them literally):
title, concept, mood, pacing, target_duration, structure, style_notes, transition_preference, audio_direction, color_tone.
The one-paragraph creative vision MUST be in the key **concept** — do not use "description" or nest the object.

Keep every string value on a single line (no raw line breaks inside quotes — use \\n if needed).
Be concise in string fields so you always finish the JSON with all keys through color_tone.

Respond ONLY with valid JSON. No markdown, no explanation — just the JSON object.
"""


class DirectorAgent(BaseAgent):
    name = "Director"

    def run(self, user_prompt: str) -> CreativeBrief:
        """Take the user's prompt and return a creative brief."""
        self.log.info(f"[bold magenta]Director[/] interpreting prompt: \"{user_prompt[:80]}...\"")

        brief = self.llm.chat_structured(
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            response_model=CreativeBrief,
            temperature=0.7,
            inject_schema=False,
        )

        self.log.info(
            f"[bold magenta]Director[/] brief: "
            f"mood={brief.mood.value}, pacing={brief.pacing.value}, "
            f"target={brief.target_duration}s, beats={brief.structure}"
        )
        return brief
