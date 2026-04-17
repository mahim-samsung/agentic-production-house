"""
Pydantic data models shared across all agents.

These models define the contracts between agents — each agent produces
and consumes well-typed structures so the pipeline stays composable.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, Field, model_validator


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class MediaType(str, Enum):
    VIDEO = "video"
    IMAGE = "image"
    AUDIO = "audio"


class Mood(str, Enum):
    CALM = "calm"
    HAPPY = "happy"
    ENERGETIC = "energetic"
    EMOTIONAL = "emotional"
    PROFESSIONAL = "professional"
    DRAMATIC = "dramatic"
    PLAYFUL = "playful"
    INSPIRATIONAL = "inspirational"
    NOSTALGIC = "nostalgic"
    NEUTRAL = "neutral"


class PacingStyle(str, Enum):
    SLOW = "slow"
    MODERATE = "moderate"
    FAST = "fast"
    DYNAMIC = "dynamic"  # varies throughout


class TransitionType(str, Enum):
    CUT = "cut"
    CROSSFADE = "crossfade"
    FADE_BLACK = "fade_to_black"
    FADE_WHITE = "fade_to_white"
    SLIDE_LEFT = "slide_left"
    SLIDE_RIGHT = "slide_right"


# ---------------------------------------------------------------------------
# Media analysis outputs
# ---------------------------------------------------------------------------

class FrameAnalysis(BaseModel):
    """Analysis of a single video frame or image."""
    timestamp: float = Field(description="Timestamp in seconds (0 for images)")
    description: str = Field(description="Natural-language description of visual content")
    tags: list[str] = Field(default_factory=list, description="Semantic tags")
    emotion: str = Field(default="neutral", description="Dominant emotion conveyed")
    visual_quality: float = Field(default=0.5, ge=0.0, le=1.0, description="Quality score")
    motion_level: float = Field(default=0.0, ge=0.0, le=1.0, description="Amount of motion")


class SceneSegment(BaseModel):
    """A detected scene within a video clip."""
    start_time: float
    end_time: float
    description: str = ""
    dominant_emotion: str = "neutral"
    relevance_score: float = Field(default=0.5, ge=0.0, le=1.0)
    tags: list[str] = Field(default_factory=list)


class AudioProfile(BaseModel):
    """Audio characteristics of a media file."""
    has_speech: bool = False
    has_music: bool = False
    transcript: str = ""
    avg_volume: float = 0.0
    language: str = "unknown"


class MediaProfile(BaseModel):
    """Complete analysis of a single media asset."""
    file_path: str
    media_type: MediaType
    duration: float = Field(default=0.0, description="Duration in seconds (0 for images)")
    width: int = 0
    height: int = 0
    fps: float = 0.0
    file_size_mb: float = 0.0

    frames: list[FrameAnalysis] = Field(default_factory=list)
    scenes: list[SceneSegment] = Field(default_factory=list)
    audio: Optional[AudioProfile] = None

    summary: str = Field(default="", description="LLM-generated overall summary")
    overall_tags: list[str] = Field(default_factory=list)
    overall_emotion: str = "neutral"
    visual_quality: float = Field(default=0.5, ge=0.0, le=1.0)
    relevance_score: float = Field(default=0.5, ge=0.0, le=1.0)


# ---------------------------------------------------------------------------
# Director output
# ---------------------------------------------------------------------------

class CreativeBrief(BaseModel):
    """High-level creative direction produced by the Director agent."""
    title: str = Field(description="Working title for the video")
    concept: str = Field(description="One-paragraph creative concept")
    mood: Mood = Mood.NEUTRAL
    pacing: PacingStyle = PacingStyle.MODERATE
    target_duration: float = Field(default=30.0, description="Target duration in seconds")
    structure: list[str] = Field(
        default_factory=list,
        description="Ordered list of narrative beats, e.g. ['opening', 'build-up', 'climax', 'resolution']",
    )
    style_notes: str = Field(default="", description="Additional stylistic guidance")
    transition_preference: TransitionType = TransitionType.CROSSFADE
    audio_direction: str = Field(default="", description="Guidance for audio/music treatment")
    color_tone: str = Field(default="natural", description="Color grading direction")

    model_config = {"extra": "ignore"}

    @model_validator(mode="before")
    @classmethod
    def _normalize_director_json(cls, data: Any) -> Any:
        """Local LLMs often emit 'description' instead of 'concept' when schema text confuses them."""
        if not isinstance(data, dict):
            return data
        out = {k: v for k, v in data.items() if k not in ("type", "$schema")}
        if "concept" not in out and "description" in out:
            d = out["description"]
            if isinstance(d, str):
                out["concept"] = d
            elif isinstance(d, dict) and "description" in d:
                inner = d["description"]
                if isinstance(inner, str):
                    out["concept"] = inner
        return out


# ---------------------------------------------------------------------------
# Writer output — Edit Decision List
# ---------------------------------------------------------------------------

class ClipSegment(BaseModel):
    """A single segment to include in the final edit."""
    source_file: str = Field(description="Path to the source media file")
    start_time: float = Field(default=0.0, description="Start time within source (seconds)")
    end_time: float = Field(default=0.0, description="End time within source (seconds, 0 = full)")
    duration: float = Field(default=0.0, description="Computed duration for this segment")
    speed: float = Field(default=1.0, description="Playback speed multiplier")
    transition_in: TransitionType = TransitionType.CUT
    transition_duration: float = Field(default=0.5, description="Transition duration in seconds")
    text_overlay: str = Field(default="", description="Optional text overlay")
    narrative_role: str = Field(default="", description="Role in the narrative (e.g., 'opening', 'climax')")
    reasoning: str = Field(default="", description="Why this segment was selected")


class EditDecisionList(BaseModel):
    """The complete edit plan — a sequence of clips with transitions."""
    title: str
    segments: list[ClipSegment] = Field(default_factory=list)
    total_duration: float = Field(default=0.0, description="Estimated total duration")
    narrative_summary: str = Field(default="", description="Summary of the narrative flow")
    audio_plan: str = Field(default="", description="Plan for audio treatment")


# ---------------------------------------------------------------------------
# Final output metadata
# ---------------------------------------------------------------------------

class ProductionReport(BaseModel):
    """Metadata about the final produced video."""
    output_path: str
    duration: float
    resolution: str
    creative_brief: CreativeBrief
    edit_decision_list: EditDecisionList
    media_profiles: list[MediaProfile] = Field(default_factory=list)
    processing_time_seconds: float = 0.0
