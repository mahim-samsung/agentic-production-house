"""
Video processing engine — assembles the final video from an EditDecisionList.

Uses MoviePy for compositing and transitions, with FFmpeg as the backend.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import numpy as np
from moviepy import (
    AudioFileClip,
    ColorClip,
    CompositeVideoClip,
    ImageClip,
    TextClip,
    VideoFileClip,
    concatenate_videoclips,
    vfx,
)
from moviepy.audio import fx as afx
from moviepy.audio.AudioClip import AudioClip

from src.core.models import (
    ClipSegment,
    EditDecisionList,
    MediaType,
    TransitionType,
)
from src.core.media import classify_media
from src.utils.config import get as cfg
from src.utils.logger import get_logger

log = get_logger("video")


def apply_timeline_audio_policy(final_clip: Any) -> Any:
    """
    When ``audio.keep_source_audio`` is false, drop any decoded clip audio and
    attach a silent stereo bed so later FFmpeg steps always see an audio stream.
    """
    keep_src = bool(cfg("audio.keep_source_audio", False))
    if keep_src:
        return final_clip
    dur = getattr(final_clip, "duration", None)
    if dur is None or dur <= 0:
        return final_clip
    if getattr(final_clip, "audio", None) is not None:
        final_clip = final_clip.without_audio()
    sr = int(cfg("audio.sample_rate", 44100))
    silent = AudioClip(
        lambda t: np.zeros(2, dtype=np.float32),
        duration=dur,
        fps=sr,
    )
    out = final_clip.with_audio(silent)
    log.info("Original clip audio omitted; silent timeline (BGM can still be added in audio pass)")
    return out


class VideoAssembler:
    """Builds the final video from an edit decision list."""

    def __init__(self):
        self.output_fps: int = cfg("video.output_fps", 30)
        res = cfg("video.output_resolution", [1920, 1080])
        self.output_width: int = res[0]
        self.output_height: int = res[1]
        self.codec: str = cfg("video.output_codec", "libx264")
        self.default_transition: float = cfg("video.transition_duration", 0.5)

    def apply_profile(self, profile: dict) -> None:
        """Apply render overrides from a platform profile."""
        if not profile:
            return

        if "output_fps" in profile:
            self.output_fps = int(profile["output_fps"])
        if "output_resolution" in profile:
            res = profile["output_resolution"]
            if isinstance(res, list) and len(res) == 2:
                self.output_width = int(res[0])
                self.output_height = int(res[1])
        if "output_codec" in profile:
            self.codec = str(profile["output_codec"])
        if "transition_duration" in profile:
            self.default_transition = float(profile["transition_duration"])

    def assemble(
        self,
        edl: EditDecisionList,
        output_path: Path,
        background_music: Optional[Path] = None,
    ) -> Path:
        """Execute the EDL and write the final video."""
        log.info(f"Assembling {len(edl.segments)} segments → {output_path}")
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        clips = []
        for i, seg in enumerate(edl.segments):
            try:
                clip = self._load_segment(seg)
                if clip is not None:
                    clips.append((clip, seg))
                    log.info(f"  Segment {i+1}: {Path(seg.source_file).name} "
                             f"[{seg.start_time:.1f}–{seg.end_time:.1f}s]")
            except Exception as e:
                log.warning(f"  Segment {i+1} failed: {e}")

        if not clips:
            raise RuntimeError("No valid clips to assemble")

        final_clips = self._apply_transitions(clips)

        if len(final_clips) == 1:
            final = final_clips[0]
        else:
            final = concatenate_videoclips(final_clips, method="compose")

        final = apply_timeline_audio_policy(final)

        if background_music and background_music.exists():
            final = self._mix_background_music(final, background_music)

        log.info(f"Writing final video ({final.duration:.1f}s) → {output_path}")
        final.write_videofile(
            str(output_path),
            fps=self.output_fps,
            codec=self.codec,
            audio_codec="aac",
            logger=None,
        )

        for clip, _ in clips:
            clip.close()
        final.close()

        log.info(f"Video written: {output_path} ({output_path.stat().st_size / 1e6:.1f} MB)")
        return output_path

    def _load_segment(self, seg: ClipSegment) -> Optional[any]:
        """Load a clip segment from disk and apply trimming/speed."""
        path = Path(seg.source_file)
        if not path.exists():
            log.warning(f"Source file not found: {path}")
            return None

        media_type = classify_media(path)

        if media_type == MediaType.VIDEO:
            keep_src = bool(cfg("audio.keep_source_audio", False))
            clip = VideoFileClip(str(path), audio=keep_src)
            if seg.end_time > 0 and seg.end_time > seg.start_time:
                clip = clip.subclipped(seg.start_time, min(seg.end_time, clip.duration))
            elif seg.start_time > 0:
                clip = clip.subclipped(seg.start_time)

            if seg.speed != 1.0:
                clip = clip.with_effects([vfx.MultiplySpeed(factor=seg.speed)])

        elif media_type == MediaType.IMAGE:
            duration = seg.duration if seg.duration > 0 else 4.0
            clip = ImageClip(str(path), duration=duration)

        else:
            return None

        clip = self._resize_clip(clip)

        if seg.text_overlay:
            clip = self._add_text_overlay(clip, seg.text_overlay)

        return clip

    def _resize_clip(self, clip) -> any:
        """Resize/pad clip to match the output resolution."""
        if clip.w == self.output_width and clip.h == self.output_height:
            return clip

        scale_w = self.output_width / clip.w
        scale_h = self.output_height / clip.h
        scale = min(scale_w, scale_h)

        new_w = int(clip.w * scale)
        new_h = int(clip.h * scale)
        clip = clip.resized((new_w, new_h))

        if new_w != self.output_width or new_h != self.output_height:
            bg = ColorClip(
                size=(self.output_width, self.output_height),
                color=(0, 0, 0),
                duration=clip.duration,
            )
            clip = CompositeVideoClip(
                [bg, clip.with_position("center")],
                size=(self.output_width, self.output_height),
            )

        return clip

    def _apply_transitions(self, clips: list[tuple]) -> list:
        """Apply transitions between consecutive clips."""
        if len(clips) <= 1:
            return [c for c, _ in clips]

        result = []
        for i, (clip, seg) in enumerate(clips):
            if i == 0:
                if seg.transition_in == TransitionType.FADE_BLACK:
                    clip = clip.with_effects([vfx.FadeIn(seg.transition_duration)])
                result.append(clip)
                continue

            transition = seg.transition_in
            t_dur = seg.transition_duration

            if transition == TransitionType.CROSSFADE:
                clip = clip.with_effects([vfx.CrossFadeIn(t_dur)])
                clip = clip.with_start(result[-1].end - t_dur)
                result.append(clip)
            elif transition in (TransitionType.FADE_BLACK, TransitionType.FADE_WHITE):
                prev = result[-1]
                result[-1] = prev.with_effects([vfx.FadeOut(t_dur)])
                clip = clip.with_effects([vfx.FadeIn(t_dur)])
                result.append(clip)
            else:
                result.append(clip)

        # Compose crossfaded clips
        has_crossfade = any(
            seg.transition_in == TransitionType.CROSSFADE
            for _, seg in clips[1:]
        )
        if has_crossfade:
            return [CompositeVideoClip(result)]

        return result

    def _add_text_overlay(self, clip, text: str):
        """Add a text overlay to the bottom of the clip."""
        try:
            txt = TextClip(
                text=text,
                font_size=40,
                color="white",
                font="Arial",
                stroke_color="black",
                stroke_width=2,
                duration=clip.duration,
            )
            txt = txt.with_position(("center", self.output_height - 100))
            return CompositeVideoClip(
                [clip, txt],
                size=(self.output_width, self.output_height),
            )
        except Exception as e:
            log.warning(f"Text overlay failed: {e}")
            return clip

    def _mix_background_music(self, video_clip, music_path: Path):
        """Mix background music under the video's existing audio."""
        try:
            music_vol = cfg("audio.music_volume", 0.15)
            fade_dur = cfg("audio.fade_duration", 1.5)

            music = AudioFileClip(str(music_path))
            if music.duration < video_clip.duration:
                loops_needed = int(video_clip.duration / music.duration) + 1
                from moviepy import concatenate_audioclips
                music = concatenate_audioclips([music] * loops_needed)

            music = music.subclipped(0, video_clip.duration)
            music = music.with_effects([
                afx.AudioFadeIn(fade_dur),
                afx.AudioFadeOut(fade_dur),
            ])
            music = music.with_volume_scaled(music_vol)

            if video_clip.audio is not None:
                from moviepy import CompositeAudioClip
                mixed = CompositeAudioClip([video_clip.audio, music])
                return video_clip.with_audio(mixed)
            else:
                return video_clip.with_audio(music)

        except Exception as e:
            log.warning(f"Background music mixing failed: {e}")
            return video_clip
