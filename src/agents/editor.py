"""
Editor Agent — takes the EDL and assembles the actual video file.

This agent bridges the planning stage (Writer) and the execution stage
(VideoAssembler). It can also make last-minute adjustments to the EDL
based on practical constraints.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from src.agents.base import BaseAgent
from src.core.models import EditDecisionList
from src.core.video import VideoAssembler, apply_timeline_audio_policy


class EditorAgent(BaseAgent):
    name = "Editor"

    def __init__(self, llm, output_dir: Path | str = "output"):
        super().__init__(llm)
        self.output_dir = Path(output_dir)
        self.assembler = VideoAssembler()

    def run(
        self,
        edl: EditDecisionList,
        output_filename: str = "final_video.mp4",
        background_music: Optional[Path] = None,
    ) -> Path:
        """Assemble the video from the edit decision list."""
        self.log.info(
            f"[bold magenta]Editor[/] assembling \"{edl.title}\" "
            f"({len(edl.segments)} segments, ~{edl.total_duration:.1f}s)"
        )

        output_path = self.output_dir / output_filename

        try:
            result_path = self.assembler.assemble(
                edl=edl,
                output_path=output_path,
                background_music=background_music,
            )
            self.log.info(f"[bold green]Editor[/] ✓ Video assembled: {result_path}")
            return result_path

        except Exception as e:
            self.log.error(f"[bold red]Editor[/] assembly failed: {e}")
            self.log.info("[bold magenta]Editor[/] attempting fallback assembly...")
            return self._fallback_assemble(edl, output_path)

    def apply_render_profile(self, profile: dict) -> None:
        """Apply platform-specific render settings to the assembler."""
        self.assembler.apply_profile(profile)

    def _fallback_assemble(self, edl: EditDecisionList, output_path: Path) -> Path:
        """Simplified assembly: concatenate clips without transitions."""
        from moviepy import VideoFileClip, ImageClip, concatenate_videoclips
        from src.core.media import classify_media
        from src.core.models import MediaType
        from src.utils.config import get as cfg

        clips = []
        for seg in edl.segments:
            try:
                path = Path(seg.source_file)
                media_type = classify_media(path)

                if media_type == MediaType.VIDEO:
                    keep_src = bool(cfg("audio.keep_source_audio", False))
                    clip = VideoFileClip(str(path), audio=keep_src)
                    if seg.end_time > 0:
                        clip = clip.subclipped(
                            seg.start_time,
                            min(seg.end_time, clip.duration),
                        )
                elif media_type == MediaType.IMAGE:
                    dur = seg.duration if seg.duration > 0 else 4.0
                    clip = ImageClip(str(path), duration=dur)
                else:
                    continue

                clips.append(clip)
            except Exception as e:
                self.log.warning(f"  Fallback skip: {seg.source_file} ({e})")

        if not clips:
            raise RuntimeError("No clips could be loaded even in fallback mode")

        final = concatenate_videoclips(clips, method="compose")
        final = apply_timeline_audio_policy(final)
        final.write_videofile(str(output_path), fps=30, codec="libx264", logger=None)

        for c in clips:
            c.close()
        final.close()

        self.log.info(f"[bold green]Editor[/] ✓ Fallback video: {output_path}")
        return output_path
