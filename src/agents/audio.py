"""
Audio Agent — handles audio enhancement, music selection, and mixing.

Responsibilities:
- Normalize audio levels across the assembled video
- Add background music if available
- Apply fade in/out
- Optionally generate subtitle track
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Optional

from src.agents.base import BaseAgent
from src.core.models import CreativeBrief, EditDecisionList
from src.utils.config import get as cfg
from src.utils.logger import get_logger


class AudioAgent(BaseAgent):
    name = "Audio"

    def run(
        self,
        video_path: Path,
        brief: CreativeBrief,
        edl: EditDecisionList,
        music_dir: Optional[Path] = None,
        music_file: Optional[Path] = None,
    ) -> Path:
        """Enhance the audio of the assembled video."""
        video_path = Path(video_path)
        self.log.info(f"[bold magenta]Audio[/] processing: {video_path.name}")

        output_path = video_path.parent / f"audio_enhanced_{video_path.name}"

        output_path = self._normalize_audio(video_path, output_path)

        output_path = self._apply_fades(output_path, brief)

        track = music_file
        if track is None and music_dir and music_dir.exists():
            track = self._select_music(music_dir, brief)

        if track and track.exists():
            output_path = self._mix_music(output_path, track, brief)
        elif music_file is None and (not music_dir or not music_dir.exists()):
            self.log.info(
                "  No background music — enable music_generation in config, use --generate-music, "
                "or add tracks to assets/music / --music-dir."
            )
        elif track is None:
            self.log.info(
                "  No audio files in music folder — add .mp3/.wav or use generated BGM."
            )

        self.log.info(f"[bold green]Audio[/] ✓ Enhanced: {output_path}")
        return output_path

    def _normalize_audio(self, input_path: Path, output_path: Path) -> Path:
        """Normalize audio levels using FFmpeg loudnorm filter."""
        try:
            cmd = [
                "ffmpeg", "-y", "-i", str(input_path),
                "-af", "loudnorm=I=-16:TP=-1.5:LRA=11",
                "-c:v", "copy",
                "-c:a", "aac", "-b:a", "192k",
                str(output_path),
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            if result.returncode == 0 and output_path.exists():
                self.log.info("  Audio normalized")
                return output_path
            else:
                self.log.warning(f"  Normalization failed: {result.stderr[:200]}")
                return input_path
        except Exception as e:
            self.log.warning(f"  Normalization error: {e}")
            return input_path

    def _apply_fades(self, input_path: Path, brief: CreativeBrief) -> Path:
        """Apply audio fade in at start and fade out at end."""
        fade_dur = cfg("audio.fade_duration", 1.5)
        output_path = input_path.parent / f"faded_{input_path.name}"

        try:
            duration = self._get_duration(input_path)
            if duration <= 0:
                return input_path

            fade_out_start = max(0, duration - fade_dur)

            cmd = [
                "ffmpeg", "-y", "-i", str(input_path),
                "-af", f"afade=t=in:st=0:d={fade_dur},afade=t=out:st={fade_out_start}:d={fade_dur}",
                "-c:v", "copy",
                "-c:a", "aac", "-b:a", "192k",
                str(output_path),
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            if result.returncode == 0 and output_path.exists():
                self.log.info("  Audio fades applied")
                # Clean up intermediate file if different from original
                if input_path.name.startswith(("audio_enhanced_", "faded_")):
                    input_path.unlink(missing_ok=True)
                return output_path
            return input_path
        except Exception as e:
            self.log.warning(f"  Fade error: {e}")
            return input_path

    def _select_music(self, music_dir: Path, brief: CreativeBrief) -> Optional[Path]:
        """Select the best background music file based on the creative brief."""
        music_extensions = {".mp3", ".wav", ".aac", ".flac", ".ogg", ".m4a"}
        music_files = [
            f for f in music_dir.iterdir()
            if f.is_file() and f.suffix.lower() in music_extensions
        ]

        if not music_files:
            self.log.info("  No music files found")
            return None

        # Simple heuristic: match mood keywords in filenames
        mood = brief.mood.value.lower()
        for f in music_files:
            if mood in f.stem.lower():
                self.log.info(f"  Selected music: {f.name} (mood match)")
                return f

        self.log.info(f"  Selected music: {music_files[0].name} (default)")
        return music_files[0]

    def _mix_music(self, video_path: Path, music_path: Path, brief: CreativeBrief) -> Path:
        """Mix background music into the video."""
        music_vol = cfg("audio.music_volume", 0.15)
        fade_dur = cfg("audio.fade_duration", 1.5)
        output_path = video_path.parent / f"music_{video_path.name}"

        try:
            duration = self._get_duration(video_path)

            cmd = [
                "ffmpeg", "-y",
                "-i", str(video_path),
                "-i", str(music_path),
                "-filter_complex",
                f"[1:a]aloop=loop=-1:size=2e+09,atrim=0:{duration},"
                f"volume={music_vol},"
                f"afade=t=in:st=0:d={fade_dur},"
                f"afade=t=out:st={max(0, duration - fade_dur)}:d={fade_dur}[music];"
                f"[0:a][music]amix=inputs=2:duration=first[out]",
                "-map", "0:v", "-map", "[out]",
                "-c:v", "copy",
                "-c:a", "aac", "-b:a", "192k",
                str(output_path),
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
            if result.returncode == 0 and output_path.exists():
                self.log.info(f"  Background music mixed (vol={music_vol})")
                if video_path.name.startswith(("audio_enhanced_", "faded_")):
                    video_path.unlink(missing_ok=True)
                return output_path
            else:
                self.log.warning(f"  Music mixing failed: {result.stderr[:200]}")
                return video_path
        except Exception as e:
            self.log.warning(f"  Music mixing error: {e}")
            return video_path

    def _get_duration(self, path: Path) -> float:
        try:
            result = subprocess.run(
                [
                    "ffprobe", "-v", "quiet",
                    "-show_entries", "format=duration",
                    "-of", "csv=p=0",
                    str(path),
                ],
                capture_output=True, text=True, timeout=10,
            )
            return float(result.stdout.strip())
        except Exception:
            return 0.0
