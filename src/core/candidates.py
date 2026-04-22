"""
Deterministic segment candidate generation for constrained editing.

The LLM may only reference candidates by integer id; start/end times and paths
come from analysis, reducing invented timestamps and bogus file paths.
"""

from __future__ import annotations

import numpy as np
from pathlib import Path

from src.core.models import MediaProfile, MediaType, SegmentCandidate
from src.utils.config import get as cfg


def _window_moment_score(profile: MediaProfile, start: float, end: float, scene_score: float | None) -> float:
    in_frames = [f for f in profile.frames if start <= f.timestamp <= end]
    if in_frames:
        frame_q = float(np.mean([f.visual_quality for f in in_frames]))
        motion = float(np.mean([f.motion_level for f in in_frames]))
    else:
        frame_q = float(profile.visual_quality)
        motion = 0.0
    base_scene = float(scene_score) if scene_score is not None else float(profile.relevance_score)
    rel = float(profile.relevance_score)
    return float(np.clip(0.42 * base_scene + 0.28 * rel + 0.22 * frame_q + 0.08 * motion, 0.0, 1.0))


def _hint_for_window(profile: MediaProfile, start: float, end: float, extra: str = "") -> str:
    tags = ", ".join(profile.overall_tags[:6])
    summ = (profile.summary or "").replace("\n", " ").strip()[:100]
    base = f"{tags}" if tags else summ or Path(profile.file_path).name
    if extra:
        return f"{base} | {extra}"[:220]
    return base[:220]


def build_segment_candidates(profiles: list[MediaProfile]) -> list[SegmentCandidate]:
    max_per = max(4, int(cfg("writer.max_candidates_per_asset", 20)))
    min_vid = float(cfg("writer.min_video_segment_seconds", 0.75))
    img_dur = float(cfg("writer.image_segment_seconds", 4.5))

    out: list[SegmentCandidate] = []
    idx = 0

    for p in profiles:
        if p.media_type == MediaType.AUDIO:
            continue

        if p.media_type == MediaType.IMAGE:
            hint = _hint_for_window(p, 0.0, 0.0)
            out.append(
                SegmentCandidate(
                    id=idx,
                    source_file=p.file_path,
                    media_type=p.media_type,
                    start_time=0.0,
                    end_time=0.0,
                    image_duration=img_dur,
                    hint=hint,
                    moment_score=round(float(np.clip(p.relevance_score, 0.0, 1.0)), 3),
                )
            )
            idx += 1
            continue

        # VIDEO
        duration = max(0.0, float(p.duration))
        scenes = list(p.scenes)
        windows: list[tuple[float, float, str, float | None]] = []

        if scenes:
            for s in sorted(scenes, key=lambda x: x.start_time):
                dur = float(s.end_time - s.start_time)
                if dur < min_vid:
                    continue
                hint_extra = (s.description or "").strip()[:80]
                windows.append(
                    (float(s.start_time), float(s.end_time), hint_extra, float(s.relevance_score))
                )
        else:
            if duration <= 0:
                continue
            n = min(max_per, max(4, int(np.ceil(duration / 6.0))))
            step = duration / n
            for i in range(n):
                t0 = i * step
                t1 = duration if i == n - 1 else (i + 1) * step
                if t1 - t0 < min_vid:
                    continue
                windows.append((t0, t1, "", None))

        if len(windows) > max_per:
            pick = np.linspace(0, len(windows) - 1, num=max_per, dtype=int)
            windows = [windows[int(j)] for j in sorted(set(pick))]

        for t0, t1, extra, scene_rel in windows:
            score = _window_moment_score(p, t0, t1, scene_rel)
            hint = _hint_for_window(p, t0, t1, extra)
            out.append(
                SegmentCandidate(
                    id=idx,
                    source_file=p.file_path,
                    media_type=p.media_type,
                    start_time=round(t0, 3),
                    end_time=round(t1, 3),
                    image_duration=0.0,
                    hint=hint,
                    moment_score=round(score, 3),
                )
            )
            idx += 1

    return out
