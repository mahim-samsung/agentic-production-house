"""
Media analysis utilities: frame extraction, CLIP embeddings, scene detection, audio analysis.

Heavy ML dependencies (torch, CLIP, whisper) are lazily imported so the system
still works in a degraded mode if they aren't installed.
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image

from src.core.models import (
    AudioProfile,
    FrameAnalysis,
    MediaProfile,
    MediaType,
    SceneSegment,
)
from src.utils.logger import get_logger
from src.utils.config import get as cfg

log = get_logger("media")

SUPPORTED_VIDEO = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".wmv"}
SUPPORTED_IMAGE = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".gif"}
SUPPORTED_AUDIO = {".mp3", ".wav", ".aac", ".flac", ".ogg", ".m4a"}


# ---------------------------------------------------------------------------
# Lazy-loaded ML models
# ---------------------------------------------------------------------------

_clip_model = None
_clip_preprocess = None
_clip_tokenizer = None
_whisper_model = None


def _load_clip():
    global _clip_model, _clip_preprocess, _clip_tokenizer
    if _clip_model is not None:
        return

    try:
        import open_clip
        import torch

        model_name = cfg("media_analysis.clip_model", "ViT-B-32")
        pretrained = cfg("media_analysis.clip_pretrained", "openai")
        device = "cuda" if torch.cuda.is_available() else "cpu"

        log.info(f"Loading CLIP model {model_name} ({pretrained}) on {device}")
        _clip_model, _, _clip_preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        _clip_model = _clip_model.to(device).eval()
        _clip_tokenizer = open_clip.get_tokenizer(model_name)
        log.info("CLIP model loaded")
    except ImportError:
        log.warning("open-clip-torch not installed — CLIP analysis disabled")
    except Exception as e:
        log.warning(f"Failed to load CLIP: {e}")


def _load_whisper():
    global _whisper_model
    if _whisper_model is not None:
        return

    try:
        import whisper

        model_size = cfg("media_analysis.whisper_model", "base")
        log.info(f"Loading Whisper model ({model_size})")
        _whisper_model = whisper.load_model(model_size)
        log.info("Whisper model loaded")
    except ImportError:
        log.warning("openai-whisper not installed — transcription disabled")
    except Exception as e:
        log.warning(f"Failed to load Whisper: {e}")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def classify_media(path: Path) -> MediaType:
    suffix = path.suffix.lower()
    if suffix in SUPPORTED_VIDEO:
        return MediaType.VIDEO
    if suffix in SUPPORTED_IMAGE:
        return MediaType.IMAGE
    if suffix in SUPPORTED_AUDIO:
        return MediaType.AUDIO
    raise ValueError(f"Unsupported media type: {path}")


def analyze_media(path: Path) -> MediaProfile:
    """Full analysis pipeline for a single media file."""
    path = Path(path)
    media_type = classify_media(path)
    file_size_mb = path.stat().st_size / (1024 * 1024)

    if media_type == MediaType.VIDEO:
        return _analyze_video(path, file_size_mb)
    elif media_type == MediaType.IMAGE:
        return _analyze_image(path, file_size_mb)
    else:
        return _analyze_audio_file(path, file_size_mb)


# ---------------------------------------------------------------------------
# Video analysis
# ---------------------------------------------------------------------------

def _analyze_video(path: Path, file_size_mb: float) -> MediaProfile:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = frame_count / fps if fps > 0 else 0.0

    frames_to_sample = cfg("media_analysis.frames_per_clip", 8)
    sampled_frames = _extract_key_frames(cap, frame_count, fps, frames_to_sample)
    cap.release()

    frame_analyses = _analyze_frames(sampled_frames)
    scenes = _detect_scenes(path)
    audio = _analyze_audio(path)

    return MediaProfile(
        file_path=str(path),
        media_type=MediaType.VIDEO,
        duration=round(duration, 2),
        width=width,
        height=height,
        fps=round(fps, 2),
        file_size_mb=round(file_size_mb, 2),
        frames=frame_analyses,
        scenes=scenes,
        audio=audio,
        overall_tags=_aggregate_tags(frame_analyses),
        visual_quality=_avg_quality(frame_analyses),
    )


def _analyze_image(path: Path, file_size_mb: float) -> MediaProfile:
    img = Image.open(path)
    width, height = img.size

    frame_analyses = _analyze_frames([(0.0, img)])

    return MediaProfile(
        file_path=str(path),
        media_type=MediaType.IMAGE,
        duration=0.0,
        width=width,
        height=height,
        file_size_mb=round(file_size_mb, 2),
        frames=frame_analyses,
        overall_tags=_aggregate_tags(frame_analyses),
        visual_quality=_avg_quality(frame_analyses),
    )


def _analyze_audio_file(path: Path, file_size_mb: float) -> MediaProfile:
    audio = _analyze_audio(path)
    duration = _get_audio_duration(path)
    return MediaProfile(
        file_path=str(path),
        media_type=MediaType.AUDIO,
        duration=duration,
        file_size_mb=round(file_size_mb, 2),
        audio=audio,
    )


# ---------------------------------------------------------------------------
# Frame extraction & CLIP analysis
# ---------------------------------------------------------------------------

def _extract_key_frames(
    cap: cv2.VideoCapture,
    frame_count: int,
    fps: float,
    num_frames: int,
) -> list[tuple[float, Image.Image]]:
    """Sample evenly-spaced frames from the video."""
    if frame_count <= 0:
        return []

    indices = np.linspace(0, frame_count - 1, num=min(num_frames, frame_count), dtype=int)
    frames: list[tuple[float, Image.Image]] = []

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if ret:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)
            timestamp = idx / fps if fps > 0 else 0.0
            frames.append((timestamp, pil_img))

    return frames


def _analyze_frames(frames: list[tuple[float, Image.Image]]) -> list[FrameAnalysis]:
    """Analyze frames using CLIP to generate descriptions and tags."""
    _load_clip()

    if _clip_model is None:
        return [
            FrameAnalysis(
                timestamp=ts,
                description="(CLIP not available)",
                tags=[],
            )
            for ts, _ in frames
        ]

    import torch
    device = next(_clip_model.parameters()).device
    tag_candidates = [
        "landscape", "cityscape", "beach", "mountain", "forest", "sunset", "sunrise",
        "indoor", "outdoor", "people", "crowd", "portrait", "close-up", "wide-shot",
        "action", "still", "food", "animal", "car", "building", "water", "sky",
        "night", "day", "celebration", "wedding", "party", "sport", "nature",
        "technology", "product", "art", "music", "dance", "travel", "work",
    ]
    emotion_candidates = [
        "happy", "calm", "excited", "sad", "dramatic", "peaceful",
        "energetic", "romantic", "nostalgic", "neutral",
    ]

    tag_tokens = _clip_tokenizer(tag_candidates).to(device)
    emotion_tokens = _clip_tokenizer(emotion_candidates).to(device)

    results = []
    with torch.no_grad():
        for timestamp, img in frames:
            img_tensor = _clip_preprocess(img).unsqueeze(0).to(device)
            img_features = _clip_model.encode_image(img_tensor)
            img_features /= img_features.norm(dim=-1, keepdim=True)

            # Tag scoring
            tag_features = _clip_model.encode_text(tag_tokens)
            tag_features /= tag_features.norm(dim=-1, keepdim=True)
            tag_scores = (img_features @ tag_features.T).squeeze().cpu().numpy()
            top_tag_idx = tag_scores.argsort()[-5:][::-1]
            top_tags = [tag_candidates[i] for i in top_tag_idx]

            # Emotion scoring
            emo_features = _clip_model.encode_text(emotion_tokens)
            emo_features /= emo_features.norm(dim=-1, keepdim=True)
            emo_scores = (img_features @ emo_features.T).squeeze().cpu().numpy()
            dominant_emotion = emotion_candidates[emo_scores.argmax()]

            # Quality heuristic based on CLIP confidence spread
            quality = float(np.clip(tag_scores.max() * 3.0, 0.0, 1.0))

            results.append(FrameAnalysis(
                timestamp=round(timestamp, 2),
                description=f"Scene with: {', '.join(top_tags[:3])}",
                tags=top_tags,
                emotion=dominant_emotion,
                visual_quality=round(quality, 2),
            ))

    return results


# ---------------------------------------------------------------------------
# Scene detection
# ---------------------------------------------------------------------------

def _detect_scenes(path: Path) -> list[SceneSegment]:
    """Use PySceneDetect for scene boundary detection."""
    try:
        from scenedetect import open_video, SceneManager
        from scenedetect.detectors import ContentDetector

        threshold = cfg("media_analysis.scene_threshold", 27.0)

        video = open_video(str(path))
        scene_manager = SceneManager()
        scene_manager.add_detector(ContentDetector(threshold=threshold))
        scene_manager.detect_scenes(video)
        scene_list = scene_manager.get_scene_list()

        segments = []
        for start, end in scene_list:
            segments.append(SceneSegment(
                start_time=start.get_seconds(),
                end_time=end.get_seconds(),
            ))

        log.info(f"Detected {len(segments)} scenes in {path.name}")
        return segments

    except ImportError:
        log.warning("scenedetect not installed — scene detection disabled")
        return []
    except Exception as e:
        log.warning(f"Scene detection failed for {path}: {e}")
        return []


# ---------------------------------------------------------------------------
# Audio analysis
# ---------------------------------------------------------------------------

def _analyze_audio(path: Path) -> Optional[AudioProfile]:
    """Extract and analyze audio from a media file."""
    _load_whisper()

    if _whisper_model is None:
        return _basic_audio_probe(path)

    try:
        import whisper

        result = _whisper_model.transcribe(str(path), fp16=False)
        transcript = result.get("text", "").strip()
        language = result.get("language", "unknown")

        return AudioProfile(
            has_speech=len(transcript) > 10,
            has_music=False,  # Whisper can't reliably detect music
            transcript=transcript,
            language=language,
        )
    except Exception as e:
        log.warning(f"Audio analysis failed: {e}")
        return _basic_audio_probe(path)


def _basic_audio_probe(path: Path) -> Optional[AudioProfile]:
    """Fallback: just check if the file has an audio stream."""
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "quiet",
                "-select_streams", "a",
                "-show_entries", "stream=codec_type",
                "-of", "csv=p=0",
                str(path),
            ],
            capture_output=True, text=True, timeout=10,
        )
        has_audio = "audio" in result.stdout.lower()
        return AudioProfile(has_speech=has_audio) if has_audio else None
    except Exception:
        return None


def _get_audio_duration(path: Path) -> float:
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _aggregate_tags(frames: list[FrameAnalysis]) -> list[str]:
    tag_counts: dict[str, int] = {}
    for f in frames:
        for t in f.tags:
            tag_counts[t] = tag_counts.get(t, 0) + 1
    return sorted(tag_counts, key=tag_counts.get, reverse=True)[:10]


def _avg_quality(frames: list[FrameAnalysis]) -> float:
    if not frames:
        return 0.5
    return round(sum(f.visual_quality for f in frames) / len(frames), 2)


def get_media_files(directory: Path) -> list[Path]:
    """Discover all supported media files in a directory."""
    all_extensions = SUPPORTED_VIDEO | SUPPORTED_IMAGE | SUPPORTED_AUDIO
    files = []
    for f in sorted(directory.iterdir()):
        if f.is_file() and f.suffix.lower() in all_extensions:
            files.append(f)
    return files
