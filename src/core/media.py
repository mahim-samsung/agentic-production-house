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
_whisper_backend: str | None = None
_siglip2_model = None
_siglip2_processor = None
_internvideo_model = None
_internvideo_processor = None
_qwen25_model = None
_qwen25_processor = None


def _infer_device() -> str:
    try:
        import torch

        configured = str(cfg("media_analysis.device", "auto")).lower()
        if configured in {"cuda", "cpu"}:
            return configured
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def _move_batch_to_device(batch: dict, device: str) -> dict:
    moved = {}
    for k, v in batch.items():
        if hasattr(v, "to"):
            moved[k] = v.to(device)
        else:
            moved[k] = v
    return moved


def _load_clip():
    global _clip_model, _clip_preprocess, _clip_tokenizer
    if _clip_model is not None:
        return

    try:
        import open_clip
        import torch

        model_name = cfg("media_analysis.clip_model", "ViT-B-32")
        pretrained = cfg("media_analysis.clip_pretrained", "openai")
        device = _infer_device()

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


def _load_siglip2():
    global _siglip2_model, _siglip2_processor
    if _siglip2_model is not None and _siglip2_processor is not None:
        return

    try:
        from transformers import AutoModel, AutoProcessor

        model_id = cfg("media_analysis.siglip2.model_id", "google/siglip2-base-patch16-224")
        device = _infer_device()

        log.info(f"Loading SigLIP2 model {model_id} on {device}")
        _siglip2_processor = AutoProcessor.from_pretrained(model_id)
        _siglip2_model = AutoModel.from_pretrained(model_id)
        if hasattr(_siglip2_model, "to"):
            _siglip2_model = _siglip2_model.to(device).eval()
        log.info("SigLIP2 model loaded")
    except ImportError:
        log.warning("transformers not installed — SigLIP2 analysis disabled")
    except Exception as e:
        log.warning(f"Failed to load SigLIP2: {e}")


def _load_internvideo2():
    global _internvideo_model, _internvideo_processor
    if _internvideo_model is not None and _internvideo_processor is not None:
        return

    if not bool(cfg("media_analysis.internvideo2.enabled", False)):
        return

    try:
        from transformers import AutoModel, AutoProcessor

        model_id = cfg("media_analysis.internvideo2.model_id", "")
        if not model_id:
            return

        device = _infer_device()
        log.info(f"Loading InternVideo2 model {model_id} on {device}")
        _internvideo_processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        _internvideo_model = AutoModel.from_pretrained(model_id, trust_remote_code=True)
        if hasattr(_internvideo_model, "to"):
            _internvideo_model = _internvideo_model.to(device).eval()
        log.info("InternVideo2 model loaded")
    except Exception as e:
        log.warning(f"InternVideo2 disabled (load failed): {e}")


def _load_whisper():
    global _whisper_model, _whisper_backend
    if _whisper_model is not None:
        return

    model_size = cfg("media_analysis.whisper_model", "large-v3")
    backend = str(cfg("media_analysis.whisper_backend", "faster-whisper")).lower()
    device = _infer_device()

    if backend == "faster-whisper":
        try:
            from faster_whisper import WhisperModel

            compute_type = cfg(
                "media_analysis.whisper_compute_type",
                "float16" if device == "cuda" else "int8",
            )
            log.info(
                f"Loading faster-whisper model ({model_size}) on {device} "
                f"[compute_type={compute_type}]"
            )
            _whisper_model = WhisperModel(model_size, device=device, compute_type=compute_type)
            _whisper_backend = "faster-whisper"
            log.info("faster-whisper model loaded")
            return
        except ImportError:
            log.warning("faster-whisper not installed — trying openai-whisper fallback")
        except Exception as e:
            log.warning(f"Failed to load faster-whisper: {e}; trying openai-whisper fallback")

    try:
        import whisper

        log.info(f"Loading openai-whisper model ({model_size})")
        _whisper_model = whisper.load_model(model_size, device=device)
        _whisper_backend = "openai-whisper"
        log.info("openai-whisper model loaded")
    except ImportError:
        log.warning("openai-whisper not installed — transcription disabled")
    except Exception as e:
        log.warning(f"Failed to load Whisper: {e}")


def _load_qwen25_vl():
    """Lazy-load Qwen2.5-VL when vlm_semantics_backend is qwen2_5_vl."""
    global _qwen25_model, _qwen25_processor
    if str(cfg("media_analysis.vlm_semantics_backend", "none")).lower() != "qwen2_5_vl":
        return
    if _qwen25_model is not None and _qwen25_processor is not None:
        return

    try:
        import torch
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    except ImportError as e:
        log.warning(f"Qwen2.5-VL needs transformers with Qwen2_5_VL support: {e}")
        return

    model_id = str(cfg("media_analysis.qwen2_5_vl.model_id", "")).strip()
    if not model_id:
        log.warning("Qwen2.5-VL enabled but media_analysis.qwen2_5_vl.model_id is empty")
        return

    device = _infer_device()
    try:
        import torch

        log.info(f"Loading Qwen2.5-VL {model_id} (device={device})")
        if device == "cuda":
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            try:
                _qwen25_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    model_id,
                    torch_dtype=dtype,
                    device_map="auto",
                    attn_implementation="sdpa",
                    trust_remote_code=True,
                )
            except (TypeError, ValueError):
                _qwen25_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    model_id,
                    torch_dtype=dtype,
                    device_map="auto",
                    trust_remote_code=True,
                )
        else:
            _qwen25_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_id,
                torch_dtype=torch.float32,
                device_map=None,
                trust_remote_code=True,
            )
            _qwen25_model = _qwen25_model.to("cpu")
        _qwen25_processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        _qwen25_model.eval()
        log.info("Qwen2.5-VL loaded")
    except Exception as e:
        log.warning(f"Qwen2.5-VL load failed: {e}")
        _qwen25_model = None
        _qwen25_processor = None


def _qwen25_vl_caption_image(image: Image.Image) -> str:
    """One short caption for a PIL image; empty string on failure."""
    _load_qwen25_vl()
    if _qwen25_model is None or _qwen25_processor is None:
        return ""

    import torch

    user_prompt = str(
        cfg(
            "media_analysis.qwen2_5_vl.caption_user_prompt",
            "In 1–2 short sentences, describe the main subjects, setting, action, and mood "
            "for a travel or event video editor. No preamble.",
        )
    )
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": user_prompt},
            ],
        }
    ]
    try:
        inputs = _qwen25_processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
    except Exception as e:
        log.warning(f"Qwen2.5-VL chat template failed: {e}")
        return ""

    device = next(_qwen25_model.parameters()).device
    inputs = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in inputs.items()}
    max_new = int(cfg("media_analysis.qwen2_5_vl.max_new_tokens", 96))
    try:
        with torch.inference_mode():
            gen_ids = _qwen25_model.generate(**inputs, max_new_tokens=max_new)
    except Exception as e:
        log.warning(f"Qwen2.5-VL generate failed: {e}")
        return ""

    in_ids = inputs["input_ids"][0]
    trimmed = gen_ids[0, in_ids.shape[0] :].detach().cpu()
    try:
        decoded = _qwen25_processor.batch_decode(
            [trimmed],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        text = (decoded[0] if decoded else "").replace("\n", " ").strip()
    except Exception:
        tok = getattr(_qwen25_processor, "tokenizer", None)
        if tok is None:
            return ""
        text = tok.decode(trimmed, skip_special_tokens=True).replace("\n", " ").strip()
    return text[:500]


def _maybe_apply_vlm_semantics(
    path: Path,
    media_type: MediaType,
    sampled_frames: list[tuple[float, Image.Image]],
    frame_analyses: list[FrameAnalysis],
    scenes: list[SceneSegment],
) -> None:
    """Optional Qwen2.5-VL captions layered on SigLIP/OpenCLIP frame tags."""
    if str(cfg("media_analysis.vlm_semantics_backend", "none")).lower() != "qwen2_5_vl":
        return

    if media_type == MediaType.VIDEO:
        max_scenes = int(cfg("media_analysis.qwen2_5_vl.max_scene_captions", 20))
        max_orphan = int(cfg("media_analysis.qwen2_5_vl.max_orphan_frame_captions", 4))

        if scenes:
            n = len(scenes)
            idxs = list(range(n))
            if n > max_scenes:
                idxs = sorted(set(np.linspace(0, n - 1, num=max_scenes, dtype=int)))
            imgs = _extract_scene_midpoint_images(path, scenes)
            for i in idxs:
                if i >= len(imgs):
                    break
                cap = _qwen25_vl_caption_image(imgs[i])
                if cap:
                    scenes[i].description = cap
            log.info(f"Qwen2.5-VL: captioned {len(idxs)} scene(s) in {path.name}")
        else:
            if not sampled_frames or not frame_analyses:
                return
            cap_n = min(len(sampled_frames), max_orphan, len(frame_analyses))
            pick = sorted(set(np.linspace(0, len(sampled_frames) - 1, num=cap_n, dtype=int)))
            for idx in pick:
                if idx >= len(sampled_frames) or idx >= len(frame_analyses):
                    continue
                _, pil_img = sampled_frames[idx]
                cap = _qwen25_vl_caption_image(pil_img)
                if not cap:
                    continue
                fa = frame_analyses[idx]
                prev = (fa.description or "").strip()
                fa.description = f"{cap} ({prev})" if prev else cap
            log.info(f"Qwen2.5-VL: captioned {len(pick)} frame(s) (no scenes) in {path.name}")

    elif media_type == MediaType.IMAGE and sampled_frames and frame_analyses:
        _, pil_img = sampled_frames[0]
        cap = _qwen25_vl_caption_image(pil_img)
        if cap:
            fa = frame_analyses[0]
            prev = (fa.description or "").strip()
            fa.description = f"{cap} ({prev})" if prev else cap
            log.info(f"Qwen2.5-VL: captioned image {path.name}")


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
    scenes = _score_scene_relevance(
        scenes=scenes,
        frames=frame_analyses,
        audio=audio,
        media_path=path,
    )
    _maybe_apply_vlm_semantics(path, MediaType.VIDEO, sampled_frames, frame_analyses, scenes)

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
    _maybe_apply_vlm_semantics(path, MediaType.IMAGE, [(0.0, img)], frame_analyses, [])

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
    """Analyze frames using configured vision backend."""
    backend = str(cfg("media_analysis.vision_backend", "open_clip")).lower()
    if backend == "siglip2":
        analyzed = _analyze_frames_siglip2(frames)
        if analyzed:
            return analyzed
        log.warning("SigLIP2 unavailable, falling back to OpenCLIP")

    analyzed = _analyze_frames_open_clip(frames)
    if analyzed:
        return analyzed

    return [
        FrameAnalysis(
            timestamp=ts,
            description="(vision model not available)",
            tags=[],
            motion_level=0.0,
        )
        for ts, _ in frames
    ]


def _analyze_frames_open_clip(frames: list[tuple[float, Image.Image]]) -> list[FrameAnalysis]:
    _load_clip()
    if _clip_model is None:
        return []

    import torch

    device = next(_clip_model.parameters()).device
    tag_candidates, emotion_candidates = _label_candidates()
    tag_tokens = _clip_tokenizer(tag_candidates).to(device)
    emotion_tokens = _clip_tokenizer(emotion_candidates).to(device)

    results = []
    with torch.no_grad():
        tag_features = _clip_model.encode_text(tag_tokens)
        tag_features /= tag_features.norm(dim=-1, keepdim=True)
        emo_features = _clip_model.encode_text(emotion_tokens)
        emo_features /= emo_features.norm(dim=-1, keepdim=True)

        prev_gray = None
        for timestamp, img in frames:
            img_tensor = _clip_preprocess(img).unsqueeze(0).to(device)
            img_features = _clip_model.encode_image(img_tensor)
            img_features /= img_features.norm(dim=-1, keepdim=True)

            tag_scores = (img_features @ tag_features.T).squeeze().cpu().numpy()
            emo_scores = (img_features @ emo_features.T).squeeze().cpu().numpy()
            top_tags, dominant_emotion, quality = _postprocess_label_scores(
                tag_scores, emo_scores, tag_candidates, emotion_candidates
            )
            motion_level, prev_gray = _calc_motion(img, prev_gray)

            results.append(
                FrameAnalysis(
                    timestamp=round(timestamp, 2),
                    description=f"Scene with: {', '.join(top_tags[:3])}",
                    tags=top_tags,
                    emotion=dominant_emotion,
                    visual_quality=round(quality, 2),
                    motion_level=round(motion_level, 2),
                )
            )
    return results


def _analyze_frames_siglip2(frames: list[tuple[float, Image.Image]]) -> list[FrameAnalysis]:
    _load_siglip2()
    if _siglip2_model is None or _siglip2_processor is None:
        return []

    import torch

    device = _infer_device()
    tag_candidates, emotion_candidates = _label_candidates()
    all_labels = tag_candidates + emotion_candidates
    results = []
    prev_gray = None

    with torch.no_grad():
        for timestamp, img in frames:
            batch = _siglip2_processor(
                images=img,
                text=all_labels,
                return_tensors="pt",
                padding=True,
            )
            batch = _move_batch_to_device(batch, device)
            out = _siglip2_model(**batch)
            logits = out.logits_per_image[0].detach().float().cpu().numpy()

            tag_scores = logits[: len(tag_candidates)]
            emo_scores = logits[len(tag_candidates) :]
            top_tags, dominant_emotion, quality = _postprocess_label_scores(
                tag_scores, emo_scores, tag_candidates, emotion_candidates
            )
            motion_level, prev_gray = _calc_motion(img, prev_gray)

            results.append(
                FrameAnalysis(
                    timestamp=round(timestamp, 2),
                    description=f"Scene with: {', '.join(top_tags[:3])}",
                    tags=top_tags,
                    emotion=dominant_emotion,
                    visual_quality=round(quality, 2),
                    motion_level=round(motion_level, 2),
                )
            )
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
# Key-moment scoring
# ---------------------------------------------------------------------------

def _score_scene_relevance(
    scenes: list[SceneSegment],
    frames: list[FrameAnalysis],
    audio: Optional[AudioProfile],
    media_path: Path,
) -> list[SceneSegment]:
    if not scenes:
        return scenes

    # Strong default heuristic: combine visual quality, motion, and speech presence.
    for s in scenes:
        in_scene = [f for f in frames if s.start_time <= f.timestamp <= s.end_time]
        if not in_scene:
            s.relevance_score = 0.5
            continue
        q = float(np.mean([f.visual_quality for f in in_scene]))
        m = float(np.mean([f.motion_level for f in in_scene]))
        speech_bonus = 0.1 if audio and audio.has_speech else 0.0
        s.relevance_score = float(np.clip(0.55 * q + 0.35 * m + speech_bonus, 0.0, 1.0))

    backend = str(cfg("media_analysis.video_moment_backend", "internvideo2")).lower()
    if backend == "internvideo2":
        _apply_internvideo2_scene_boost(media_path, scenes)

    return scenes


def _apply_internvideo2_scene_boost(media_path: Path, scenes: list[SceneSegment]) -> None:
    _load_internvideo2()
    if _internvideo_model is None or _internvideo_processor is None or not scenes:
        return

    try:
        # Lightweight boost: compare middle-frame embeddings against generic "highlight" prompts.
        import torch

        prompt_labels = [
            "a key highlight moment",
            "an emotional peak",
            "a dramatic action moment",
            "an unimportant filler scene",
        ]
        device = _infer_device()
        midpoint_frames = _extract_scene_midpoint_images(media_path, scenes)
        if not midpoint_frames:
            return

        boost_scores = []
        for img in midpoint_frames:
            batch = _internvideo_processor(
                images=img,
                text=prompt_labels,
                return_tensors="pt",
                padding=True,
            )
            batch = _move_batch_to_device(batch, device)
            with torch.no_grad():
                out = _internvideo_model(**batch)
                logits = out.logits_per_image[0].detach().float().cpu().numpy()

            pos = float(np.max(logits[:3]))
            neg = float(logits[3])
            boost = float(np.clip((pos - neg) / 12.0, -0.15, 0.15))
            boost_scores.append(boost)

        for i, s in enumerate(scenes):
            if i < len(boost_scores):
                s.relevance_score = float(np.clip(s.relevance_score + boost_scores[i], 0.0, 1.0))
    except Exception as e:
        log.warning(f"InternVideo2 scene boost failed: {e}")


def _extract_scene_midpoint_images(path: Path, scenes: list[SceneSegment]) -> list[Image.Image]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return []
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    images: list[Image.Image] = []
    try:
        for s in scenes:
            mid = max(0.0, (s.start_time + s.end_time) / 2.0)
            idx = int(min(frame_count - 1, max(0, mid * fps)))
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if ok:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                images.append(Image.fromarray(rgb))
    finally:
        cap.release()
    return images


# ---------------------------------------------------------------------------
# Audio analysis
# ---------------------------------------------------------------------------

def _analyze_audio(path: Path) -> Optional[AudioProfile]:
    """Extract and analyze audio from a media file."""
    _load_whisper()

    if _whisper_model is None:
        return _basic_audio_probe(path)

    try:
        transcript = ""
        language = "unknown"

        if _whisper_backend == "faster-whisper":
            segments, info = _whisper_model.transcribe(
                str(path),
                beam_size=int(cfg("media_analysis.whisper_beam_size", 5)),
                vad_filter=bool(cfg("media_analysis.whisper_vad_filter", True)),
            )
            transcript = " ".join(seg.text.strip() for seg in segments).strip()
            language = getattr(info, "language", "unknown")
        else:
            result = _whisper_model.transcribe(
                str(path),
                fp16=(_infer_device() == "cuda"),
            )
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

def _label_candidates() -> tuple[list[str], list[str]]:
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
    return tag_candidates, emotion_candidates


def _postprocess_label_scores(
    tag_scores: np.ndarray,
    emo_scores: np.ndarray,
    tag_candidates: list[str],
    emotion_candidates: list[str],
) -> tuple[list[str], str, float]:
    top_tag_idx = tag_scores.argsort()[-5:][::-1]
    top_tags = [tag_candidates[i] for i in top_tag_idx]
    dominant_emotion = emotion_candidates[int(np.argmax(emo_scores))]
    quality = float(np.clip(float(np.max(tag_scores)) / 10.0 + 0.5, 0.0, 1.0))
    return top_tags, dominant_emotion, quality


def _calc_motion(img: Image.Image, prev_gray: Optional[np.ndarray]) -> tuple[float, np.ndarray]:
    gray = np.asarray(img.convert("L"), dtype=np.float32)
    if prev_gray is None:
        return 0.0, gray
    diff = np.mean(np.abs(gray - prev_gray))
    return float(np.clip(diff / 32.0, 0.0, 1.0)), gray


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
