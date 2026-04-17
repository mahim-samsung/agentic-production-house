"""
Background music generation with Meta MusicGen (Hugging Face Transformers).

Uses a text prompt derived from the creative brief. MusicGen is limited to ~30s
per generation; the audio mixer already loops shorter BGM to cover the full video.

Install: pip install -r requirements-musicgen.txt
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from src.core.models import CreativeBrief, EditDecisionList
from src.utils.config import get as cfg
from src.utils.logger import get_logger

log = get_logger("musicgen")

# Lazy singletons (model is heavy)
_processor = None
_model = None
_model_id_loaded: Optional[str] = None


def _get_device_dtype():
    import torch

    if cfg("music_generation.device", "auto") == "cpu":
        return "cpu", torch.float32
    if torch.cuda.is_available():
        return "cuda", torch.float16
    return "cpu", torch.float32


def _load_model(model_id: str):
    global _processor, _model, _model_id_loaded
    if _model is not None and _model_id_loaded == model_id:
        return

    import torch
    from transformers import AutoProcessor, MusicgenForConditionalGeneration

    device, dtype = _get_device_dtype()
    log.info(f"Loading MusicGen [{model_id}] on {device} (first run may download weights)")

    _processor = AutoProcessor.from_pretrained(model_id)
    _model = MusicgenForConditionalGeneration.from_pretrained(model_id, torch_dtype=dtype)
    _model.to(device)
    _model.eval()
    _model_id_loaded = model_id


def brief_to_music_prompt(
    brief: CreativeBrief,
    user_prompt: str,
    edl: Optional[EditDecisionList] = None,
) -> str:
    """Turn editorial context into a MusicGen-friendly text prompt."""
    pacing = brief.pacing.value
    mood = brief.mood.value
    tone = brief.color_tone or "natural"
    audio_dir = brief.audio_direction or ""
    concept_snip = (brief.concept or "")[:400]

    narrative = ""
    if edl and edl.narrative_summary:
        narrative = edl.narrative_summary[:300]

    # MusicGen works best with concrete genre/mood/instrument words
    return (
        f"High-quality instrumental background music for video, no vocals, no drums solo. "
        f"Mood: {mood}. Energy and tempo: {pacing}. Color feel: {tone}. "
        f"{audio_dir} "
        f"Editorial brief: {concept_snip} "
        f"{narrative} "
        f"User request: {user_prompt[:350]}. "
        f"Polished mix, wide stereo image, suitable as underscore."
    ).strip()


def generate_bgm_file(
    brief: CreativeBrief,
    user_prompt: str,
    video_duration: float,
    output_path: Path,
    edl: Optional[EditDecisionList] = None,
) -> Path:
    """
    Generate a WAV file of background music suited to the brief.

    Duration is capped at MusicGen's ~30s limit; caller should loop in the mixer.
    """
    try:
        import scipy.io.wavfile
        import torch
    except ImportError as e:
        raise RuntimeError(
            "Music generation requires scipy and transformers. "
            "Install: pip install -r requirements-musicgen.txt"
        ) from e

    model_id = cfg("music_generation.model", "facebook/musicgen-large")
    max_seconds = float(cfg("music_generation.max_seconds", 30))
    target = min(max(video_duration * 0.5, 12.0), max_seconds)
    # ~50 tokens per second of audio for MusicGen
    max_new_tokens = min(int(target * 50), 1500)
    guidance_scale = float(cfg("music_generation.guidance_scale", 3.0))

    _load_model(model_id)
    device, _ = _get_device_dtype()

    description = brief_to_music_prompt(brief, user_prompt, edl)
    log.info(f"MusicGen prompt ({len(description)} chars): {description[:160]}…")

    inputs = _processor(text=[description], padding=True, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        audio_values = _model.generate(
            **inputs,
            do_sample=True,
            guidance_scale=guidance_scale,
            max_new_tokens=max_new_tokens,
        )

    sampling_rate = _model.config.audio_encoder.sampling_rate
    wav = _waveform_to_numpy(audio_values)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    peak = abs(wav).max()
    if peak > 1e-8:
        wav = wav / peak
    scipy.io.wavfile.write(str(output_path), sampling_rate, (wav * 32767).astype("int16"))

    log.info(f"Generated BGM: {output_path} (~{len(wav) / sampling_rate:.1f}s @ {sampling_rate} Hz)")
    return output_path


def is_enabled() -> bool:
    return bool(cfg("music_generation.enabled", False))


def _waveform_to_numpy(audio_values) -> Any:
    """Normalize MusicGen output tensors to 1D float waveform."""
    import torch

    t = audio_values if isinstance(audio_values, torch.Tensor) else torch.as_tensor(audio_values)
    t = t.float()

    if t.dim() >= 3:
        t = t[0]
    if t.dim() == 2:
        if t.shape[0] == 1:
            t = t[0]
        elif t.shape[0] == 2:
            t = (t[0] + t[1]) * 0.5
        else:
            t = t[0]
    elif t.dim() != 1:
        t = t.reshape(-1)

    return t.detach().cpu().numpy()
