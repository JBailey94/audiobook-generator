"""
F5-TTS integration module.

Wraps the upstream ``f5_tts`` engine and adds the bits a command line user
needs: reference audio that always arrives in the format F5-TTS wants, speech
recognition for the reference text that is cached on disk, and audio helpers for
building chapter files and final audiobooks.
"""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import os
import random
import shutil
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

# F5-TTS v1 Base renders at 24 kHz; resampling the reference clip to this rate
# avoids a redundant resample inside the model.
TARGET_SAMPLE_RATE = 24000

# Longest reference clip F5-TTS will use; anything above this is truncated.
MAX_REFERENCE_SECONDS = 12.0

WHISPER_MODEL_ID = "openai/whisper-base"

# Whisper needs 16 kHz input.
_WHISPER_SAMPLE_RATE = 16000


class RefTextError(RuntimeError):
    """Raised when the reference text cannot be determined automatically."""


# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------


def _resolve_ffmpeg() -> str | None:
    return shutil.which("ffmpeg")


def has_ffmpeg() -> bool:
    """True if ffmpeg is on PATH (required by F5-TTS' pydub audio loading)."""
    return _resolve_ffmpeg() is not None


def ffmpeg_missing_message() -> str:
    return (
        "ffmpeg was not found on your PATH.\n"
        "F5-TTS uses ffmpeg to decode reference audio, so it must be installed:\n"
        "  Debian/Ubuntu : sudo apt install ffmpeg\n"
        "  Fedora        : sudo dnf install ffmpeg\n"
        "  Arch          : sudo pacman -S ffmpeg\n"
        "  macOS         : brew install ffmpeg\n"
        "  Windows       : winget install Gyan.FFmpeg"
    )


def prepare_reference_audio(
    audio_path: str,
    target_sr: int = TARGET_SAMPLE_RATE,
    cache_dir: str | None = None,
) -> str:
    """
    Normalise a reference clip to mono PCM WAV at ``target_sr``.

    Accepts any format ``soundfile`` can read (WAV, FLAC, OGG, MP3) so users do
    not have to convert a phone recording by hand. The result is cached next to
    the other cached data so repeat runs skip the work.

    Returns the path to a WAV file. The caller owns the file if it lives in a
    temporary directory.
    """
    source = Path(audio_path)
    if not source.is_file():
        raise FileNotFoundError(f"Voice sample not found: {audio_path}")

    try:
        data, sr = sf.read(str(source), always_2d=False)
    except Exception as exc:
        raise ValueError(
            f"Could not read audio from {source.name}: {exc}\n"
            "WAV, FLAC, OGG and MP3 are supported."
        ) from exc

    if data.size == 0:
        raise ValueError(f"{source.name} contains no audio samples.")

    # Downmix to mono.
    if data.ndim > 1:
        data = data.mean(axis=1)

    # Resample if needed.
    if sr != target_sr:
        data = _resample(data, sr, target_sr)
        sr = target_sr

    if cache_dir is not None:
        key = _file_digest(source)
        out_dir = Path(cache_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        cached = out_dir / f"ref_{key}.wav"
        if not cached.exists():
            sf.write(str(cached), data, sr)
        return str(cached)

    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    sf.write(tmp.name, data, sr)
    return tmp.name


def _resample(data: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Resample a mono float array using whichever backend is available."""
    try:
        import librosa

        return librosa.resample(data, orig_sr=orig_sr, target_sr=target_sr)
    except Exception:
        pass

    # Fallback: linear interpolation via numpy (adequate for speech).
    duration = data.shape[0] / float(orig_sr)
    new_length = int(round(duration * target_sr))
    if new_length <= 0:
        return data
    old_positions = np.linspace(0.0, 1.0, num=data.shape[0], endpoint=False)
    new_positions = np.linspace(0.0, 1.0, num=new_length, endpoint=False)
    return np.interp(new_positions, old_positions, data).astype(np.float32)


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()[:16]


def get_audio_duration(path: str) -> float:
    """Return the duration of an audio file in seconds."""
    try:
        return float(sf.info(path).duration)
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# Reference text (speech recognition)
# ---------------------------------------------------------------------------


def transcribe_audio(
    audio_path: str,
    device: str = "cpu",
    cache_dir: str | None = None,
) -> str:
    """
    Transcribe a voice sample with Whisper so F5-TTS knows what was said.

    Results are cached on disk keyed by the audio content, so the model runs at
    most once per sample. Falls back to F5-TTS' own transcriber only if
    Whisper is unavailable.
    """
    source = Path(audio_path)
    key = _file_digest(source)
    cache_path = None
    if cache_dir is not None:
        cache_path = Path(cache_dir) / f"reftext_{WHISPER_MODEL_ID.replace('/', '_')}_{key}.json"
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        if cache_path.exists():
            try:
                cached = json.loads(cache_path.read_text(encoding="utf-8"))
                text = cached.get("text", "").strip()
                if text:
                    return text
            except (json.JSONDecodeError, OSError):
                pass

    text = _transcribe_with_whisper(str(source), device) or ""

    if not text:
        # f5-tts ships its own ASR fallback (a much larger model), so only reach
        # for it when Whisper genuinely failed.
        try:
            from f5_tts.infer.utils_infer import transcribe as f5_transcribe

            text = (f5_transcribe(str(source)) or "").strip()
        except Exception:
            text = ""

    if not text:
        raise RefTextError(
            "Could not work out what is said in the voice sample.\n"
            "Re-run with the exact text of your sample using:\n"
            '  -r "the words spoken in the sample"'
        )

    if cache_path is not None:
        try:
            cache_path.write_text(json.dumps({"text": text}), encoding="utf-8")
        except OSError:
            pass
    return text


def _transcribe_with_whisper(audio_path: str, device: str) -> str:
    """Run Whisper on a reference clip. Returns "" if anything goes wrong."""
    import torch

    try:
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
    except Exception as exc:
        raise RefTextError(
            "The 'transformers' package is required to transcribe voice samples.\n"
            "Re-run ./setup.sh to repair the installation."
        ) from exc

    # Transformers is chatty about API deprecations; none of it is actionable
    # for someone who just wants to hear their text read aloud.
    try:
        from transformers import logging as hf_logging

        hf_logging.set_verbosity_error()
    except Exception:
        pass

    use_cuda = device.startswith("cuda") and torch.cuda.is_available()
    target_device = "cuda" if use_cuda else "cpu"
    dtype = torch.float16 if use_cuda else torch.float32

    try:
        processor = AutoProcessor.from_pretrained(WHISPER_MODEL_ID)
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            WHISPER_MODEL_ID, dtype=dtype, low_cpu_mem_usage=True
        )
        model.to(target_device)
    except TypeError:
        # Older transformers releases only accept torch_dtype.
        try:
            processor = AutoProcessor.from_pretrained(WHISPER_MODEL_ID)
            model = AutoModelForSpeechSeq2Seq.from_pretrained(
                WHISPER_MODEL_ID, torch_dtype=dtype, low_cpu_mem_usage=True
            )
            model.to(target_device)
        except Exception as exc:
            raise RefTextError(
                f"Could not load the speech recognition model ({WHISPER_MODEL_ID}): {exc}\n"
                "Check your internet connection, then re-run ./setup.sh."
            ) from exc
    except Exception as exc:
        raise RefTextError(
            f"Could not load the speech recognition model ({WHISPER_MODEL_ID}): {exc}\n"
            "Check your internet connection, then re-run ./setup.sh."
        ) from exc

    audio_data, sample_rate = sf.read(audio_path)
    if audio_data.ndim > 1:
        audio_data = audio_data.mean(axis=1)
    if sample_rate != _WHISPER_SAMPLE_RATE:
        audio_data = _resample(audio_data, sample_rate, _WHISPER_SAMPLE_RATE)

    inputs = processor(
        audio_data, sampling_rate=_WHISPER_SAMPLE_RATE, return_tensors="pt"
    )
    input_features = inputs.input_features.to(target_device)
    if use_cuda:
        input_features = input_features.to(torch.float16)

    with torch.no_grad():
        # task="transcribe" stops Whisper defaulting to translation.
        try:
            predicted_ids = model.generate(
                input_features, task="transcribe", max_new_tokens=128
            )
        except TypeError:
            predicted_ids = model.generate(input_features, max_new_tokens=128)

    text = processor.batch_decode(predicted_ids, skip_special_tokens=True)
    return (text[0] if text else "").strip()


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class F5TTSEngine:
    """Thin wrapper around the upstream F5-TTS API."""

    def __init__(
        self,
        model_dir: str,
        device: str = "cuda",
        model_name: str = "F5TTS_v1_Base",
        quiet: bool = False,
    ):
        from f5_tts.api import F5TTS

        _clamp_seed_everything()

        ckpt_file = os.path.join(model_dir, "model_1250000.safetensors")
        vocab_file = os.path.join(model_dir, "vocab.txt")

        if not os.path.exists(ckpt_file):
            raise FileNotFoundError(
                f"Model checkpoint not found: {ckpt_file}\n"
                "Run ./setup.sh to download it."
            )
        if not os.path.exists(vocab_file):
            raise FileNotFoundError(
                f"Vocab file not found: {vocab_file}\nRun ./setup.sh to download it."
            )

        self.quiet = quiet
        # Passing show_info/progress keeps the engine's chatty per-call output
        # out of the way so the CLI can print its own progress line.
        show_info = (lambda *a, **k: None) if quiet else print

        if quiet:
            # F5TTS.__init__ also prints (model paths, vocoder download) with
            # bare prints, so silence the constructor too.
            with contextlib.redirect_stdout(io.StringIO()):
                self.tts = F5TTS(
                    model=model_name,
                    ckpt_file=ckpt_file,
                    vocab_file=vocab_file,
                    device=device,
                )
        else:
            self.tts = F5TTS(
                model=model_name,
                ckpt_file=ckpt_file,
                vocab_file=vocab_file,
                device=device,
            )

        self.device = device
        self._show_info = show_info
        self._progress = _NULL_PROGRESS if quiet else None

    def generate(
        self,
        ref_audio: str,
        ref_text: str,
        gen_text: str,
        output_path: str,
        speed: float = 1.0,
        remove_silence: bool = True,
        seed: int | None = None,
    ) -> str:
        """Generate audio for one text chunk and save it to ``output_path``."""
        kwargs = {}
        if self._progress is not None:
            kwargs["progress"] = self._progress

        # Pick the seed up front so the value handed to infer() is ours, not the
        # engine's out-of-range default. See _clamp_seed_everything.
        seed = _pick_seed(seed)

        infer_kwargs = dict(
            ref_file=ref_audio,
            ref_text=ref_text,
            gen_text=gen_text,
            file_wave=output_path,
            speed=speed,
            remove_silence=remove_silence,
            show_info=self._show_info,
            seed=seed,
            **kwargs,
        )

        if self.quiet:
            # f5-tts prints progress with bare print() calls in places, so
            # show_info alone is not enough to keep --quiet quiet. Buffer it,
            # and surface the text if something actually goes wrong.
            captured = io.StringIO()
            try:
                with contextlib.redirect_stdout(captured):
                    self.tts.infer(**infer_kwargs)
            except Exception as exc:
                _sanitize_pythonhashseed()
                detail = captured.getvalue().strip()
                if detail:
                    raise RuntimeError(f"{exc}\n\nF5-TTS output:\n{detail}") from exc
                raise
        else:
            self.tts.infer(**infer_kwargs)

        _sanitize_pythonhashseed()
        return output_path


class _NullProgress:
    """
    Stand-in for the ``tqdm`` module.

    f5-tts calls ``progress.tqdm(iterable)``; this returns the iterable
    untouched so ``--quiet`` produces no progress bars.
    """

    @staticmethod
    def tqdm(iterable=None, *args, **kwargs):
        if iterable is None:
            return iter(())
        return iterable


_NULL_PROGRESS = _NullProgress()

# CPython only accepts PYTHONHASHSEED in 0..2**32-1. f5-tts' seed_everything()
# copies the inference seed into that variable, and infer() defaults that seed
# to random.randint(0, sys.maxsize) - far outside the legal range. Any Python
# process spawned afterwards then dies with "Fatal Python error:
# config_init_hash_seed: PYTHONHASHSEED must be ...". CPU inference spawns
# multiprocessing helpers, so the crash shows up on machines without a GPU.
_MAX_SEED = 2**32 - 1


def _pick_seed(seed: int | None) -> int:
    """Return a seed CPython will accept, randomising when none was given."""
    if seed is None:
        return random.randint(0, _MAX_SEED)
    return max(0, min(int(seed), _MAX_SEED))


def _clamp_seed_everything() -> None:
    """
    Stop f5-tts from poisoning PYTHONHASHSEED.

    ``f5_tts.api`` imports ``seed_everything`` by name, so the replacement has
    to land on the module it was copied from *and* on the copy.
    """
    try:
        from f5_tts import api as f5_api
        from f5_tts.model import utils as f5_utils
    except ImportError:
        return

    original = getattr(f5_utils, "seed_everything", None)
    if original is None or getattr(original, "_f5_cli_clamped", False):
        return

    def seed_everything(seed=0, *args, **kwargs):
        return original(_pick_seed(seed), *args, **kwargs)

    seed_everything._f5_cli_clamped = True
    f5_utils.seed_everything = seed_everything
    if hasattr(f5_api, "seed_everything"):
        f5_api.seed_everything = seed_everything


def _sanitize_pythonhashseed() -> None:
    """Drop an out-of-range PYTHONHASHSEED so child interpreters still start."""
    value = os.environ.get("PYTHONHASHSEED")
    if value is None:
        return
    try:
        seed = int(value)
    except ValueError:
        seed = -1
    if seed < 0 or seed > _MAX_SEED:
        del os.environ["PYTHONHASHSEED"]


# ---------------------------------------------------------------------------
# Combining audio
# ---------------------------------------------------------------------------


def combine_audio_files(input_paths: list, output_path: str) -> str:
    """Concatenate WAV files that share a sample rate."""
    if not input_paths:
        raise ValueError("No audio files to combine")

    sample_rate = None
    audio_chunks = []
    for path in input_paths:
        data, sr = sf.read(path)
        if data.ndim > 1:
            data = data.mean(axis=1)
        if sample_rate is None:
            sample_rate = sr
        elif sr != sample_rate:
            data = _resample(data, sr, sample_rate)
        audio_chunks.append(data)

    parent = os.path.dirname(os.path.abspath(output_path))
    if parent:
        os.makedirs(parent, exist_ok=True)

    sf.write(output_path, np.concatenate(audio_chunks), sample_rate)
    return output_path
