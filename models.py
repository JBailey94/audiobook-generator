"""
Model discovery and download.

Both ``setup.sh`` and ``cli.py`` need the same two models:

* the F5-TTS v1 Base checkpoint (~1.3 GB), and
* a small Whisper model used to transcribe voice samples (~290 MB).

Keeping the logic here means the first-run experience is identical whether a
user runs the setup script or simply runs a command for the first time.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# F5-TTS checkpoint, mirrored from Hugging Face.
F5_REPO_ID = "SWivid/F5-TTS"
F5_SUBDIR = "F5TTS_v1_Base"
F5_CHECKPOINT = "model_1250000.safetensors"
F5_VOCAB = "vocab.txt"
F5_DOWNLOAD_SIZE_HINT = "~1.3 GB"

# Whisper model used to transcribe reference clips.
WHISPER_MODEL_ID = "openai/whisper-base"

# The neural vocoder F5-TTS uses to turn spectrograms into waveforms. Small,
# but it is a separate download that would otherwise happen mid-generation.
VOCODER_REPO_ID = "charactr/vocos-mel-24khz"
VOCODER_FILES = ("config.yaml", "pytorch_model.bin")

PROJECT_DIR = Path(__file__).resolve().parent


def default_model_dir() -> str:
    """Where the F5-TTS checkpoint lives by default."""
    env = os.environ.get("F5_MODEL_DIR")
    if env:
        return env
    return str(PROJECT_DIR / "f5_model" / F5_SUBDIR)


def cache_dir() -> Path:
    """Directory for derived artefacts (normalised clips, ref text, etc.)."""
    env = os.environ.get("F5_CACHE_DIR")
    return Path(env) if env else PROJECT_DIR / ".cache"


def model_is_downloaded(model_dir: str | os.PathLike) -> bool:
    """True if both the checkpoint and its vocab file are present."""
    model_path = Path(model_dir)
    return (model_path / F5_CHECKPOINT).is_file() and (model_path / F5_VOCAB).is_file()


def ensure_model(
    model_dir: str | os.PathLike | None = None,
    quiet: bool = False,
    show_progress: bool = True,
) -> str:
    """
    Make sure the F5-TTS checkpoint is on disk and return its directory.

    The snapshot is downloaded straight into ``<parent>/f5_model`` using the
    repository's own ``F5TTS_v1_Base/`` layout, so the files land exactly where
    the engine expects them. That also means an interrupted download resumes
    from where it stopped instead of starting over.
    """
    model_path = Path(model_dir) if model_dir else Path(default_model_dir())
    model_path = model_path.expanduser()

    if model_is_downloaded(model_path):
        return str(model_path)

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        raise RuntimeError(
            "huggingface_hub is not installed, so the model cannot be downloaded.\n"
            "Run ./setup.sh to install dependencies and fetch the model."
        )

    # Download into the parent directory: the snapshot contains a
    # 'F5TTS_v1_Base/' prefix that must be preserved.
    target_parent = model_path.parent
    target_parent.mkdir(parents=True, exist_ok=True)

    if not quiet:
        print("=" * 58)
        print("  First run: downloading the F5-TTS model")
        print("=" * 58)
        print(f"  Model  : {F5_REPO_ID} ({F5_SUBDIR})")
        print(f"  Size   : {F5_DOWNLOAD_SIZE_HINT}, may take several minutes")
        print(f"  To     : {model_path}")
        print("")

    snapshot_kwargs = {
        "repo_id": F5_REPO_ID,
        "local_dir": str(target_parent),
        "allow_patterns": [f"{F5_SUBDIR}/*"],
    }
    if not show_progress:
        # tqdm_class is the supported way to mute the download bars.
        snapshot_kwargs["tqdm_class"] = _SilentTqdm

    snapshot_download(**snapshot_kwargs)

    if not model_is_downloaded(model_path):
        raise RuntimeError(
            "The model download finished but the files are missing.\n"
            f"Expected: {model_path / F5_CHECKPOINT}"
        )

    if not quiet:
        print("\n  Model ready.")
    return str(model_path)


def ensure_vocoder(quiet: bool = False) -> bool:
    """Pre-fetch the Vocos vocoder weights used by F5-TTS."""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        return False

    if not quiet:
        print(f"  Downloading vocoder ({VOCODER_REPO_ID})...")
    try:
        for filename in VOCODER_FILES:
            hf_hub_download(repo_id=VOCODER_REPO_ID, filename=filename)
    except Exception as exc:
        if not quiet:
            print(f"  ! Could not pre-download the vocoder: {exc}")
            print("    It will be fetched automatically on first use.")
        return False

    if not quiet:
        print("  Vocoder ready.")
    return True


class _SilentTqdm:

    def __init__(self, *args, **kwargs):
        self.total = kwargs.get("total", 0)

    def update(self, n=1):
        pass

    def close(self):
        pass

    def set_description(self, *args, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def ensure_whisper_model(quiet: bool = False) -> str:
    """
    Pre-fetch the Whisper model used for transcribing voice samples.

    Doing this during setup keeps the first generation run from stalling on a
    surprise download, and surfaces network problems while the user still has
    ./setup.sh in front of them.
    """
    try:
        from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
    except ImportError:
        if not quiet:
            print(
                "  ! transformers is not installed; skipping voice-sample "
                "transcription model.\n    (Run ./setup.sh again to retry.)"
            )
        return ""

    if not quiet:
        print(f"  Downloading speech recognition model ({WHISPER_MODEL_ID})...")

    try:
        AutoProcessor.from_pretrained(WHISPER_MODEL_ID)
        AutoModelForSpeechSeq2Seq.from_pretrained(WHISPER_MODEL_ID)
    except Exception as exc:
        if not quiet:
            print(f"  ! Could not pre-download {WHISPER_MODEL_ID}: {exc}")
            print("    Voice samples will be transcribed on first use instead.")
        return ""

    if not quiet:
        print("  Speech recognition model ready.")
    return WHISPER_MODEL_ID


def human_size(num_bytes: float) -> str:
    """Format a byte count for humans."""
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(num_bytes) < 1024.0:
            return f"{num_bytes:3.1f} {unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.1f} PB"


if __name__ == "__main__":
    # Allow `python3 models.py [model_dir]` for use from setup scripts.
    ensure_model(sys.argv[1] if len(sys.argv) > 1 else None)
    ensure_vocoder()
    ensure_whisper_model()
