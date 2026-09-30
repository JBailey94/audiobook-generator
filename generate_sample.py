#!/usr/bin/env python3
"""
Generate a short example WAV file using F5-TTS with a voice sample and text.

Kept for backwards compatibility. ./run.sh (cli.py) does the same thing and is
what the README documents; this script now delegates to the same helpers so
both paths behave identically:

    ./run.sh -t "Hello there" -v voices/jason_sample.wav -o hello.wav
"""

import argparse
import os
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import models as model_manager
from tts import F5TTSEngine, RefTextError, prepare_reference_audio, transcribe_audio


def default_device() -> str:
    try:
        import torch
    except ImportError:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def generate_sample(
    voice_sample: str,
    text: str,
    output_path: str,
    model_dir: str,
    ref_text: str | None = None,
    device: str = "cpu",
    speed: float = 1.0,
    remove_silence: bool = True,
):
    """Generate a short WAV file using F5-TTS."""
    cache_dir = str(model_manager.cache_dir())

    ref_audio = prepare_reference_audio(voice_sample, cache_dir=cache_dir)

    if ref_text is None:
        print("==> No reference text provided, transcribing voice sample...")
        try:
            ref_text = transcribe_audio(ref_audio, device=device, cache_dir=cache_dir)
        except RefTextError as exc:
            print(f"ERROR: {exc}")
            sys.exit(1)
        print(f"==> Reference text: {ref_text[:100]}...")

    print("==> Loading F5-TTS model...")
    engine = F5TTSEngine(model_dir=model_dir, device=device)

    print(f"==> Generating audio for: {text[:80]}...")
    engine.generate(
        ref_audio=ref_audio,
        ref_text=ref_text,
        gen_text=text,
        output_path=output_path,
        speed=speed,
        remove_silence=remove_silence,
    )

    print(f"==> Audio saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate a short example WAV file using F5-TTS with a voice sample and text."
    )
    parser.add_argument(
        "--voice", "-v",
        required=True,
        help="Path to the voice sample WAV file",
    )
    parser.add_argument(
        "--text", "-t",
        required=True,
        help="Text to synthesize into speech",
    )
    parser.add_argument(
        "--output", "-o",
        default="generated.wav",
        help="Output WAV file path (default: generated.wav)",
    )
    parser.add_argument(
        "--ref-text", "-r",
        default=None,
        help="Reference text (transcription of voice sample). Auto-transcribed if not provided.",
    )
    parser.add_argument(
        "--model-dir",
        default=model_manager.default_model_dir(),
        help="Path to the F5-TTS model directory",
    )
    parser.add_argument(
        "--device",
        default=default_device(),
        help="Device to use (cpu, cuda or mps)",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="Speech speed factor (default: 1.0)",
    )
    parser.add_argument(
        "--no-remove-silence",
        action="store_true",
        help="Disable silence removal from output",
    )

    args = parser.parse_args()

    # Validate inputs
    if not os.path.exists(args.voice):
        print(f"Error: Voice sample not found: {args.voice}")
        sys.exit(1)

    model_dir = model_manager.ensure_model(args.model_dir)
    if not model_manager.model_is_downloaded(model_dir):
        print(f"Error: Model directory not found: {model_dir}")
        sys.exit(1)

    # Create output directory if needed
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    generate_sample(
        voice_sample=args.voice,
        text=args.text,
        output_path=args.output,
        model_dir=model_dir,
        ref_text=args.ref_text,
        device=args.device,
        speed=args.speed,
        remove_silence=not args.no_remove_silence,
    )


if __name__ == "__main__":
    main()