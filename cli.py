#!/usr/bin/env python3
"""
F5-TTS CLI - Generate speech from text with voice cloning.

Works with direct text or a whole book (PDF, EPUB, Markdown, TXT). Everything
needed to run - Python packages, the F5-TTS checkpoint and the speech
recognition model - is downloaded automatically the first time it is used.
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

VOICES_DIR = PROJECT_DIR / "voices"

# Set from --quiet. Status/progress output goes through log(); errors and the
# final result line always print so the user is never left guessing.
_QUIET = False


def log(*parts) -> None:
    """Print status output unless --quiet was requested."""
    if not _QUIET:
        print(*parts)


try:
    import torch
except ImportError:  # pragma: no cover - handled with a friendly message
    print(
        "ERROR: PyTorch is not installed.\n"
        "Run ./setup.sh to install everything this tool needs."
    )
    sys.exit(1)

import models as model_manager
from tts import (
    F5TTSEngine,
    MAX_REFERENCE_SECONDS,
    RefTextError,
    combine_audio_files,
    ffmpeg_missing_message,
    get_audio_duration,
    has_ffmpeg,
    prepare_reference_audio,
    transcribe_audio,
)
from text_processing import process_sections

# Measured seconds per ~100-character chunk, used for the "this will take a
# while" warning before any audio exists. Real numbers replace these after the
# first chunk. A chunk is roughly 6 seconds of speech, so on CPU the tool runs
# about 20x slower than real time; that is inherent to the model, not a bug.
ROUGH_SECONDS_PER_CHUNK = {"cuda": 3.5, "mps": 20.0, "cpu": 140.0}

AUDIOBOOK_FORMATS = (".m4b", ".mp3")


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def format_duration(seconds: float) -> str:
    seconds = int(max(seconds, 0))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {secs:02d}s"
    if minutes:
        return f"{minutes}m {secs:02d}s"
    return f"{secs}s"


def safe_filename(name: str, fallback: str = "Chapter") -> str:
    """Turn a chapter title into a filesystem-friendly name."""
    cleaned = re.sub(r'[\\/:*?"<>|\r\n\t]+', " ", name or "")
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" .")
    cleaned = cleaned.encode("ascii", "ignore").decode().strip()
    if not cleaned:
        cleaned = fallback
    return cleaned[:80]


def available_voices() -> list[Path]:
    """Voice samples the user has placed in voices/."""
    if not VOICES_DIR.is_dir():
        return []
    exts = {".wav", ".flac", ".ogg", ".mp3", ".m4a"}
    return sorted(p for p in VOICES_DIR.iterdir() if p.suffix.lower() in exts)


NO_VOICE_HELP = """\
F5-TTS clones a voice, so it needs a recording of the voice you want to use:
5-10 seconds of one person talking clearly, with no music.

  Clone one from a YouTube video:
    ./get_voice_sample.sh "https://www.youtube.com/watch?v=VIDEO_ID" 1:30 1:38 my_voice.wav

  Or record one yourself (a phone voice memo is fine) and point at it:
    ./run.sh -t "Hello there" -v my_voice.wav -o hello.wav

  Drop a sample into voices/ and it is picked up automatically:
    ./run.sh -t "Hello there" -o hello.wav
"""


def resolve_voice(path: str | None) -> Path:
    """Resolve the voice sample to clone, guiding the user when none is given."""
    if path:
        resolved = Path(path).expanduser()
        if not resolved.is_file():
            raise FileNotFoundError(f"Voice sample not found: {resolved}")
        return resolved

    voices = available_voices()
    if voices:
        log(f"==> No -v given, using the voice sample in voices/: {voices[0].name}")
        return voices[0]

    print("ERROR: no voice sample was given.\n")
    print(NO_VOICE_HELP)
    sys.exit(1)


def confirm(question: str, assume_yes: bool) -> bool:
    if assume_yes:
        return True
    if not sys.stdin.isatty():
        return True
    try:
        answer = input(f"{question} [Y/n] ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print()
        return False
    return answer in ("", "y", "yes")


# ---------------------------------------------------------------------------
# doctor / list-voices
# ---------------------------------------------------------------------------


def list_voices() -> int:
    voices = available_voices()
    if not voices:
        print("No voice samples are in voices/.")
        print(NO_VOICE_HELP)
        return 1
    print("Voice samples in voices/ (these need no full path):\n")
    for voice in voices:
        duration = get_audio_duration(str(voice))
        print(f"  {voice.name:<28} {duration:5.1f}s")
    print(f"\n  ./run.sh -t \"Hello there\" -v {voices[0].name} -o hello.wav")
    return 0


def _check(label: str, ok: bool, detail: str = "", required: bool = True) -> bool:
    mark = "OK " if ok else ("ERR" if required else "-- ")
    if detail:
        dots = "." * max(3, 26 - len(label))
        print(f"  [{mark}] {label} {dots} {detail}")
    else:
        print(f"  [{mark}] {label}")
    return ok


def run_doctor() -> int:
    """Check that everything this tool needs is present and working."""
    print("=" * 58)
    print("  F5-TTS environment check")
    print("=" * 58)
    print()

    print("Core")
    _check("Python", True, sys.version.split()[0])

    torch_ok = _check("PyTorch", True, torch.__version__)

    cuda = torch.cuda.is_available()
    if cuda:
        gpu = torch.cuda.get_device_name(0)
        _check("GPU (CUDA)", True, gpu)
        device = "cuda"
    else:
        mps = getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
        if mps:
            _check("GPU (Apple Metal)", True, "mps")
            device = "mps"
        else:
            _check("GPU", False, "not detected - running on CPU is much slower", required=False)
            device = "cpu"
    print()

    print("System tools")
    if has_ffmpeg():
        _check("ffmpeg", True, shutil.which("ffmpeg"))
        ffmpeg_ok = True
    else:
        ffmpeg_ok = _check("ffmpeg", False, "required to read voice samples")
    print()

    print("Engine")
    engine_ok, engine_detail = _check_engine()
    if engine_ok:
        _check("F5-TTS engine", True, "ready to generate audio")
    else:
        _check("F5-TTS engine", False, engine_detail)
    print()

    print("Models")
    model_dir = model_manager.default_model_dir()
    model_ok = model_manager.model_is_downloaded(model_dir)
    if model_ok:
        size = sum(
            f.stat().st_size for f in Path(model_dir).iterdir() if f.is_file()
        )
        _check("F5-TTS checkpoint", True, f"{model_manager.human_size(size)} in {model_dir}")
    else:
        _check("F5-TTS checkpoint", False, f"missing in {model_dir} (run ./setup.sh)")

    whisper_ok = _download_state(model_manager.WHISPER_MODEL_ID)
    if whisper_ok:
        _check("Whisper (voice samples)", True, f"{model_manager.WHISPER_MODEL_ID} cached")
    else:
        _check(
            "Whisper (voice samples)",
            False,
            f"{model_manager.WHISPER_MODEL_ID} not cached (downloaded on demand)",
            required=False,
        )
    print()

    print("Voices")
    voices = available_voices()
    if voices:
        _check("Voice samples", True, f"{len(voices)} found in voices/")
    else:
        _check(
            "Voice samples",
            False,
            "none yet - create one with ./get_voice_sample.sh",
            required=False,
        )
    print()

    print("=" * 58)
    critical_ok = torch_ok and ffmpeg_ok and model_ok and engine_ok
    if critical_ok:
        print("  Ready to use.  Example:")
        print('    ./run.sh -t "Hello, this is a test." -v my_voice.wav -o hello.wav')
        if not whisper_ok:
            print("    (first run will download the speech recognition model)")
        if not voices:
            print()
            print("  No voice sample yet - get your own with ./get_voice_sample.sh")
    else:
        print("  Not ready yet.  Fix the [ERR] items above, or just run:")
        print("    ./setup.sh")
        if not engine_ok:
            print("")
            print("  A broken engine usually means a mismatched PyTorch install.")
            print("  ./setup.sh --recreate rebuilds it from scratch.")
    print("=" * 58)
    return 0 if critical_ok else 1


def _check_engine() -> tuple[bool, str]:
    """
    Import the pieces F5-TTS needs at runtime.

    This catches broken installs - a CUDA torchcodec inside a CPU-only
    environment is the common one - and reports the real reason instead of
    letting it blow up halfway through generating an audiobook.
    """
    try:
        import f5_tts.api  # noqa: F401
        import torchcodec  # noqa: F401
    except Exception as exc:
        # The useful message is usually on the innermost cause, e.g.
        # "libnvrtc.so.13: cannot open shared object file".
        message = f"{type(exc).__name__}: {exc}"
        seen = set()
        cause = exc.__cause__ or exc.__context__
        while cause is not None and id(cause) not in seen:
            seen.add(id(cause))
            if str(cause).strip():
                message = f"{type(cause).__name__}: {cause}"
            cause = cause.__cause__ or cause.__context__
        return False, message
    return True, ""


def _download_state(model_id: str) -> bool:
    """True if a Hugging Face model is already in the local cache."""
    try:
        from huggingface_hub import try_to_load_from_cache
    except ImportError:
        return False
    try:
        return try_to_load_from_cache(model_id, "config.json") is not None
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Reference audio
# ---------------------------------------------------------------------------


def prepare_reference(voice: Path) -> tuple[str, float]:
    """Normalise the voice sample and report its duration."""
    duration = get_audio_duration(str(voice))

    if duration <= 0:
        raise ValueError(f"Could not read audio from {voice.name}")
    if duration > MAX_REFERENCE_SECONDS:
        log(
            f"==> Note: {voice.name} is {duration:.1f}s long. F5-TTS only uses the "
            f"first {MAX_REFERENCE_SECONDS:.0f}s; a 5-10s clip gives the best results."
        )

    ref_audio = prepare_reference_audio(str(voice), cache_dir=str(model_manager.cache_dir()))
    return ref_audio, duration


def prepare_ref_text(args, ref_audio: str) -> str:
    """Work out the text spoken in the reference clip."""
    if args.ref_text:
        return args.ref_text

    log("==> Listening to the voice sample to work out what it says...")
    try:
        text = transcribe_audio(
            ref_audio, device=args.device, cache_dir=str(model_manager.cache_dir())
        )
    except RefTextError as exc:
        print(f"\nERROR: {exc}")
        sys.exit(1)

    log(f"==> Voice sample says: {text[:120]}")
    return text


# ---------------------------------------------------------------------------
# Single text
# ---------------------------------------------------------------------------


def generate_from_text(args) -> int:
    """Generate audio from direct text input."""
    voice = resolve_voice(args.voice)
    ref_audio, _ = prepare_reference(voice)
    ref_text = prepare_ref_text(args, ref_audio)

    model_dir = model_manager.ensure_model(
        args.model_dir, quiet=args.quiet, show_progress=not args.quiet
    )
    engine = F5TTSEngine(model_dir=model_dir, device=args.device, quiet=args.quiet)

    log(f"==> Generating speech: {args.text[:80]}")
    started = time.time()
    engine.generate(
        ref_audio=ref_audio,
        ref_text=ref_text,
        gen_text=args.text,
        output_path=args.output,
        speed=args.speed,
        remove_silence=not args.no_remove_silence,
    )
    print(f"==> Saved to: {args.output} ({format_duration(time.time() - started)})")
    return 0


# ---------------------------------------------------------------------------
# Books
# ---------------------------------------------------------------------------


class ProgressTracker:
    """Tracks throughput so the user sees a live ETA."""

    def __init__(self, total: int, device: str):
        self.total = total
        self.done = 0
        self.device = device
        self.started = time.time()
        self.elapsed_chunks = 0.0

    def record(self, seconds: float) -> None:
        self.done += 1
        self.elapsed_chunks += seconds

    @property
    def per_chunk(self) -> float:
        return self.elapsed_chunks / self.done if self.done else 0.0

    @property
    def remaining(self) -> float:
        return max(self.total - self.done, 0) * self.per_chunk

    def line(self, prefix: str) -> str:
        avg = self.per_chunk
        eta = f", ~{format_duration(self.remaining)} left" if avg else ""
        return f"{prefix} [{self.done}/{self.total}]{eta}"


def plan_chunks(args) -> tuple[list[tuple[str, list[str]]], int]:
    """Split the input book into titled sections of TTS-ready chunks."""
    log(f"==> Reading {Path(args.input_file).name}")
    sections = process_sections(
        args.input_file,
        lang=args.language,
        max_chars=args.max_chars,
        target_chars=args.section_chars,
        min_chars=args.min_section_chars,
    )
    log(f"==> Found {len(sections)} chapters, {sum(len(c) for _, c in sections)} chunks")

    if args.start_chapter:
        sections = sections[args.start_chapter :]
        log(f"==> Skipping the first {args.start_chapter} chapters")
    if args.max_chapters is not None:
        sections = sections[: args.max_chapters]
        log(f"==> Limiting to the first {len(sections)} chapters")

    # Drop empty sections, then apply chunk-level limits across the whole book.
    sections = [(title, chunks) for title, chunks in sections if chunks]

    if args.start_chunk:
        remaining = args.start_chunk
        trimmed: list[tuple[str, list[str]]] = []
        for title, chunks in sections:
            if remaining >= len(chunks):
                remaining -= len(chunks)
                continue
            trimmed.append((title, chunks[remaining:]))
            remaining = 0
        sections = trimmed
        log(f"==> Skipping the first {args.start_chunk} chunks")

    if args.max_chunks is not None:
        budget = args.max_chunks
        trimmed = []
        for title, chunks in sections:
            if budget <= 0:
                break
            trimmed.append((title, chunks[:budget]))
            budget -= len(chunks[:budget])
        sections = trimmed
        log(f"==> Limiting to {args.max_chunks} chunks")

    return sections, sum(len(chunks) for _, chunks in sections)


def show_dry_run(args) -> int:
    """Preview the plan without downloading models or generating audio."""
    sections, total = plan_chunks(args)

    # plan_chunks() already applied --max-chunks, so if it was given, list the
    # exact chunks that would be generated instead of a per-chapter summary.
    if args.max_chunks is not None:
        print("\nChunks:")
        shown = 0
        for title, chunks in sections:
            for chunk in chunks:
                if shown >= args.max_chunks:
                    break
                print(f"  [{shown:>4}] ({len(chunk):>3} chars) {chunk[:70]}")
                shown += 1
            if shown >= args.max_chunks:
                break
    else:
        print("\nChapters:")
        for index, (title, chunks) in enumerate(sections, 1):
            chars = sum(len(c) for c in chunks)
            print(f"  {index:>3}. {title}  ({len(chunks)} chunks, ~{chars:,} chars)")

    rough = ROUGH_SECONDS_PER_CHUNK.get(args.device, 6.0)
    print(f"\nTotal: {total} chunks")
    print(
        f"Estimated time on {args.device.upper()}: about "
        f"{format_duration(total * rough)}"
    )
    print("(nothing generated - this was a --dry-run)")
    return 0


def generate_book(args) -> int:
    """Render a whole book to chapter files, then optionally an audiobook."""
    # Resolve the voice before the estimate/confirmation so the user is told
    # about it before committing to a long job.
    voice = resolve_voice(args.voice)
    sections, total = plan_chunks(args)
    if not sections:
        print("ERROR: nothing to generate.")
        return 1

    rough = ROUGH_SECONDS_PER_CHUNK.get(args.device, 6.0)
    print(
        f"\n==> This will generate {total} chunks of audio, roughly "
        f"{format_duration(total * rough)} on {args.device.upper()}."
    )
    if not confirm("Start generation?", args.yes):
        print("Cancelled.")
        return 1

    ref_audio, _ = prepare_reference(voice)
    ref_text = prepare_ref_text(args, ref_audio)

    model_dir = model_manager.ensure_model(
        args.model_dir, quiet=args.quiet, show_progress=not args.quiet
    )
    engine = F5TTSEngine(model_dir=model_dir, device=args.device, quiet=args.quiet)

    output_path = Path(args.output)
    chapter_dir = (
        Path(args.chapter_dir)
        if args.chapter_dir
        else output_path.parent / f"{output_path.stem}_chapters"
    )
    chunk_root = Path(args.output_dir) if args.output_dir else chapter_dir / ".chunks"
    chapter_dir.mkdir(parents=True, exist_ok=True)
    chunk_root.mkdir(parents=True, exist_ok=True)

    tracker = ProgressTracker(total, args.device)
    started = time.time()
    used_names: set[str] = set()
    chapter_files: list[tuple[str, Path]] = []

    for index, (title, chunks) in enumerate(sections, 1):
        slug = safe_filename(title, fallback=f"Chapter {index}")
        while slug.lower() in used_names:
            slug = f"{slug} ({index})"
        used_names.add(slug.lower())

        chapter_path = chapter_dir / f"{index:03d} - {slug}.wav"
        chunk_dir = chunk_root / f"{index:03d}"
        chunk_dir.mkdir(parents=True, exist_ok=True)

        prefix = f"Chapter {index}/{len(sections)}: {slug[:38]}"
        pending: list[Path] = []

        for offset, chunk in enumerate(chunks):
            chunk_path = chunk_dir / f"chunk_{offset:04d}.wav"

            if args.resume and chunk_path.is_file() and chunk_path.stat().st_size > 44:
                tracker.done += 1
                pending.append(chunk_path)
                log(tracker.line(prefix))
                continue

            chunk_started = time.time()
            engine.generate(
                ref_audio=ref_audio,
                ref_text=ref_text,
                gen_text=chunk,
                output_path=str(chunk_path),
                speed=args.speed,
                remove_silence=not args.no_remove_silence,
            )
            tracker.record(time.time() - chunk_started)
            pending.append(chunk_path)
            log(tracker.line(prefix), f"| {chunk[:44]}")

        if pending:
            combine_audio_files([str(p) for p in pending], str(chapter_path))
        chapter_files.append((title, chapter_path))
        log(f"    chapter saved: {chapter_path.name}")

    if not args.keep_chunks:
        shutil.rmtree(chunk_root, ignore_errors=True)

    print(
        f"\n==> {len(chapter_files)} chapters in {chapter_dir} "
        f"({format_duration(time.time() - started)} elapsed)"
    )

    if output_path.suffix.lower() == ".wav":
        if args.combine:
            log(f"==> Combining chapters into {output_path.name}...")
            combine_audio_files([str(p) for _, p in chapter_files], str(output_path))
        print(f"==> Saved to: {output_path}")
        return 0

    if args.combine:
        log(
            "==> Note: --combine only applies to .wav output; "
            f"{output_path.name} already contains every chapter."
        )

    log(f"==> Building audiobook: {output_path.name}")
    build_audiobook(chapter_files, output_path, title=args.book_title)
    print(f"==> Saved to: {output_path}")
    return 0


def build_audiobook(chapter_files, output_path: Path, title: str | None = None) -> Path:
    """
    Mux chapter files into a single m4b/mp3, embedding chapter markers.
    """
    suffix = output_path.suffix.lower()
    if suffix not in AUDIOBOOK_FORMATS:
        raise ValueError(f"Unsupported audiobook format: {suffix}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    list_file = output_path.parent / f".{output_path.stem}_concat.txt"
    meta_file = output_path.parent / f".{output_path.stem}_chapters.txt"

    durations = []
    with open(list_file, "w", encoding="utf-8") as handle:
        for _, path in chapter_files:
            resolved = path.resolve()
            escaped = str(resolved).replace("'", "'\\''")
            handle.write(f"file '{escaped}'\n")
            durations.append(get_audio_duration(str(resolved)))

    total_ms = 0
    with open(meta_file, "w", encoding="utf-8") as handle:
        handle.write(";FFMETADATA1\n")
        if title:
            handle.write(f"title={title}\n")
        for (chapter_title, _), seconds in zip(chapter_files, durations):
            start_ms = total_ms
            total_ms += int(seconds * 1000)
            handle.write("[CHAPTER]\n")
            handle.write("TIMEBASE=1/1000\n")
            handle.write(f"START={start_ms}\n")
            handle.write(f"END={total_ms}\n")
            handle.write(f"title={chapter_title.replace(chr(10), ' ')}\n")

    if suffix == ".mp3":
        try:
            convert_audio_from_list(
                str(list_file), str(meta_file), str(output_path), "libmp3lame", "-q:a", "2"
            )
        except RuntimeError:
            # Some ffmpeg builds ship without libmp3lame; fall back to the
            # default MP3 encoder rather than failing outright.
            convert_audio_from_list(
                str(list_file), str(meta_file), str(output_path), "mp3", "-q:a", "2"
            )
    else:
        convert_audio_from_list(str(list_file), str(meta_file), str(output_path), "aac", "-b:a", "128k")

    list_file.unlink(missing_ok=True)
    meta_file.unlink(missing_ok=True)
    return output_path


def convert_audio_from_list(
    list_file: str, meta_file: str, output_path: str, codec: str, *extra: str
) -> str:
    """Concatenate via an ffmpeg concat list, attaching ffmetadata chapters."""
    import subprocess

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError(ffmpeg_missing_message())

    cmd = [
        ffmpeg, "-y", "-loglevel", "error",
        "-f", "concat", "-safe", "0", "-i", list_file,
    ]
    if Path(meta_file).is_file():
        cmd += ["-f", "ffmetadata", "-i", meta_file, "-map_metadata", "1", "-map_chapters", "1"]
    cmd += ["-map", "0:a", "-c:a", codec, *extra, output_path]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg could not build {Path(output_path).name}:\n{result.stderr.strip()}"
        )
    return output_path


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="f5tts",
        description="Generate speech with voice cloning from text, PDFs, EPUBs, Markdown or TXT.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Get a voice to clone out of a YouTube video, then speak with it
  ./get_voice_sample.sh "https://www.youtube.com/watch?v=VIDEO_ID" 1:30 1:38 my_voice.wav
  ./run.sh -t "Hello" -v my_voice.wav -o hello.wav

  # Make an audiobook, one file per chapter, plus a single .m4b
  ./run.sh -f book.epub -v my_voice.wav -o book.m4b

  # Preview the plan first (no models or voice sample needed)
  ./run.sh -f book.pdf --dry-run

  # Stop and continue later
  ./run.sh -f book.epub -v my_voice.wav -o book.m4b --resume --yes

  # Check the installation
  ./run.sh --doctor
        """,
    )

    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("-t", "--text", help="Text to speak directly")
    mode.add_argument("-f", "--input-file", help="Book to narrate (PDF, EPUB, MD, TXT)")
    mode.add_argument(
        "--doctor",
        action="store_true",
        help="Check that everything is installed and report problems",
    )
    mode.add_argument(
        "--list-voices",
        action="store_true",
        help="List the voice samples in voices/",
    )

    parser.add_argument(
        "-v",
        "--voice",
        help="Voice sample to clone (WAV/FLAC/OGG/MP3). A file in voices/ "
        "can be given by name; otherwise it must be created first, e.g. "
        "with ./get_voice_sample.sh.",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="output.wav",
        help="Output file. Use .m4b or .mp3 for an audiobook. (default: output.wav)",
    )
    parser.add_argument(
        "-r",
        "--ref-text",
        help="Exact words spoken in the voice sample. Detected automatically if omitted.",
    )
    parser.add_argument(
        "--model-dir",
        default=model_manager.default_model_dir(),
        help="Where the F5-TTS model lives (default: f5_model/F5TTS_v1_Base)",
    )
    parser.add_argument(
        "--device",
        default=default_device(),
        help="cpu, cuda or mps (default: auto-detected)",
    )

    parser.add_argument(
        "--speed", type=float, default=1.0, help="Speech speed factor (default: 1.0)"
    )
    parser.add_argument(
        "--no-remove-silence", action="store_true", help="Keep silence in the output"
    )
    parser.add_argument(
        "--language", default="eng", help="Language code of the text (default: eng)"
    )
    parser.add_argument(
        "--max-chars", type=int, default=None, help="Characters per chunk (default: auto)"
    )

    # Book options
    parser.add_argument(
        "--chapters",
        dest="chapter_dir",
        default=None,
        metavar="DIR",
        help="Directory for the per-chapter WAV files "
        "(default: <output>_chapters). Audio is always written per chapter.",
    )
    parser.add_argument(
        "--section-chars",
        type=int,
        default=None,
        help="Target characters per chapter when the book has no structure (default: 3000)",
    )
    parser.add_argument(
        "--min-section-chars",
        type=int,
        default=None,
        help="Skip sections shorter than this (default: 120). Use 0 to keep everything.",
    )
    parser.add_argument(
        "--book-title", default=None, help="Title to embed in the m4b/mp3"
    )
    parser.add_argument(
        "--combine",
        action="store_true",
        help="Also write one WAV containing every chapter (default: only for .wav output)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse chunks already generated instead of starting over",
    )
    parser.add_argument(
        "--keep-chunks", action="store_true", help="Keep the intermediate chunk files"
    )
    parser.add_argument(
        "--output-dir", default=None, help="Directory for intermediate chunk files"
    )
    parser.add_argument(
        "--max-chunks", type=int, default=None, help="Only generate this many chunks"
    )
    parser.add_argument(
        "--start-chunk", type=int, default=0, help="Start from this chunk index"
    )
    parser.add_argument(
        "--max-chapters", type=int, default=None, help="Only narrate this many chapters"
    )
    parser.add_argument(
        "--start-chapter", type=int, default=0, help="Skip the first N chapters"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be generated and how long it would take, then stop",
    )
    parser.add_argument(
        "--yes", "-y", action="store_true", help="Do not ask before starting"
    )
    parser.add_argument(
        "--quiet", "-q", action="store_true", help="Less output; useful for scripting"
    )
    parser.add_argument(
        "--version", action="version", version=_version_string()
    )

    return parser


def _version_string() -> str:
    """Version string, read from __init__.py if present."""
    init_file = PROJECT_DIR / "__init__.py"
    try:
        match = re.search(
            r'__version__\s*=\s*["\']([^"\']+)["\']',
            init_file.read_text(encoding="utf-8"),
        )
        if match:
            return f"f5-tts-cli {match.group(1)}"
    except OSError:
        pass
    return "f5-tts-cli"


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> int:
    global _QUIET

    parser = build_parser()
    args = parser.parse_args()

    _QUIET = args.quiet

    if args.doctor:
        return run_doctor()
    if args.list_voices:
        return list_voices()

    if not args.text and not args.input_file:
        parser.print_help()
        print("\nERROR: give some text with -t, or a book with -f.\n")
        return 2

    if args.input_file and not Path(args.input_file).is_file():
        print(f"ERROR: Input file not found: {args.input_file}")
        return 1

    if not has_ffmpeg():
        print("ERROR: ffmpeg is required but was not found.\n")
        print(ffmpeg_missing_message())
        return 1

    output_parent = Path(args.output).expanduser().parent
    if str(output_parent) not in ("", "."):
        output_parent.mkdir(parents=True, exist_ok=True)
    args.output = str(Path(args.output).expanduser())

    if args.input_file:
        if Path(args.output).suffix.lower() not in (".wav",) + AUDIOBOOK_FORMATS:
            print(
                f"ERROR: output must end in .wav, .m4b or .mp3 (got {Path(args.output).suffix})"
            )
            return 2
        try:
            if args.dry_run:
                return show_dry_run(args)
            return generate_book(args)
        except FileNotFoundError as exc:
            print(f"ERROR: {exc}")
            return 1
        except KeyboardInterrupt:
            print("\n\nInterrupted. Re-run with --resume to continue where you stopped.")
            return 130
    else:
        try:
            return generate_from_text(args)
        except (FileNotFoundError, ValueError) as exc:
            print(f"ERROR: {exc}")
            return 1
        except KeyboardInterrupt:
            print("\nInterrupted.")
            return 130


if __name__ == "__main__":
    sys.exit(main())
