#!/usr/bin/env bash
#
# Create an F5-TTS voice sample from a YouTube video.
#
# Downloads only the requested few seconds of audio, trims silence from the
# edges and writes a mono 24 kHz WAV, which is what F5-TTS expects.
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PY="$SCRIPT_DIR/.venv/bin/python"
TARGET_RATE=24000
TRIM_SILENCE=0

usage() {
    cat <<EOF
Create a voice sample from a YouTube video.

Usage: ./get_voice_sample.sh <youtube-url> <start> <end> [output.wav]

  start, end   Timestamps. Accepts 90, 1:30 or 00:01:30.
  output.wav   Where to save the clip (default: voice_sample.wav)

Options:
  --trim-silence   Also trim long silences inside the clip
  -h, --help       Show this help

Aim for 5 to 10 seconds of one person talking, with no music or background
noise. F5-TTS ignores anything after the first 12 seconds.

Examples:
  ./get_voice_sample.sh "https://www.youtube.com/watch?v=VIDEOID" 1:30 1:38 my_voice.wav
  ./get_voice_sample.sh "https://youtu.be/VIDEOID" 62 70
EOF
}

# ---------------------------------------------------------------------------
# Arguments
# ---------------------------------------------------------------------------

if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
    usage
    exit 0
fi

POSITIONAL=()
while [ $# -gt 0 ]; do
    case "$1" in
        --trim-silence) TRIM_SILENCE=1 ;;
        -h|--help) usage; exit 0 ;;
        -*) echo "Unknown option: $1" >&2; echo >&2; usage >&2; exit 1 ;;
        *) POSITIONAL+=("$1") ;;
    esac
    shift
done

if [ "${#POSITIONAL[@]}" -lt 3 ]; then
    usage >&2
    exit 1
fi

URL="${POSITIONAL[0]}"
START="${POSITIONAL[1]}"
END="${POSITIONAL[2]}"
OUTPUT="${POSITIONAL[3]:-voice_sample.wav}"
[ "${OUTPUT##*.}" = "$OUTPUT" ] && OUTPUT="${OUTPUT}.wav"

# Accept 90, 1:30 or 00:01:30 and normalise to seconds.
to_seconds() {
    python3 - "$1" <<'PY'
import sys
raw = sys.argv[1].strip()
parts = raw.split(":")
if not all(p.strip().isdigit() for p in parts):
    sys.exit(f"ERROR: cannot understand the timestamp '{raw}'. Try 90, 1:30 or 00:01:30.")
total = 0
for part in parts:
    total = total * 60 + int(part)
print(total)
PY
}

START_SEC="$(to_seconds "$START")"
END_SEC="$(to_seconds "$END")"
DURATION=$((END_SEC - START_SEC))

if [ "$DURATION" -le 0 ]; then
    echo "ERROR: the end time must come after the start time." >&2
    exit 1
fi

if [ "$DURATION" -gt 12 ]; then
    cat >&2 <<EOF
WARNING: that clip is ${DURATION}s long. F5-TTS only uses the first 12
         seconds of a sample. A 5-10s clip gives the best results.
EOF
fi
if [ "$DURATION" -lt 3 ]; then
    echo "WARNING: that clip is only ${DURATION}s long. 5-10s works much better." >&2
fi

# ---------------------------------------------------------------------------
# Dependencies
# ---------------------------------------------------------------------------

install_hint() {
    cat >&2 <<EOF
ERROR: $1 is required but was not found.

  ffmpeg is needed to cut the audio:
    Debian/Ubuntu : sudo apt install ffmpeg
    Fedora        : sudo dnf install ffmpeg
    Arch          : sudo pacman -S ffmpeg
    macOS         : brew install ffmpeg
    Windows       : winget install Gyan.FFmpeg
EOF
}

if ! command -v ffmpeg >/dev/null 2>&1; then
    install_hint "ffmpeg"
    exit 1
fi

if [ -x "$VENV_PY" ]; then
    YTDLP=("$VENV_PY" -m yt_dlp)
    if ! "$VENV_PY" -c "import yt_dlp" >/dev/null 2>&1; then
        echo "==> yt-dlp is not in the project environment. Installing it..."
        "$VENV_PY" -m pip install --quiet --upgrade yt-dlp
    fi
elif command -v yt-dlp >/dev/null 2>&1; then
    YTDLP=(yt-dlp)
else
    echo "==> yt-dlp is not installed. Installing it..."
    if command -v pipx >/dev/null 2>&1; then
        pipx install --quiet yt-dlp || true
        YTDLP=(yt-dlp)
    elif command -v pip3 >/dev/null 2>&1; then
        pip3 install --quiet --user --upgrade yt-dlp || true
        YTDLP=(yt-dlp)
    else
        install_hint "yt-dlp"
        exit 1
    fi
fi

if ! command -v "${YTDLP[0]}" >/dev/null 2>&1 && ! "${YTDLP[0]}" --version >/dev/null 2>&1; then
    install_hint "yt-dlp"
    exit 1
fi

# ---------------------------------------------------------------------------
# Extract
# ---------------------------------------------------------------------------

WORK_DIR="$(mktemp -d)"
trap 'rm -rf "$WORK_DIR"' EXIT
RAW="$WORK_DIR/source.%(ext)s"

echo "==> Fetching audio from $URL (${START}s to ${END}s)"

# --download-sections asks yt-dlp to fetch just that slice instead of the whole
# track, which is much faster. It needs ffmpeg, which we have just checked for.
# The clip it produces already starts at START_SEC, so it needs no further
# trimming; only a full download has to be cut here.
TRIM_ARGS=()

if "${YTDLP[@]}" \
        --no-warnings \
        --no-playlist \
        --extract-audio \
        --audio-format wav \
        --audio-quality 0 \
        -f bestaudio \
        --download-sections "*${START_SEC}-${END_SEC}" \
        --force-keyframes-at-cuts \
        -o "$RAW" \
        "$URL"; then
    :
else
    echo "" >&2
    echo "==> Partial download failed. Falling back to a full download..." >&2
    "${YTDLP[@]}" \
        --no-warnings \
        --no-playlist \
        --extract-audio \
        --audio-format wav \
        -f bestaudio \
        -o "$RAW" \
        "$URL"
    TRIM_ARGS=(-ss "$START_SEC" -to "$END_SEC")
fi

# yt-dlp appends an extension when the template has none.
SOURCE_FILE="$(find "$WORK_DIR" -maxdepth 1 -name 'source.*' -type f | head -n 1)"
[ -n "$SOURCE_FILE" ] || { echo "ERROR: the download produced no audio file." >&2; exit 1; }

echo "==> Converting to ${TARGET_RATE} Hz mono WAV"
mkdir -p "$(dirname "$OUTPUT")"

FILTERS="aresample=${TARGET_RATE},aformat=channel_layouts=mono"
if [ "$TRIM_SILENCE" -eq 1 ]; then
    FILTERS="silenceremove=start_periods=1:start_silence=0.1:start_threshold=-45dB:detection=peak,$FILTERS"
fi

ffmpeg -y -loglevel error -i "$SOURCE_FILE" \
    "${TRIM_ARGS[@]}" \
    -af "$FILTERS" \
    -ac 1 -ar "$TARGET_RATE" -c:a pcm_s16le \
    "$OUTPUT"

if [ ! -s "$OUTPUT" ]; then
    echo "ERROR: no audio came out. Check that the timestamps are inside the video." >&2
    exit 1
fi

ACTUAL_DURATION="$(ffprobe -v error -show_entries format=duration \
    -of default=noprint_wrappers=1:nokey=1 "$OUTPUT" 2>/dev/null || true)"

echo ""
echo "Voice sample saved: $OUTPUT"
if [ -n "$ACTUAL_DURATION" ]; then
    echo "  Length: ${ACTUAL_DURATION}s   Format: ${TARGET_RATE} Hz mono WAV"
fi
echo ""
echo "Preview it, then narrate something with:"
echo "  ./run.sh -t \"This is a test of my cloned voice.\" -v \"$OUTPUT\" -o test.wav"
