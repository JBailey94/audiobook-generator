# F5-TTS CLI

Turn text into speech in **your own voice**. The tool clones a voice from a
short recording you provide — 5–10 seconds is enough. Point it at a PDF, EPUB,
Markdown or text file and it produces an audiobook with one file per chapter
plus a single `.m4b` you can load into any audiobook app.

Everything else is downloaded for you. There is nothing to configure.

```bash
./setup.sh                                   # once: installs everything
./get_voice_sample.sh "<video-url>" 1:30 1:38 my_voice.wav  # gets your voice
./run.sh -t "Hello, this is a test." -v my_voice.wav -o hello.wav
```

---

## Contents

- [What you need](#what-you-need)
- [Setup](#setup)
- [Get a voice to clone](#get-a-voice-to-clone)
- [Your first audio clip](#your-first-audio-clip)
- [Make an audiobook](#make-an-audiobook)
- [How long will it take?](#how-long-will-it-take)
- [All options](#all-options)
- [Tips for a good voice clone](#tips-for-a-good-voice-clone)
- [If something goes wrong](#if-something-goes-wrong)
- [How this is put together](#how-this-is-put-together)
- [Please use this responsibly](#please-use-this-responsibly)

---

## What you need

| | |
|---|---|
| **Required** | Python 3.10 or newer, [ffmpeg](https://ffmpeg.org/download.html), ~4 GB of disk space, internet access for the first run |
| **Needed too** | A 5–10 second recording of the voice you want to clone (see [Get a voice to clone](#get-a-voice-to-clone)) |
| **Optional** | An NVIDIA or Apple GPU. Without one it still works, just much more slowly. |

`ffmpeg` is the one thing `setup.sh` cannot install for you, because it is a
system program rather than a Python package. Install it with one of:

```bash
sudo apt install ffmpeg        # Debian, Ubuntu, WSL
sudo dnf install ffmpeg        # Fedora
sudo pacman -S ffmpeg          # Arch
brew install ffmpeg            # macOS
winget install Gyan.FFmpeg     # Windows
```

Check it worked:

```bash
ffmpeg -version
```

---

## Setup

```bash
./setup.sh
```

This one command:

1. creates a private Python environment in `.venv`, so it never touches the
   rest of your system;
2. installs the Python packages;
3. downloads the F5-TTS model (~1.3 GB), the speech recogniser and the vocoder;
4. verifies everything with a health check.

It takes 10–20 minutes, most of it downloading. It is safe to run again.

Pick the PyTorch build that matches your machine if the automatic choice is
wrong:

```bash
./setup.sh --cpu     # CPU only, smallest download (~250 MB)
./setup.sh --cuda    # NVIDIA GPU
```

Check the install at any time:

```bash
./run.sh --doctor
```

```
System tools
  [OK ] ffmpeg ......................... /usr/sbin/ffmpeg

Models
  [OK ] F5-TTS checkpoint ............... 1.3 GB in .../f5_model/F5TTS_v1_Base
  [OK ] Whisper (voice samples) ........ openai/whisper-base cached
```

---

## Get a voice to clone

F5-TTS clones a voice from 5–10 seconds of clean speech. The voice has to be
yours, or belong to someone who has agreed to it being used. Two ways to get a
good recording:

| | What you need | Commands shown below |
|---|---|---|
| **From YouTube** | A video where one person talks to camera with no music | `./get_voice_sample.sh` |
| **Your own recording** | A phone voice memo, 5–10 s, no background noise | nothing extra |

Let the script work out what the sample says on its own, or tell it with `-r`
(see [Your first audio clip](#your-first-audio-clip)).

### From a YouTube video

The helper script pulls exactly 5–10 seconds out of a YouTube video for you.

**1. Find a timestamp.** Open the video on YouTube and note the moment you want
— somewhere the host is talking clearly. Most videos have a transcript: click
**…more** under the video, then **Show transcript**, and scroll to a section
with real sentences. Click a line to jump to that timestamp, then note the time.

**2. Run the script:**

```bash
./get_voice_sample.sh "https://www.youtube.com/watch?v=VIDEO_ID" 1:30 1:38 my_voice.wav
```

Times can be written as `90`, `1:30` or `00:01:30`. The first run installs
`yt-dlp` for you if it is missing.

**3. Try it:**

```bash
./run.sh -t "This is a test of my cloned voice." -v my_voice.wav -o test.wav
```

Open `test.wav` and listen. If it sounds right, you are ready to narrate
anything.

### From your own recording

Record 5–10 seconds of someone speaking normally (a phone voice memo is
perfect), no music or background noise. Any of `.wav`, `.flac`, `.ogg`,
`.mp3` or `.m4a` works:

```bash
./run.sh -t "Hello there" -v ~/Downloads/my_voice.m4a -o hello.wav
```

If you use `-v` with just a file name, the file is looked for in `voices/` —
so dropping your recordings there lets you say `-v my_voice.wav` anywhere, and
`./run.sh --list-voices` shows what you have.

Remember: **only clone voices you have permission to use.** See
[Please use this responsibly](#please-use-this-responsibly).

---

## Your first audio clip

```bash
./run.sh -t "Hello, this is a test of the system." -v my_voice.wav -o hello.wav
```

The script works out what is said in the voice sample on its own, then reads
your text aloud in that voice.

If it mishears your sample, tell it the truth instead — this is the single
biggest quality win available to you:

```bash
./run.sh -t "Hello there" -v my_voice.wav -o hello.wav -r "The exact words spoken in my sample."
```

---

## Make an audiobook

### 1. Look before you leap

Generation takes a long time, so preview the plan first. This is instant and
downloads nothing (it does not even need your voice sample):

```bash
./run.sh -f "my book.epub" -o book.m4b --dry-run
```

```
==> Found 92 chapters, 14978 chunks

Chapters:
    1. Table of Contents  (7 chunks, ~819 chars)
    3. PROLOGUE  (190 chunks, ~20,329 chars)
    4. BRAN  (149 chunks, ~15,848 chars)
    ...

Total: 14978 chunks
Estimated time on CUDA: about 14h 34m
```

If the chapters or the time estimate look wrong, adjust before you start:

```bash
# Skip the copyright page and the table of contents
./run.sh -f "my book.epub" -o book.m4b --dry-run --start-chapter 3

# Only the first two chapters, to check quality
./run.sh -f "my book.epub" -o book.m4b -v my_voice.wav --max-chapters 2 --yes
```

Always test a couple of chapters before committing to the whole book.

### 2. Generate it

Add your voice sample and go:

```bash
./run.sh -f "my book.epub" -o book.m4b -v my_voice.wav
```

You will be asked to confirm once it knows how long the job is. Progress looks
like this:

```
Chapter 5/60: CAMILLE [312/4391] | The morning light came through the curtains and
    chapter saved: 005 - CAMILLE.wav
```

### 3. What you get

```
book.m4b                          the whole audiobook, chapters embedded
book_chapters/
  001 - Copyright.wav             one file per chapter
  002 - Dedication.wav
  003 - CONTENTS.wav
  004 - Chapter One.wav
  ...
```

The `.m4b` carries chapter markers, so your audiobook app shows a table of
contents and you can jump around. The chapter WAVs are there if you would
rather listen to them in a normal audio player, edit them, or burn them to a CD.

Keep `book_chapters/` if you want the individual files; it is not needed to
play the `.m4b`.

### 4. If you need to stop

Press `Ctrl+C` whenever you like, then pick up exactly where you left off:

```bash
./run.sh -f "my book.epub" -o book.m4b -v my_voice.wav --resume
```

`--resume` reuses every chunk already generated and only makes the missing
ones, so a crash or a reboot costs you nothing.

### Notes

- **Supported formats**: `.epub`, `.pdf`, `.md`, `.txt`. EPUB is best: chapter
  titles come straight from the book's table of contents.
- **Scanned PDFs** contain images of text, not text. This cannot read them. Use
  EPUB, or a PDF that has been through OCR.
- **To narrate into another format**, change the extension: `-o book.mp3` gives
  a smaller MP3, `-o book.wav` gives chapter WAVs only.
- **Add a title** to the file with `--book-title "My Book"`.
- **Speed** it up or down for the whole book with `--speed 1.1`.
- The same voice narrates the whole book. A different book? Re-run with a new
  `-v`.

---

## How long will it take?

Measured on this project, per ~100 characters of text (about 6 seconds of
speech):

| | Speed | A 10-hour book |
|---|---|---|
| NVIDIA GPU | ~3.5 s / chunk | ~6 hours |
| CPU only | ~2.3 min / chunk | ~9 days |

The text is generated in small pieces and stitched together, so the cost scales
with the length of the book. A chapter is 3000 characters by default, so
`--max-chapters 1` gives you a good feel for your own machine in a few minutes.

**CPU is genuinely this slow** — it is roughly 20× slower than real time. That
is a property of the model on CPU, not a misconfiguration. If you have no GPU:

- test a single chapter first to check the voice sounds right;
- then run the book in the background with `nohup`, or in stages with
  `--max-chapters`, using `--resume` to continue.

Run `--dry-run` before committing and it will print the estimate for your
machine.

---

## All options

Run `./run.sh --help` for the same list.

### Input

| Flag | Meaning |
|---|---|
| `-t`, `--text TEXT` | Speak this text |
| `-f`, `--input-file FILE` | Narrate this book (`.epub`, `.pdf`, `.md`, `.txt`) |
| `--dry-run` | Show the plan and time estimate, generate nothing |
| `--doctor` | Check the installation |
| `--list-voices` | List the voice samples in `voices/` |
| `--version` | Print the version |

### Voice

| Flag | Meaning |
|---|---|
| `-v`, `--voice FILE` | The voice to clone (WAV/FLAC/OGG/MP3). A file in `voices/` can be given by name, otherwise a path |
| `-r`, `--ref-text TEXT` | The exact words in the sample. Detected automatically if omitted |
| `--speed FLOAT` | Speech speed, `0.8` is slower and `1.2` faster (default `1.0`) |
| `--no-remove-silence` | Keep the pauses instead of tightening them |

### Output

| Flag | Meaning |
|---|---|
| `-o`, `--output FILE` | `.wav` for chapters only, `.m4b` or `.mp3` for an audiobook |
| `--book-title TEXT` | Title to embed in the `.m4b`/`.mp3` |
| `--chapters DIR` | Where to put the chapter files (default `<output>_chapters/`) |
| `--combine` | With `-o book.wav`, also write one file containing everything |

### Choosing what to narrate

| Flag | Meaning |
|---|---|
| `--start-chapter N` | Skip the first N chapters |
| `--max-chapters N` | Stop after N chapters |
| `--start-chunk N` | Skip the first N chunks of speech |
| `--max-chunks N` | Stop after N chunks |
| `--min-section-chars N` | Skip sections shorter than this (default `120`; `0` keeps everything) |
| `--section-chars N` | Target chapter length for books with no structure (default `3000`) |
| `--language CODE` | Language code of the text (default `eng`) |
| `--max-chars N` | Characters per chunk (default: automatic) |

### Long jobs

| Flag | Meaning |
|---|---|
| `--resume` | Reuse chunks already generated instead of starting over |
| `--keep-chunks` | Keep the intermediate per-chunk WAV files |
| `-y`, `--yes` | Do not ask before starting |
| `-q`, `--quiet` | Only print the result |
| `--device` | `cpu`, `cuda` or `mps` (default: detected) |

---

## Tips for a good voice clone

1. **One speaker, no music.** Background music is copied into your output too.
2. **5–10 seconds.** Under 3 seconds the voice sounds thin; over 12 seconds
   F5-TTS throws the extra away anyway.
3. **No shouting or whispering.** Normal speaking voice works best.
4. **Give the exact transcript** with `-r` if you can. Whisper is good but not
   perfect, and a wrong reference text is the most common cause of a poor clone.
5. **A GPU helps a lot** — not for quality, but for how long you wait.

---

## If something goes wrong

**`./run.sh --doctor` is the first thing to try.** It names what is
missing and how to fix it.

**"no voice sample was given"** — the tool needs a recording of the voice to
clone. Follow [Get a voice to clone](#get-a-voice-to-clone) and pass it with
`-v`, or drop it into `voices/`.

**"ffmpeg is required"** — install it, see [What you need](#what-you-need).

**"Model checkpoint not found"** — run `./setup.sh`, or generate something
once and it will download the model on demand.

**The model download stops part-way** — run `./setup.sh` again. It resumes
rather than starting over.

**"Could not work out what is said in the voice sample"** — pass the text
yourself with `-r "..."`.

**The audio contains the words of my sample, or odd noises** — the reference
text is wrong. Re-run with `-r` and the exact words.

**It is very slow** — you are probably on CPU, where the model runs about 20×
slower than real time (see [How long will it take?](#how-long-will-it-take)).
Check with `./run.sh --doctor`; if it says `GPU not detected`, try installing
PyTorch for your GPU with `./setup.sh --cuda`.

**`error: externally-managed-environment`** — you installed into the system
Python by mistake. Use `./run.sh`, which uses the private `.venv`.

**A long book is taking forever** — turn it into chapters with `-o book.m4b`,
use `--resume`, and consider running only part of the book with
`--max-chapters`.

---

## How this is put together

To share this, send the whole folder — it is under a megabyte before setup.
The models download afterwards.

| File | Role |
|---|---|
| `setup.sh` | Installs dependencies and downloads the models |
| `run.sh` | Launcher; runs setup automatically if it has not been run |
| `get_voice_sample.sh` | Pulls a 5–10 s voice sample out of a YouTube video |
| `cli.py` | The command line tool: doctor, chapter splitting, audio assembly |
| `models.py` | Finds and downloads the F5-TTS, Whisper and Vocos models |
| `tts.py` | Wraps the F5-TTS engine; audio conversion helpers |
| `text_processing.py` | Sentence splitting and text cleanup |
| `file_processing.py` | Reads PDF/EPUB/MD/TXT and finds their chapters |
| `voices/` | Your voice samples go here; the tool ships none |
| `f5_model/` | Created on first run: the F5-TTS checkpoint (~1.3 GB) |
| `.venv/` | Created on first run: the Python environment |
| `.cache/` | Created on first run: prepared voice clips and transcripts |

Only the models are large. Moving the folder to another machine and running
`./setup.sh` there rebuilds everything from scratch.

### Using it somewhere else on your system

```bash
export PATH="/path/to/f5-tts-cli:$PATH"
alias f5tts="/path/to/f5-tts-cli/run.sh"
```

### Credits

Built on [F5-TTS](https://github.com/SWivid/F5-TTS) by SWivid, with Whisper for
transcription and Vocos for audio synthesis. The text processing pipeline is
adapted from [ebook2audiobook](https://github.com/DrewThomasson/ebook2audiobook).

---

## Please use this responsibly

This tool can imitate any voice from a short sample. That is powerful, and it
is easy to abuse.

- **Only clone voices you have permission to use.** If the person did not agree,
  do not do it.
- **Do not use a voice to say things its owner did not say.** Impersonation and
  fraud are illegal in most places.
- **Say it is synthetic.** Most platforms and jurisdictions now require AI
  audio to be disclosed.
- **Check the rules of wherever you publish it.**

If you clone your own voice, or a voice you have explicit permission to use,
you are in the clear. That is the case this tool is designed for.