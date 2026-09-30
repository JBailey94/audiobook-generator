# Your voice samples

Put your own voice recordings here (`.wav`, `.flac`, `.ogg`, `.mp3` or `.m4a`)
and they are picked up automatically:

```bash
./run.sh -t "Hello there" -o hello.wav          # uses the only sample in voices/
./run.sh --list-voices                          # see what is here
```

Or skip this folder entirely and point at a file anywhere on disk:

```bash
./run.sh -t "Hello there" -v ~/Downloads/my_voice.wav -o hello.wav
```

Don't have a recording of the voice you want to clone yet? Pull 5-10 seconds
out of a YouTube video:

```bash
./get_voice_sample.sh "https://www.youtube.com/watch?v=VIDEO_ID" 1:30 1:38 my_voice.wav
```

For the best clone: one person, normal speaking voice, no music or background
noise, 5-10 seconds long. Anything you drop in here stays on your machine —
the folder is not part of the distributed project.