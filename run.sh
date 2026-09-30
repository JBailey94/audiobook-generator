#!/usr/bin/env bash
#
# F5-TTS CLI - launcher.
#
# Uses the environment created by ./setup.sh. If it is missing, setup is run
# automatically so a fresh copy of this folder works with one command.
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/.venv"
VENV_PY="$VENV_DIR/bin/python"

# Never cd: the CLI resolves its own paths, and a user may be pointing at a
# book in another directory.
if [ -x "$VENV_PY" ]; then
    exec "$VENV_PY" "$SCRIPT_DIR/cli.py" "$@"
fi

if [ ! -d "$VENV_DIR" ] && [ -z "${F5TTS_NO_AUTOSETUP:-}" ]; then
    cat <<EOF
This copy of F5-TTS CLI has not been set up yet.
Running setup now - it installs the dependencies and downloads the models.

EOF
    "$SCRIPT_DIR/setup.sh"
    exec "$VENV_PY" "$SCRIPT_DIR/cli.py" "$@"
fi

cat <<EOF
ERROR: no Python environment found at .venv

Run this once to install everything:

    ./setup.sh

Already installed it? Then use that interpreter directly:

    .venv/bin/python cli.py $*
EOF
exit 1
