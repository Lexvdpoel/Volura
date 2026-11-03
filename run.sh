#!/bin/sh
# Start Volura from source on Linux (creates a virtual environment the first time).
cd "$(dirname "$0")"
if [ ! -x .venv/bin/python ]; then
    python3 -m venv .venv
    .venv/bin/python -m pip install -r requirements.txt
fi
exec .venv/bin/python main.py "$@"
