#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")"

"${PYTHON:-python}" trainSpeakerNet.py "$@"
