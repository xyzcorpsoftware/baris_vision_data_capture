#!/usr/bin/env bash
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source /opt/ros/humble/setup.bash
set -u
exec "${SCRIPT_DIR}/dataset_capture_gui.py" "$@"
