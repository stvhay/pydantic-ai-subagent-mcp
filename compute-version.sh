#!/usr/bin/env bash
set -euo pipefail

# Thin wrapper that delegates to compute_version.py
# Usage:
#   ./compute-version.sh          # Print next version
#   ./compute-version.sh --ci     # CI mode: read bump from CHANGELOG, update files
#   ./compute-version.sh --update # Update version files without changelog rewrite

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Check Python is available
if ! command -v python3 &>/dev/null; then
    echo "Error: python3 is required" >&2
    exit 1
fi

exec python3 "${SCRIPT_DIR}/compute_version.py" "$@"
