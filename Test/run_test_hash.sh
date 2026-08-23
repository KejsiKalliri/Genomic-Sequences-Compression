#!/usr/bin/env bash

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT_DIR"

if command -v python >/dev/null 2>&1; then
    PYTHON_CMD="python"
elif command -v python3 >/dev/null 2>&1; then
    PYTHON_CMD="python3"
elif command -v py >/dev/null 2>&1; then
    PYTHON_CMD="py -3"
else
    echo "Error: Python was not found. Please install Python and add it to PATH."
    exit 1
fi

echo "Using Python: $PYTHON_CMD"

echo "Running Hash-based compression/decompression tests..."

$PYTHON_CMD Test/test_scripts/test_hash_based.py

echo "Hash-based tests completed successfully."