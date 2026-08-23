#!/usr/bin/env bash

set -e

# Move to the repository root, independently of where the script is executed from
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT_DIR"

# Select Python command
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

echo "Building RL-based C extension..."

cd "Compression/RL-based approach"
$PYTHON_CMD setup.py build_ext --inplace

cd "$ROOT_DIR"

echo "Running RL-based compression/decompression tests..."

$PYTHON_CMD Test/test_scripts/test_rl_based.py

echo "RL-based tests completed successfully."
