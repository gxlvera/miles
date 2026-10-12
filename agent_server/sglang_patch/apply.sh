#!/bin/sh
# Apply the agent_server changes to an SGLang checkout: sh apply.sh <sglang repo root>
set -e
here="$(cd "$(dirname "$0")" && pwd)"
git -C "$1" apply "$here/sglang.patch"
cp -r "$here/continuous_token" "$1/python/sglang/srt/continuous_token"
find "$1/python/sglang/srt/continuous_token" -name __pycache__ -prune -exec rm -rf {} +
