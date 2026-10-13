#!/bin/sh
# Build and install sglang-router (the Rust router Miles launches by default) with router.patch, so chat
# requests keep prefix_message_count on their way to SGLang. Without it the router re-serializes the
# request from its own schema and drops the field.
#
# Usage: sh agent_server/sglang_patch/build_router.sh
set -eu

here=$(cd "$(dirname "$0")" && pwd)
repo=https://github.com/radixark/sgl-router-for-miles.git
# The router fork Miles' image builds from (docker/Dockerfile, SGL_ROUTER_REPO); the commit the patch is made against.
commit=374ff283397254d1a3c9577058d7089bcb91a7b1

work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
git init -q "$work/router"
git -C "$work/router" fetch -q --depth 1 "$repo" "$commit"
git -C "$work/router" checkout -q FETCH_HEAD
git -C "$work/router" apply "$here/router.patch"

if ! command -v cargo >/dev/null 2>&1; then
  curl --proto '=https' --tlsv1.2 --retry 3 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal
fi
export PATH="$HOME/.cargo/bin:$PATH"
python3 -m pip install -q maturin

ulimit -n 65536 || true
cd "$work/router/bindings/python"
maturin build --release --features vendored-openssl --out "$work/wheels"
python3 -m pip install -q --force-reinstall --no-deps "$work"/wheels/sglang_router-*.whl
python3 -c 'import sglang_router; assert sglang_router.FORWARDS_PREFIX_MESSAGE_COUNT; print("sglang-router", sglang_router.__version__, "forwards prefix_message_count")'
