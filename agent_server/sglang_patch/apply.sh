#!/bin/sh
# Apply the agent server's changes to an SGLang checkout: sh apply.sh <sglang repo root>
#
# Made against sglang-miles 6b5f3df, the SGLang in radixark/miles:dev-202610102218. sglang.patch:
# - /v1/chat/completions takes prefix_message_count (input_ids are the tokens of that many messages) and
#   hands the messages, tools and template kwargs to TokenizerManager, whose _continue_prompt runs
#   continuous token before anything reads input_ids;
# - the processor outputs of every image go on as one preprocessed item, which the Qwen-VL processor takes
#   with the ready ids instead of loading the images again;
# - the reply's meta_info.prompt_edit says how the prompt extends the sent tokens;
# - Qwen2.5-VL's vision rotary keeps its angles, cos and sin in fp32, as transformers does.
# continuous_token/ is copied in as sglang.srt.continuous_token.
set -e
here="$(cd "$(dirname "$0")" && pwd)"
git -C "$1" apply "$here/sglang.patch"
cp -r "$here/continuous_token" "$1/python/sglang/srt/continuous_token"
find "$1/python/sglang/srt/continuous_token" -name __pycache__ -prune -exec rm -rf {} +
