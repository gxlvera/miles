# SGLang patch: continuous token in SGLang

What SGLang runs for the agent server ([DESIGN.md §3](../DESIGN.md#3-prompt-construction-continuous-token-in-sglang), [§6](../DESIGN.md#6-multimodal)). Every agent server run needs it. It targets sglang-miles `6b5f3df`, the build in `radixark/miles:dev-202610102218`.

- `continuous_token/` is the continuous token (CT) package: builders, template rendering and image encoding. It is copied into SGLang as `sglang.srt.continuous_token`, and the agent server's tests import it from here.
- `sglang.patch` changes six files:
  - `/v1/chat/completions` takes `prefix_message_count`: `input_ids` are the tokens of the first that many messages. `serving_chat` hands the messages, tools and template kwargs to `GenerateReqInput.continuation`.
  - `TokenizerManager._continue_prompt` runs `continue_prompt` before anything reads `input_ids`. It renders the remaining messages onto the tokens; a group with images is encoded by the HF processor, which expands its placeholders. Earlier images go through the image processor only.
  - The processor outputs of every image are passed on as one preprocessed `processor_output` item. The Qwen-VL processor takes them with the ready ids instead of loading images itself, then builds per-image items, offsets and M-RoPE.
  - The reply's `meta_info.prompt_edit` carries `removed_prefix_token_count`, `input_delta` and `routed_experts_start_len`.
  - Qwen2.5-VL's vision tower keeps its rotary angles, cos and sin in fp32, as transformers does. SGLang rounded them to bf16, which puts the angles of large images up to half a radian off.

Apply it to an SGLang checkout:

```bash
sh agent_server/sglang_patch/apply.sh /sgl-workspace/sglang
```

Whatever sits between the agent server and SGLang must forward `prefix_message_count`. The Rust `sglang_router`, Miles' default inference router, parses chat requests into its own schema and drops it, so Miles requires `--use-miles-router` with `--use-agent-server`. If a reply comes without `prompt_edit`, the agent server rejects the turn with a 502 instead of committing it.

Not done: SGLang processes every earlier image of the prompt again on each turn. A cache keyed by a hash the agent server supplies would skip that.
