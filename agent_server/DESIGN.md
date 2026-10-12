# Trajectory Manager + TITO Design Doc

**TL;DR**

Two main components:
1. An http server (agent server) that does:
- Request Forwarding: intercept request from agent harness and forward it to SGLang engine for generation;
- Trajectory Representation: records every turn's trajectory in a *prefix-trie* data structure and conduct prefix matching for turn-merge, and exposes customize trajectory post processor API.
2. A tokenization component implemented on SGLang side that tokenize each turn's messages incrementally to ensure the assembly of multi-turn tokens are zero-drift. It also allows *multimodality* at *large scale*. 

## 1. Scope
This is a middle layer that sits between the agent harness layer and the RL infrastructure. It only covers a pipeline that transform the agent harness behavior into training evidence. 
The below are not implemented in this take-home but is used directly.
1. agent harness layer is supported by third-party provider. (e.g. E2B sandbox for this take-home)
2. RL infrastructure is provided by miles, SGLang and Megatron/FSDP. 

## 2. Overall Architecture and workflow

![Agent server architecture](assets/architecture.png)

The agent server is one Router plus N replicas. Each replica registers with the Router when it starts. An episode runs as follows:

1. Miles asks the Router to create a session.
2. The Router picks a replica, creates the session on it, and returns the session id and that replica's URL to Miles.
3. Miles injects the URL into the agent's task runtime. From then on, the task runtime sends every request directly to that replica; the Router is no longer involved. 
4. The agent server replica calls SGLang for generation, SGLang conducts multimodal object download (if need) and tokenization.
5. agent server send back generation result to harness layer, and that loop continues.
6. When a task is done, the session is closed and miles pulls trajectory data from agent server.


## 3. Prompt construction: continuous token in SGLang

Continuous token (CT) runs inside SGLang's TokenizerManager, patched by [sglang_patch/](sglang_patch/README.md), for every agent server request. Each request carries three things:

- `input_ids`: the parent turn's full token sequence, empty for a new root;
- the full `messages`;
- `prefix_message_count`: how many of those messages the tokens already cover.

SGLang renders `messages[prefix_message_count:]` onto the tokens, generates, and returns in `meta_info.prompt_edit` how the prompt extends them: `removed_prefix_token_count`, `input_delta` and `routed_experts_start_len`.

This keeps the agent server free of tokenizers, templates and processors. It also means images are encoded where they are fetched, while SGLang stays stateless because every request carries what it needs.

- **Builder choice.** The builders are a base `ContinuousTokenBuilder` and `QwenContinuousTokenBuilder`, selected by the served model's `model_type`. The Qwen builder covers `qwen2`, `qwen3`, `qwen3_moe`, `qwen3_5`, `qwen3_5_moe`, `qwen2_vl`, `qwen2_5_vl`, `qwen3_vl` and `qwen3_vl_moe`.
- **New root.** With `prefix_message_count = 0`, `build_initial_tokens(messages, tools)` renders the whole request.
- **Continuing a turn.** `merge_context_tokens(messages[:k], messages, input_ids, tools)` renders only the appended messages inside a synthetic context and appends the diff to the parent's tokens.
  - It may replace a short tail of the parent (`removed_prefix_token_count`). The Qwen builder never does; it only inserts the `\n` the template puts after `<|im_end|>`, which a stopped generation does not emit.
  - Assistant turns are never re-rendered: their stored tokens are reused as generated.
  - The tokens before the kept prefix are never decoded.
- **Images.** A group with images is rendered to text and encoded by the HF processor together with its images, so the processor itself expands their placeholders ([§6](#6-multimodal)). The synthetic context never holds images.
- **Template.**
  - The template is SGLang's: the checkpoint's own, or `--chat-template-path`, which Miles passes on as `--sglang-chat-template`. No Miles TITO model is used.
  - Template kwargs are `--apply-chat-template-kwargs` merged with the client's `chat_template_kwargs`, and are sent with every request.
  - SGLang's reasoning and tool-call parsers are chosen explicitly (`--sglang-reasoning-parser`, `--sglang-tool-call-parser`).

## 4. Trajectory Manager

### 4.1 Granularity: one sample per turn vs. best-effort turn merge

- **One sample per turn.** Each generation is a sample: full prompt (mask 0) plus output (mask 1). T turns give T samples that repeat the shared history, costing Σ L_t tokens (L_t = length of turn t).
- **Best-effort turn merge.** A turn that extends a stored turn continues its tokens, so a root-to-leaf path is one sample of L_T tokens. Requests that extend nothing (compaction, sub-agents) start new roots and retries start branches; each is still exported, so no trajectory is lost and the trainable tokens are the same.

Ablation: 64 Claude Code sessions on SWE-Gym (4 tasks × 16), Qwen3.5-4B, 8×H100, 64K context; one rollout batch trained both ways for one step.

| | [Per turn](https://forge.coreweave.com/wandb/xiaoleg-carnegie-mellon-university/humans&take-home/runs/proxy-per-turn-01) | [Turn merge](https://forge.coreweave.com/wandb/xiaoleg-carnegie-mellon-university/humans&take-home/runs/proxy-merge-01) |
| --- | ---: | ---: |
| Samples | 4,439 | 64 |
| Training tokens | 105.3 M | 2.71 M |
| Loss tokens | 698,954 | 698,954 |
| Training time | 5,164 s | 596 s (8.7×) |
| Rollout + training | 5,890 s | 1,322 s (4.5×) |

Per turn computes 38.8× as many tokens for the same loss tokens, so the agent server uses best-effort turn merge.

### 4.2 Representation: multi-linear chains vs. prefix trie

![Multi-linear chains vs. prefix-trie storage](assets/trajectory_storage.png)

- **Multi-linear chains** (e.g. verl uni-agent) copy the shared prefix into every chain: O(NL) time and memory, and how chains relate is lost.
- **Prefix trie** stores each turn once under the turn it extends: O(L) lookup, O(U) memory. Retries, best-of-N siblings and sub-agents stay visible, so the picker and postprocessor hooks (§5) can pick trajectories and assign rewards per branch.

Since perfix trie representation is more efficient and it allows for customized trajectory picking and reward assignment, agent server adopts this method.

### 4.3 Delta nodes and efficient CPU memory

```text
tokens(node) = tokens(parent)[:prefix_len] + input_delta + output_tokens
```

A node stores only its delta information, so each token, logprob, message, R3 and image reference is stored once which saves CPU memory; full sequences are rebuilt only for in-flight prompts and export. 


### 4.4 Data structures and interface

```python
prepare(messages, *, context_key) -> PreparedTurn        # find the parent, return its tokens
commit(prepared, generation) -> TurnNode                 # store SGLang's prompt edit and output
tokens(node_id) -> list[int]                             # rebuild a full sequence from deltas
leaf_samples(*, max_seq_len, tokenizer) -> list[Sample]  # one sample per root-to-leaf path
tree_metadata() -> dict                                  # node/leaf tables for picker/postprocessor
clear() -> None
```

A request runs `prepare` → SGLang → `commit`. Both are synchronous on one event loop, so no lock is needed. The parent is fixed in `prepare`, so concurrent requests from one turn become siblings.

| Data | Fields |
| --- | --- |
| PreparedTurn (in flight) | `parent_id`, `parent_tokens`, `prefix_message_count`, new messages, context key, fingerprints, reusable R3 rows |
| PromptEdit (from SGLang) | `removed_prefix_token_count`, `input_delta`, `routed_experts_start_len` |
| TurnNode (stored) | `node_id`, `parent_id`, `prefix_len`, `input_delta`, `output_tokens`, `output_logprobs`, `delta_messages`, `finish_reason`, `weight_versions`, `response_id`, `r3`, `images` |
| R3Chunk | `row_start`, `values` |

- **Finding the parent.** Messages are fingerprinted and walked through a trie keyed by tools and template kwargs; the deepest position where a committed turn ends is the parent (latest commit on ties). The last message is never consumed, so an identical request becomes a retry sibling.
- **Truncated turns** (`finish_reason == "length"`) are stored but not indexed, so a replay attaches to an earlier turn.
- **R3.** `r3(node) = r3(parent)[:row_start] + values`. Each turn asks for rows from `len(tokens(parent)) - 1` (all rows under `retract`); export takes only rows no descendant replaced, and a missing row is an error.

## 5. Miles integration and sample export

- **Flags.**
  - `--use-agent-server`, `--agent-server-workers N` (default 1) and `--agent-server-port`.
  - Mutually exclusive with `--use-session-server`; requires `--hf-checkpoint`.
  - Requires `--use-miles-router`. CT needs `prefix_message_count` to reach SGLang, and Miles' default Rust `sglang_router` re-serializes chat requests from its own schema and drops it. MilesRouter forwards the body unchanged but balances by load, so a session's turns may land on different engines and share less prefix cache.
  - SGLang must carry [sglang_patch/](sglang_patch/README.md).
  - Rejected at startup: `--partial-rollout`, `--use-rollout-indexer-replay`, sampling-support replay (`--rollout-top-p < 1` or `--rollout-top-k > 0`), `--rollout-top-logprobs-num`, `--group-rm`.
- **Hooks.**
  - `miles/ray/specs/entrypoint.py` adds the Router and replica worker specs.
  - `miles/ray/specs/rollout.py`, `miles/ray/rollout/rollout_executor.py` and `miles/ray/placement_group.py` wait for registration and set `args.agent_server_router_url`.
- **Rollout.**
  - `--custom-generate-function-path agent_server.miles_adapter.generate` creates a session through the Router.
  - It runs the agent function (`--custom-agent-function-path`) with `base_url = agent_base_url`, collects the samples, and deletes the session.
- **Backend call.**
  - `/v1/chat/completions` on Miles' inference router, with the session id as `X-SMG-Routing-Key`.
  - The request carries the parent's tokens as `input_ids`, `prefix_message_count`, the merged `chat_template_kwargs`, `logprobs=true` and `return_meta_info=true`, next to the client's `messages` and `tools`.
  - A reply without `meta_info.prompt_edit` is answered with 502 and nothing is committed: SGLang then generated from the parent's tokens alone.
  - Output token ids and logprobs come from `meta_info.output_token_logprobs`, never from re-tokenizing text.
  - `stream: true` is served as one burst of SSE events after generation finishes, because a turn is committed from the complete reply.
- **Export.**
  - One sample per root-to-leaf path. Prompt-side tokens (tool results, user messages, template tokens) get loss mask 0, and policy output gets mask 1 and its sampled logprob.
  - `max_seq_len` cuts at turn level: a turn whose prompt fills the budget is dropped with everything after it, and a turn whose output crosses it is cut and marked truncated.
  - Miles' `--session-sample-picker-path` and `--session-sample-postprocessor-path` hooks then run. By default they drop leaves superseded by a retry, count each shared completion in exactly one sample, and assign the agent's reward.
  - Samples travel in Miles' samples codec (`COMPUTED_FIELDS_V2`), whose JSON fields include `reward` and `multimodal_inputs`.

## 6. Multimodal

### 6.1 Motivation

The usual pipeline has three problems:

1. It renders messages to text with one image placeholder per image, lets the HF processor expand the placeholders and encode, and sends ids plus raw images to SGLang. SGLang then processes the images again and may decode the ids and re-encode them. Re-encoding drifts (BPE can split differently), so the trained tokens are no longer the tokens the model saw.
2. Expanding a placeholder requires the image, so whoever tokenizes has to download every image. The agent server runs on a comparatively small CPU tier where download bandwidth is the bottleneck. SGLang and the trainer are widely distributed, each of their machines only downloads its own share, and SGLang needs the pixels anyway.
3. Pixel tensors are large and none of them needs to travel between the agent server, SGLang and the trainer.

### 6.2 Design

Principle: the agent server handles only messages, image references and token ids; SGLang and the trainer fetch and process images; every token before a turn's kept prefix is reused as stored and never decoded.

```text
agent server : {parent tokens, messages (image URLs / data URIs), prefix_message_count}
SGLang       : CT; a group with new images → processor(text, images) → expanded ids + their pixels
               earlier images → image processor only (already expanded in the parent tokens)
               → items in prompt order, offsets from placeholder runs, M-RoPE → generate
               → prompt_edit (removed tail, input_delta with the new images expanded)
agent server : commit the edit and the turn's new image references
trainer      : expanded tokens + image references → image processor on each rank, tokens untouched
```

- **New images.**
  - They are encoded by the processor together with their group's text, so the processor decides how each one expands. No per-model formula, no counting.
  - Other modalities would expand the same way, e.g. video with its timestamp tokens.
- **Earlier images.** Their placeholders are already expanded in the parent's tokens and are not touched. SGLang still needs three things from each of them:
  - pixels, for the forward pass wherever the prefix cache misses (after a retract, an eviction, or on another engine);
  - its grid, for M-RoPE;
  - a hash, for prefix-cache matching.

  So SGLang runs the image processor on them, without text. A turn costs one processor call per appended group with images, plus one for all earlier images.
- **Assembly.**
  - The processor outputs of all images, earlier first and in prompt order, go to SGLang's existing preprocessed-input path as one `processor_output` item.
  - That path splits it per image by grid, and assigns the placeholder runs of the prompt to the images in order. The number of runs must equal the number of images.
- **Same pixels on both sides.** The CT path calls the HF processor on CPU and decodes images with PIL, as the trainer does, so SGLang and the trainer process identical pixels.
- **Commit.** The agent server stores `input_delta` with `prefix_len = len(tokens(parent)) - removed_prefix_token_count`. From then on every coordinate is in expanded positions: child `prefix_len`, R3 rows, `max_seq_len` cuts, weight-version spans.
- **Trainer.**
  - Samples carry the expanded tokens and `multimodal_inputs = {"images": [references in prompt order]}`.
  - Each trainer rank loads the images of its own samples and runs the checkpoint's image processor. That yields `pixel_values` and `image_grid_thw`; M-RoPE is computed inside the model from tokens and grids.
  - Tokens are only checked, never changed: the number of image tokens must equal `sum(grid.prod()) / merge_size²`.
- **Identity.** An image counts by its full URL or data URI in message fingerprints, so a different image is a different message.

### 6.3 Invariants

- The tokens of a turn's kept prefix are never changed or decoded downstream.
- Only SGLang and the trainer fetch images. No pixel tensor travels between components.
- SGLang and the trainer use the same processor configuration (the checkpoint's `preprocessor_config.json`), on CPU, with the same image decoding.
- A turn is committed only with SGLang's `prompt_edit`.

### 6.4 Implementation

| Part | Change |
| --- | --- |
| SGLang ([sglang_patch/](sglang_patch/README.md), against sglang-miles `6b5f3df`) | `ChatCompletionRequest.prefix_message_count`. `serving_chat` hands messages, tools and template kwargs to `GenerateReqInput.continuation`, taken from the request body as sent: the request models would reorder tool keys, which `tojson` renders. `TokenizerManager._continue_prompt` runs `continuous_token.continue_prompt` and puts `prompt_edit` into `meta_info`. The Qwen-VL processor takes the ready ids and processor outputs instead of loading images itself |
| Agent server | `prepare` only finds the parent. `build_request` sends `prefix_message_count`; `parse_generation` requires `prompt_edit`. `message_images` (shared with SGLang through `sglang_patch/continuous_token/media.py`) gives each turn's image references |
| Wire | samples codec v2 carries `multimodal_inputs` as JSON (a null keeps the driver's value). Train data carries `image_refs` (msgpack) instead of `multimodal_train_inputs` |
| Trainer | `get_rollout_data` calls `image_refs_to_train_inputs` with the processor of `--hf-checkpoint`, cached per process, before tensors move to the GPU. Megatron and FSDP both go through it. Samples with image references are not scheduled on the rollout side |
| Router | `--use-miles-router` (§5) |

### 6.5 Not done yet

- **Image cache in SGLang.** Every turn loads and processes all of the prompt's earlier images again. A cache keyed by a hash the agent server supplies (a URL or content hash) would skip both, leaving one processor call per turn.
- **Trainer image loading.** It could load on one rank per data-parallel group and broadcast, instead of on every tensor-/context-parallel rank.
- **Rest.** Session affinity behind MilesRouter (or the Rust router forwarding `prefix_message_count`); Anthropic image blocks (`/v1/messages`); video, untested.

## 7. Implementation decisions

- **Fingerprints.** A message is projected to role, content, `reasoning_content` and tool calls, with empty values treated as absent. Some fields change on a client round trip without changing what the model saw, so they are relaxed:
  - surrounding whitespace in content and reasoning is ignored, and content made only of text blocks counts as the joined string;
  - a tool call counts by its function name only. Clients rewrite arguments on replay: Claude Code fills in defaults and drops parameters the model made up, and the Anthropic ↔ OpenAI conversion changes ids and JSON formatting;
  - text an agent rewrites on every request is dropped from system messages, such as Claude Code's `<total_tokens>N tokens left</total_tokens>`.

  A matched turn reuses its stored tokens, so none of these relaxations can change a prompt token. They were found one by one with Claude Code ([experiments/swe_claude_code](experiments/swe_claude_code/EXPERIMENT_LOG.md)).
- **Request fields the agent server owns.** Clients must not send `input_ids`, and only `n = 1` is supported. A training session always samples at the run's temperature, because the trainer assumes rollout logprobs were sampled at it.
- **Anthropic.** `/v1/messages` converts with SGLang's Anthropic converters and Miles' adapter helpers. It drops client `thinking` and `output_config`, since whether the policy thinks is decided by its chat template, and it clears `is_error` on tool results.
- **Diagnostics.** A replica logs, once per session, when a replayed history matched no stored turn and when a request branches off an older turn, with the shape of the first differing message. `collect` logs turns, roots and leaves per session; a growing single chain means replay matched.

## 8. APIs and Files

**APIs**

| Process | Endpoint | Purpose |
| --- | --- | --- |
| Router | `POST /replicas` | register a replica (`url` for Miles, `agent_url` for agents) |
| Router | `GET /replicas` | registered replicas; Miles starts rollout once all N are there |
| Router | `POST /sessions` | pick a replica at random, create the session there, return the handle |
| Replica | `POST /sessions` | create a session and its `TrajectoryManager` |
| Replica | `POST /sessions/{id}/v1/chat/completions` | find the parent → SGLang renders (CT) and generates → commit the turn |
| Replica | `POST /sessions/{id}/v1/messages` | Anthropic messages, converted to the chat path above |
| Replica | `POST /sessions/{id}/samples` | export samples; carries the agent's reward and metadata |
| Replica | `DELETE /sessions/{id}` | drop everything the session holds |
| Replica | `/sessions/{id}/{path}` | anything else (e.g. Anthropic `count_tokens`) is forwarded to the backend unrecorded |

**Files**

| Path | Content |
| --- | --- |
| `router.py` | replica registration, session placement |
| `server.py` | replica HTTP app, session lifecycle, prepare → generate → commit |
| `trajectory.py` | message trie, delta nodes, image references, sample assembly |
| `protocol.py` | `SessionHandle`, `PreparedTurn`, `Generation`, `TurnNode`, `R3Chunk`, `AgentServerError` |
| `backend.py` | request building, reply parsing (tokens, logprobs, versions, R3, prompt edit) |
| `anthropic.py` | `/v1/messages` |
| `miles_adapter.py` | worker specs, readiness, the Miles generate function |
| `sglang_patch/` | what SGLang runs for the agent server: `continuous_token/` (CT builders, templates, image encoding; copied into SGLang) and `sglang.patch` (request fields and hooks), applied with `apply.sh` |
| `tests/` | `test_trajectory.py` (trie, deltas, export, R3, hooks; toy tokenizer); `test_continuous_token.py` (CT against full renders); `test_replica.py` (Router plus two replicas plus a fake SGLang running CT); `test_multimodal.py` (turn-by-turn prompts, with and without images, against the one-shot HF processor; SGLang emulated with the same CT code, or a real patched SGLang) |
| `experiments/` | Modal runs: Claude Code on SWE-Gym, DeepEyes |
