# agent_server

Miles 的 agent session 服务：一个 Router 加 N 个 Replica 进程。Replica 用消息前缀 Trie 找父节点，每个 turn 节点只存自己新增的 token、logprob 和消息；prompt 由打了 [sglang_patch](sglang_patch/README.md) 的 SGLang 用 continuous token 在父节点 token 上增量构造（带图的消息由 processor 编码）。设计见 [DESIGN.md](DESIGN.md)。

## 在 Miles 里用

```bash
python train.py ... \
  --use-agent-server --agent-server-workers 4 --use-miles-router \
  --custom-generate-function-path agent_server.miles_adapter.generate \
  --custom-agent-function-path <your agent fn> \
  --sglang-reasoning-parser qwen3 --sglang-tool-call-parser qwen3_coder
```

- `--use-agent-server` 不能和 `--use-session-server` 同时用，也不需要 `--tito-model`。SGLang 必须打 [sglang_patch](sglang_patch/README.md)（`sh agent_server/sglang_patch/apply.sh <sglang 仓库>`），并且必须用 `--use-miles-router`：默认的 Rust router 会丢掉 `prefix_message_count`。prompt 用 SGLang 的聊天模板，即 checkpoint 自带的（可用 `--chat-template-path`、`--apply-chat-template-kwargs` 覆盖）。
- agent 函数签名是 `async def run(base_url, prompt, request_kwargs, metadata, **kwargs) -> dict | None`。拿到 `base_url` 后请求 `{base_url}/v1/chat/completions`，Claude Code 这类 Anthropic 客户端用 `{base_url}/v1/messages`；返回的 dict 作为 agent metadata（含 `reward`）随 collect 一起提交。agent 要原样回放上一轮的 assistant 消息（包括 `reasoning_content` 和 tool calls），否则会开新根。
- picker/postprocessor 沿用 `--session-sample-picker-path` / `--session-sample-postprocessor-path`；agent 返回的 `reward` 由默认 postprocessor 写进 sample。
- ALFWorld：`python scripts/run_qwen3_5_4b_alfworld.py --use-agent-server ...`。
- agent 在集群外（例如 Modal / E2B sandbox）时：用 `--agent-server-port` 固定 replica 端口（第 i 个是 port+i），并在启动 Miles 的环境里设置 `AGENT_SERVER_PUBLIC_URLS='{"<port>": "<外部可达 URL>"}'`。Router 把这个 URL 交给 agent，Miles driver 仍用集群内地址。
- Claude Code + SWE-Gym / SWE-bench Verified 在 Modal 上的完整实验见 [experiments/swe_claude_code](experiments/swe_claude_code/EXPERIMENT_LOG.md)。
- 多模态（Qwen-VL）：agent 用 OpenAI `image_url` 块发图（URL 或 data URI）。agent server 只存图片引用；SGLang 里带新图的消息由 processor 连同文本一起编码，历史图只跑 image processor；trainer 按样本里的图片引用自己处理图片。设计见 [DESIGN.md §6](DESIGN.md#6-multimodal)，DeepEyes 示例见 [examples/experimental/deepeyes](../examples/experimental/deepeyes/README.md)，Modal 上 30 步的验证见 [experiments/deepeyes](experiments/deepeyes/EXPERIMENT_LOG.md)。

## 单独起进程

```bash
python -m agent_server.router --config-json '{"host": "0.0.0.0", "port": 9000}'
python -m agent_server.server --config-json '{"host": "127.0.0.1", "port": 9001,
  "router_url": "http://127.0.0.1:9000", "backend_url": "http://127.0.0.1:30000",
  "hf_checkpoint": "Qwen/Qwen3.5-4B"}'
curl -X POST localhost:9000/sessions -d '{}'   # -> {session_id, replica_url, agent_base_url}
```

`backend_url` 必须是 Miles 的 SGLang（chat completions 要支持 `input_ids` 和 `meta_info.output_token_logprobs`）；`/v1/messages` 需要 sglang-miles 里的 Anthropic 转换函数。

## 测试

```bash
pytest --confcutdir=agent_server agent_server/tests
```

- `test_trajectory.py`：纯 CPU，用玩具渲染模拟 SGLang，测 Trie 的线性、fork、rollback、重试、新根、截断、尾部修正，和平铺参考序列逐 token 比较，以及导出、`max_seq_len`、R3（合成数组）和 Miles v2 hooks。
- `test_continuous_token.py`：Qwen builder（`sglang_patch/continuous_token/`，在 SGLang 里运行）和完整模板渲染逐 token 比较；需要本地有 Qwen3.5 tokenizer（`AGENT_SERVER_TEST_TOKENIZER` 可换成别的 Qwen 模型）。
- `test_replica.py`：进程内 Router + 两个 Replica + 用同一份 CT 代码的假 SGLang；需要 SGLang 和上面的 tokenizer。
- `test_multimodal.py`：每轮都有新图的 mock 轨迹（以及去掉图的纯文本版本），逐轮经过 SGLang 侧的 CT（本地用同一份代码模拟），与 HF processor 一次性 encode 逐 token 比较；SGLang 侧和 trainer 侧的 `pixel_values` / `image_grid_thw` 也逐元素比较。需要本地有 Qwen3-VL processor（`AGENT_SERVER_TEST_PROCESSOR`）。设置 `AGENT_SERVER_TEST_SGLANG_URL` 后改用真实的 patched SGLang。
