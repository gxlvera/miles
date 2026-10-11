# agent_proxy

Miles 的 agent session 服务：一个 Router 加 N 个 Replica 进程。Replica 用 continuous token（从 verl 搬来，不 import verl）增量构造 prompt，用消息前缀 Trie 找父节点，每个 turn 节点只存自己新增的 token、logprob 和消息。设计见 [DESIGN.md](DESIGN.md)。

## 在 Miles 里用

```bash
python train.py ... \
  --use-proxy-server --proxy-server-workers 4 \
  --custom-generate-function-path agent_proxy.miles_adapter.generate \
  --custom-agent-function-path <your agent fn> \
  --sglang-reasoning-parser qwen3 --sglang-tool-call-parser qwen3_coder
```

- `--use-proxy-server` 不能和 `--use-session-server` 同时用，也不需要 `--tito-model`；prompt 用 `--hf-checkpoint` 自带的聊天模板（可用 `--chat-template-path`、`--apply-chat-template-kwargs` 覆盖）。
- agent 函数签名是 `async def run(base_url, prompt, request_kwargs, metadata, **kwargs) -> dict | None`。拿到 `base_url` 后请求 `{base_url}/v1/chat/completions`，Claude Code 这类 Anthropic 客户端用 `{base_url}/v1/messages`；返回的 dict 作为 agent metadata（含 `reward`）随 collect 一起提交。agent 要原样回放上一轮的 assistant 消息（包括 `reasoning_content` 和 tool calls），否则会开新根。
- picker/postprocessor 沿用 `--session-sample-picker-path` / `--session-sample-postprocessor-path`；agent 返回的 `reward` 由默认 postprocessor 写进 sample。
- ALFWorld：`python scripts/run_qwen3_5_4b_alfworld.py --use-proxy-server ...`。

## 单独起进程

```bash
python -m agent_proxy.router --config-json '{"host": "0.0.0.0", "port": 9000}'
python -m agent_proxy.server --config-json '{"host": "127.0.0.1", "port": 9001,
  "router_url": "http://127.0.0.1:9000", "backend_url": "http://127.0.0.1:30000",
  "hf_checkpoint": "Qwen/Qwen3.5-4B"}'
curl -X POST localhost:9000/sessions -d '{}'   # -> {session_id, replica_url, agent_base_url}
```

`backend_url` 必须是 Miles 的 SGLang（chat completions 要支持 `input_ids` 和 `meta_info.output_token_logprobs`）；`/v1/messages` 需要 sglang-miles 里的 Anthropic 转换函数。

## 测试

```bash
pytest --confcutdir=agent_proxy agent_proxy/tests
```

- `test_trajectory.py`：纯 CPU，用玩具 builder 测 Trie 的线性、fork、rollback、重试、新根、截断、尾部修正，和平铺参考序列逐 token 比较，以及导出、`max_seq_len`、R3（合成数组）和 Miles v2 hooks。
- `test_continuous_token.py`：搬来的 Qwen builder 和完整模板渲染逐 token 比较；需要本地有 Qwen3.5 tokenizer（`AGENT_PROXY_TEST_TOKENIZER` 可换成别的 Qwen 模型）。
- `test_replica.py`：进程内 Router + 两个 Replica + 假 SGLang；需要 SGLang 和上面的 tokenizer。
