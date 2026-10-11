# Proxy Server MVP

目标：用自包含的 Proxy Server 作为 Miles Session Server的替代品，复用已有 continuous token，新增多进程架构和增量 prefix trie。MVP 已按本文实现；实现时定下的细节和与原计划的差异见第 7 节。

## 1. 范围和已有代码


- 旧代码来源：`verl` 仓库 `neo-agent-service-dev` 分支上的 **stash@{0}**（2026-08-27，"[temp]agent service implementation"），比分支 HEAD 多约 6k 行，含 `proxy/process.py`；工作区里只剩 `.pyc`。可复用消息规范化、OpenAI 协议处理和 continuous token 算法；重写 chain storage、进程组织和 Miles adapter。
- 已确认旧 `proxy/continuous_tokens.py` 的 factory 会 import `verl.utils.tokenizer.continuous_token_wiring`。因此要复制底层 builder 和必要 helper，改成本目录内依赖；保留来源和许可证。先搬 base + Qwen，既不 import verl，也不使用 Miles TITO。实际来源是 `verl-gxl-ct-rollback-support`（commit e18c3167），而不是旧 proxy 自带的那份：后者每轮为了拿 generation prompt 要把整段历史渲染两遍；前者已合入 verl main 的“generation prompt 与最后一组消息一起渲染”。
- proxy server为miles仓库里的独立folder，可以合理import miles，但是不可以import verl的代码。

## 2. 进程与请求路径

```text
Replica 启动并注册 URL ──→ Proxy Router
Miles rollout ──创建 session──→ Router ──random pick──→ Replica
Miles rollout ←── {session_id, replica_url, agent_base_url} ──┘
Agent ──每轮请求──→ Replica ──→ Miles/SGLang inference router ──→ 模型
Miles rollout ──collect / delete──→ 同一个 Replica
```

一个 Router 进程、N 个 Replica 进程；每个 Replica 内用 async 处理多个 session。Router 只保存已注册的 replica 地址，不转发每轮请求，不保存 trajectory。Session 创建成功后返回实际 owner 地址；agent、collect、delete 都直连 owner。

Miles 已支持多个 Session Server 进程，目前由 rollout driver 随机选实例。这里新增的是独立 Router 和注册流程，多进程本身不是相对 Miles 的新能力。

| 所属 | 接口 | 职责 |
| --- | --- | --- |
| Router | `POST /replicas` | 注册 replica URL |
| Router | `GET /replicas` | 已注册的 replica；Miles 等到数量够了再开始 rollout |
| Router | `POST /sessions` | 随机选 replica，调用其创建接口，返回 session handle |
| Replica | `POST /sessions` | 创建本地 session 和 TrajectoryManager |
| Replica | `POST /sessions/{id}/v1/chat/completions` | continuous token → 推理 → 保存本轮 |
| Replica | `POST /sessions/{id}/v1/messages` | Anthropic 协议（Claude Code），转换后走同一条 chat 路径 |
| Replica | `POST /sessions/{id}/samples` | 导出 merged samples，接收 agent 的 reward/metadata |
| Replica | `DELETE /sessions/{id}` | 释放 session 的全部数据 |

复用 Miles 的进程管理启动和回收服务，等 N 个 replica 注册后开始 rollout。MVP 不做 session 迁移、故障恢复、持久化、动态扩缩容；进程失败直接让本次运行暴露错误。

## 3. TrajectoryManager 和存储

```python
prepare(messages, *, tools, builder, context_key) -> PreparedTurn
commit(prepared, generation) -> TurnNode
leaf_samples(*, max_seq_len, tokenizer) -> list[Sample]  # 每个 leaf 一个原始 sample
tree_metadata() -> dict                                 # Miles v2 picker/postprocessor 读的节点、叶子表
clear() -> None
```

`prepare` 找到可复用的已完成 turn，调用自己的 continuous token 得到实际 `input_ids`；`commit` 保存返回的真实 output tokens/logprobs。父节点在 prepare 时固定，避免并发完成顺序改变分支。prepare/commit 都是同步函数，replica 只有一个 event loop，它们本身就是原子的，所以不加锁；推理等待期间不碰 session 状态。picker/postprocessor 和 reward 在 server 的 collect 里接上（复用 Miles v2 hooks）。

消息前缀 Trie 按 canonical message 的指纹逐层索引，完成 turn 的位置记录 generation node ID；匹配还必须使用相同 tools/template 配置。Trie 只保存增量消息和引用，不保存完整历史作为 key，也不扫描所有 chain。文本相同的独立 generation 仍有独立记录，不能合并它们的 logprobs/version；MVP 对歧义匹配采用最近提交的候选。

| 数据 | 必要字段 |
| --- | --- |
| Session | `session_id`、`evaluation`、采样默认值、manager、`closed`（tools/template 配置是每个请求的，作为 Trie 的根 key） |
| PreparedTurn（临时） | `parent_id`、`prefix_len`、实际 `input_ids`、新增消息、context key、消息指纹、可复用的 R3 行数 |
| TurnNode（常驻） | `node_id`、`parent_id`、`prefix_len`、`input_delta`、`output_tokens`、`output_logprobs`、`delta_messages`、`finish_reason`、`weight_versions`、可选 `r3` |
| R3Chunk（可选） | `row_start`、连续 ndarray `values`，保留后端 dtype |

```text
tokens(node) = tokens(parent)[:prefix_len] + input_delta + output_tokens
```

- `prefix_len` 支持 continuous token 对父路径尾部的修正；父节点不可变。rollback 从旧节点长出新分支，不删除旧分支；无法接续的 compaction 建新根。
- 不在 node、metadata 或 request log 中留完整 messages/input_ids/response 的常驻副本。Trie 和 turn 记录引用同一份新增消息。完整序列只在推理、导出时临时还原。
- R3 解码后只存数组，不保留 base64 副本。后端明确返回 additional rows 时才继承父数据；返回 full rows 就保存 full，不能按 token 前缀相同擅自裁掉 routing。MVP 用合成数组验证存取，MoE replay e2e 不包含在本轮承诺里。
- 暂不做 BlobStore、磁盘 spill、压缩算法或跨 session 去重。收益限定为 session 常驻存储；并发请求的完整输入和最终 Samples 仍占内存。

## 4. Miles 接入与输出协议

新增 `--use-proxy-server`、`--proxy-server-workers N`，与 `--use-session-server` 互斥。Proxy 模式初始化本地 continuous token，不传原来的 `--tito-model qwen35`。ALFWorld 启动脚本的 `--per-turn-samples` 只拆 Session Server v2 的输出，不能和 `--use-proxy-server` 一起用。

- **参数和启动**：两个 flag 直接加在 Miles `arguments.py`（这样没有 `agent_proxy/` 时 Miles 照样能 import）；worker specs 和两处 rollout readiness 只在开启时懒加载 `agent_proxy.miles_adapter`。按原 Session Server 的 `CommandWorkerSpec` 方式管理 Router/Replica。
- **Rollout**：新增本目录的 generate adapter，复用 Miles agent runner。创建 session 后注入 `agent_base_url`；collect 后转换为 Miles `Sample`，最后 delete。ALFWorld harness 无须感知服务变化。
- **推理后端**：旧 `/agent_service/generate` 不能直接复用。按当前 Miles 的方式调用 `/v1/chat/completions`，传明确的 `input_ids`、请求 logprobs，并读取 `meta_info` 中的真实 token IDs；透传 session routing key。连续 token 由本地 builder 负责，reasoning/tool-call 解析复用后端已有能力。
- **输出**：使用 Miles `Sample` 和已有 binary codec，不另造 trainer 协议。核心字段为 `tokens`、`response_length`、`loss_mask`、`rollout_log_probs`、`weight_versions`、可选 `rollout_routed_experts`、`status`、`reward`、`metadata`。
- 每条选中的根到叶路径导出一个 sample；环境和模板 token 的 mask 为 0。长度裁剪同步作用于 tokens/mask/logprobs。同 episode 的多个 sample 保留共同 `rollout_id` 和输入 group 身份；先沿用 episode terminal reward，共享 generation 只在一个选中 sample 中计入 loss。

## 5. 需要写的文件

所有新实现、测试和文档放在 Miles 根目录的 `agent_proxy/`。目录外只加上述薄接入钩子和训练脚本开关。

| 文件 | 工作 |
| --- | --- |
| `router.py` | replica 注册、session 分配 |
| `server.py` | Replica HTTP、session 生命周期、prepare/generate/commit 串联 |
| `trajectory.py` | 消息 Trie、增量节点、路径还原、sample 字段组装 |
| `protocol.py` | SessionHandle、PreparedTurn、TurnNode、Generation、R3Chunk |
| `continuous_token/` | 搬入原 builder 和必要 helper，去掉 verl 依赖 |
| `backend.py` | Miles/SGLang 请求、真实 tokens/logprobs/version/R3 提取 |
| `anthropic.py` | `/v1/messages`：复用 Miles session server 的 Anthropic 转换 |
| `miles_adapter.py` | 进程 specs/readiness、generate 入口、Sample/codec 转换 |
| `tests/`、`DESIGN.md`、`README.md` | 最小验收和使用说明 |

不搬旧 AgentService 的 controller、execution backend、Task/Environment 抽象或 selector registry；这些不属于这次 Proxy 的职责。

## 6. 顺序和工作量

1. 抽出 continuous token，先用固定 token trace 验证不改变旧行为。
2. 实现增量 Trie，验证线性、fork、rollback、新根和边界 token 修正；与同一套 TITO 下的平铺参考序列逐 token 比较。
3. 接 Replica，再接 Router 和多进程；用 fake backend 验证两个 replica 的注册、session 固定归属、collect 和释放。
4. 加 Miles hooks，跑一次真实生成和 collect，再交接 ALFWorld e2e。与 Miles 原生对照前先对齐模板配置，不能预设两套 TITO 输出必然逐 token 相同。

**估计：代码与 CPU 验收约 2–3 个工作日；GPU 联调和第 4 项另留半天到一天。** Router 工作量小，主要时间在 Trie 的训练字段对齐、Qwen 模板兼容和 Miles 接入。若必须两天内完成，先锁住上面的 dense、非流式范围，R3 只做协议与合成 trace，不扩展黑盒 agent 协议。

## 7. 实现中的决定

- **消息指纹**：沿用 Miles `strict_message_matches` 的投影（role、content、reasoning_content、tool_calls，空值归一），另外放宽两点：tool call 只看函数名和参数，参数按 JSON 对象比较；全是 text block 的 content 视为拼接后的字符串。这两点在 Anthropic ↔ OpenAI 转换后常常变，但不影响模型看到的 token。agent 回放时如果丢了 `reasoning_content`，请求会变成新根：token 仍然正确，只是不共享前缀。
- **截断的 turn**（`finish_reason=length`）不进 Trie，永远不当父节点。回放它的请求会挂到更早的 turn 上，截断的文本作为上下文由 CT 重新渲染（mask 0）。Miles v2 在这里直接返回 409。
- **请求必须至少有一条新消息**：Trie 匹配不吃掉最后一条请求消息，所以和某个 turn 完全相同的请求会挂到它的父节点，成为兄弟（重试）。
- **导出**：每个 leaf 一条路径，用一组增长的 buffer 回放 `tokens(parent)[:prefix_len] + input_delta + output_tokens`；`max_seq_len` 和 Miles `truncate_samples_by_total_tokens` 一样按 turn 裁剪。之后交给 Miles 的 `drop_same_prompt_retries` 和 `default_postprocess`，所以 reward 和共享 completion 的 loss 归属与 v2 一致。
- **R3**：`r3(node) = r3(parent)[:row_start] + values`。additional 模式 `row_start = min(prefix_len, len(parent) - 1)`；`--pause-generation-mode retract` 下 `row_start = 0`，每轮存 full。导出时从 leaf 往上只取未被替换的行，拼一次；缺行直接报错，不补零。只用合成数组测过。
- **流式**：OpenAI 和 Anthropic 的 `stream: true` 都是假流式：生成完后一次性发出（和 Miles session server 相同），因为 turn 要等完整回复才能提交。
- **模板**：用 checkpoint 自带的 HF 模板（或 `--chat-template-path`），不用 Miles 的 `qwen3.5_fixed.jinja`。所以和原生 Miles 对照时 token 不会逐个相同。ALFWorld 启动脚本在 proxy 模式下显式传 `--sglang-reasoning-parser qwen3 --sglang-tool-call-parser qwen3_coder`。
- **不支持**（启动时报错）：`--partial-rollout`、`--use-rollout-indexer-replay`、`--use-sampling-support-replay`、`--rollout-top-logprobs-num`、`--group-rm`。也没有 replica 的对外地址配置（对应 `--session-server-external-host`）：sandbox 外的 agent 需要能直连 replica 注册时用的地址。
