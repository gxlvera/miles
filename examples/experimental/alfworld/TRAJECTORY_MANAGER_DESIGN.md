# Trajectory Manager 设计

> 历史方案，已由 [Agent Server](../../../agent_server/DESIGN.md) 替代。新方案重写 agent server，并使用独立搬入的 continuous token，不再复用 Miles TITO。

**内部用增量节点保存轨迹，导出时生成 Miles 原生 `Sample`。** 每个 Session 持有一个 Manager；复用 Miles 的 TITO、message matcher、sample picker、reward postprocessor 和 wire codec。

## 1. 抽象和接口

一个节点对应一次 model generation，保存“本轮新增输入 + 本轮输出”。父子关系表示可复用的上下文前缀。一个 Session 可以有多个根：没有可复用前缀的 compaction 或 sub-agent 请求成为新根，仍属于同一个 episode。

```python
class TrajectoryManager:
    def prepare(self, request_args: dict) -> PreparedTurn: ...
    def commit(self, prepared: PreparedTurn, response: dict) -> int: ...
    def build_leaf_samples(self, max_seq_len: int | None = None) -> list[Sample]: ...
    def tree_metadata(self) -> dict: ...
    def clear(self) -> None: ...
```

| 接口 | 职责 |
| --- | --- |
| `prepare` | 按消息前缀找到父节点；复用 Miles 的请求参数解析和 `TITOTokenizer`，生成实际送给模型的 `input_ids` |
| `commit` | 将成功返回的 generation 写成增量节点；分配递增 `node_id` 并返回 |
| `build_leaf_samples` | 每条根到叶路径生成一个 merged `Sample`；处理长度截断，不修改存储中的节点 |
| `tree_metadata` | 返回现有 hooks 所需的节点、父子关系、completion spans 和 leaf paths |
| `clear` | Session 关闭时释放节点及其数组 |

Session Server 保留 HTTP、后端转发、锁和 Session 生命周期。调用顺序为：

```text
prepare → 调用模型 → commit
collect → build_leaf_samples → Miles sample_picker → default_postprocess → encode_samples
```

`prepare` 和 `commit` 在原 Session 锁内执行，模型调用在锁外。父节点在 `prepare` 时确定；并发请求从同一父节点生成兄弟节点，不能在返回时重新选择父节点。

匹配以完整 turn 边界为准，沿 `delta_messages` 匹配，复用 Miles 的消息等价规则。rollback 是从旧节点继续长出分支，不删除已有节点；相同文本的两次独立 generation 也保留为两个节点。

## 2. 内部数据协议

### PreparedTurn

仅在请求进行期间存在，不常驻到树里。

| 字段 | 类型 | 含义 |
| --- | --- | --- |
| `parent_id` | `int \| None` | 本次请求选中的父节点 |
| `request_body` | `dict` | Miles 解析后的完整请求，含 messages、input_ids 和采样参数 |
| `request_timestamp` | `float` | 请求开始时间，用于保留原有计时信息 |

### TurnNode

| 字段 | 类型 | 含义 |
| --- | --- | --- |
| `node_id` | `int` | Session 内递增 ID，同时作为 commit 顺序 |
| `parent_id` | `int \| None` | 可复用前缀所在的节点 |
| `prefix_len` | `int` | 本轮保留父节点的前多少个 token；根节点为 0 |
| `input_delta` | `list[int]` | 实际 prompt 在 `prefix_len` 之后的 token，含工具反馈和模板 token |
| `output_tokens` | `list[int]` | 本轮实际生成的 token |
| `output_logprobs` | `list[float]` | 与 `output_tokens` 一一对应，保留原始精度 |
| `delta_messages` | `list[dict]` | 新增请求消息及本轮 assistant 消息；保留 reasoning、tool calls 等原字段 |
| `turn_args` | `dict` | 用 Miles `filter_turn_args` 去掉 messages、input_ids 后的已解析请求参数 |
| `finish_reason` | `str` | 正常结束、length 或 abort 等原始状态 |
| `weight_versions` | `WeightVersionsPerCall` | 复用 Miles 类型，记录本轮输出的 policy version spans |
| `r3` | `R3Chunk \| None` | 本轮返回的 routing replay 数据，见下节 |
| `response_id` | `str` | 后端响应 ID，用于关联 agent trace |
| `trace_meta` | `dict` | 请求/完成时间、cache/speculative 计数等小型诊断字段，不放 token 或 R3 副本 |

完整序列按下面的规则恢复：

```text
tokens(node) = tokens(parent)[:prefix_len] + input_delta + output_tokens
prompt_len  = prefix_len + len(input_delta)
completion_span = [prompt_len, prompt_len + len(output_tokens))
```

`prefix_len` 不能简单等于父节点长度：TITO 可能替换末尾少量 stop/template token。保留 Miles 的 `max_trim_tokens` 校验；父节点不可修改，否则会影响其他分支。

Manager 用 `nodes[node_id]` 保存节点，children 索引由 `parent_id` 维护；节点和 payload 数组提交后只读。**不常驻完整 `token_ids`、完整 messages 或原始 `SessionRecord`**；否则只改 Node 字段仍会通过 record/turn_args 留下重复历史。完整路径仅在 TITO 请求、调试读取和 Sample 导出时恢复。

### R3Chunk

```python
class R3Chunk:
    row_start: int
    values: ndarray[int32]  # [num_rows, num_layers, top_k]
```

- `row_start` 使用 Miles/SGLang 的 routing-row 坐标，不是 completion 内偏移。区间是 `[row_start, row_start + num_rows)`。
- **additional R3 模式**：复用 Miles 的 `routed_experts_start_len` 规则，只保存后端返回的增量。父子节点共享历史节点中的数组，不复制前缀。
- **full R3 模式**：`row_start=0`，保存这次调用实际返回的完整数组。不能仅凭 token 前缀相同就丢弃旧行；不同调用的 routing 未必相同。
- 一次解码为连续 `int32` 数组，树中只保留这一份；不同时保存 base64 原文。第一版直接用进程内数组，不引入 BlobStore、磁盘 spill 或压缩系统。

恢复规则：

```text
r3(node) = r3(parent)[:row_start] + values
```

增量继承必须满足 Miles 的稳定前缀检查，并且没有缺行。导出形状严格为 `[len(sample.tokens) - 1, num_layers, top_k]`；启用 replay 后数据缺失就报错，不填零。R3 前缀与 token 前缀分别按各自坐标处理。

Miles 已有 additional R3；这里的改动是明确其存储归属和导出协议，不能把 R3 增量本身当作新增收益。

## 3. 输出 Sample 协议

**直接使用 `miles.utils.types.Sample`，不另造 trainer 协议。** Manager 生成训练字段，原 postprocessor 填 reward，driver 保留原任务身份。

| 字段 | 导出规则 |
| --- | --- |
| `tokens` | 根到叶的实际 TITO token 序列 |
| `response_length` | 从首轮 completion 起到序列末尾的长度，包含中间环境反馈 |
| `loss_mask` | 长度等于 `response_length`；模型生成部分为 1，工具反馈、模板新增 token、外部注入内容为 0 |
| `rollout_log_probs` | 同样对齐 response 区间；生成位置用原 logprob，环境位置填 0 并由 mask 排除 |
| `weight_versions` | 保留各轮版本记录；截断时同步裁剪 spans |
| `rollout_routed_experts` | 按路径重建的连续 R3 数组；未启用时为 `None` |
| `status` | 取实际保留路径末尾的生成状态，长度截断为 `TRUNCATED` |
| `response` | 从 response 区间 decode，仅用于展示，不重新 tokenize |
| `metadata` | 保留原 hooks 所需的 `leaf`、`turn_args`、`accumulated_token_ids`；这些在导出时生成 |
| `group_index / index / rollout_id` | driver 从输入 Sample 继承；同 episode 的所有输出共享 `rollout_id` |
| `reward` | 原 postprocessor 分配 episode terminal reward；第一版不改变 credit assignment |

导出时每条叶路径只维护一组可增长的 token/mask/logprob buffer，按 `prefix_len` 裁剪尾部并追加增量；不先创建每轮完整 Sample 再反复 deepcopy/merge。父节点中被后续 TITO 替换掉的输出尾部，不进入这条路径的训练 mask/logprob。

选完 leaves 后，复用 Miles 原有 postprocessor：共享祖先的 completion 只在一个选中 sample 中计入 loss，其他 sample 保留其上下文但 mask 为 0。第一版沿用“最早提交的选中 leaf 拥有该 completion”的规则。

最终 `Sample` 和 wire payload 仍需要完整序列；这次减少的是 **Session 内常驻的重复历史**，不承诺消除 trainer 输入或网络传输中的所有重复。

## 4. 接入和验证范围

- 在 Session Server v2 的 prepare、commit、leaf assembly 三处接入；保留原 HTTP 协议及 picker/postprocessor/codec。
- 保留原生 Manager 路径用于对照。现有 `--per-turn-samples` 继续控制训练粒度，默认关闭；它与存储实现的选择独立。
- 第一版支持线性多轮、fork、rollback、新根及 TITO 尾部修正；R3 支持 full/additional 两种输入。多模态、OPD、top-k/support replay 暂不接入，启用时显式拒绝。
- 用同一批 trace 比较原生与新实现导出的 tokens、mask、logprobs、版本 spans 和 R3；覆盖分叉、尾部修正、截断及不完整 R3。ALFWorld Qwen3.5-4B 验证 dense e2e；R3 用合成数组验证，不能由 dense e2e 代替。

## 代码对照

- [当前树节点与匹配](../../../miles/rollout/session/v2/tree_trajectory.py)
- [Session prepare 和 commit](../../../miles/rollout/session/v2/session_state.py)
- [现有 R3 请求规则](../../../miles/rollout/session/core.py)
- [现有 Sample 组装与 R3 合并](../../../miles/rollout/session/samples/merge.py)
- [共享 completion 的 loss ownership](../../../miles/rollout/session/v2/postprocessor_hub/default_postprocess.py)
