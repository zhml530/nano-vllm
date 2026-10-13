# Nano-vLLM Continuous Batching 实现分析

Nano-vLLM 的 continuous batching（连续批处理）并不是让一个固定 batch 从
prefill 一直运行到所有请求生成结束，而是以一次模型推理为调度周期：

> 每执行一次模型 forward，调度器都会根据当前的 waiting/running 请求重新组成 batch。

实现主要由以下组件协作完成：

- [`LLMEngine`](./nanovllm/engine/llm_engine.py)：驱动逐轮调度和模型执行。
- [`Scheduler`](./nanovllm/engine/scheduler.py)：维护请求队列并动态组 batch。
- [`Sequence`](./nanovllm/engine/sequence.py)：保存单个请求的状态和生成进度。
- [`BlockManager`](./nanovllm/engine/block_manager.py)：管理 paged KV cache。
- [`ModelRunner`](./nanovllm/engine/model_runner.py)：准备变长 batch 并执行模型。
- [`Attention`](./nanovllm/layers/attention.py)：通过 FlashAttention 访问 paged KV cache。

## 1. 总体执行流程

请求生成由 [`LLMEngine.generate()`](./nanovllm/engine/llm_engine.py) 驱动：

```text
prompts
  |
  v
创建 Sequence 并加入 waiting 队列
  |
  v
while 请求尚未全部结束:
    Scheduler.schedule()       动态选择本轮请求
    ModelRunner.run()          执行一次 prefill 或 decode
    Scheduler.postprocess()    更新请求状态并释放资源
```

每轮的核心逻辑位于 `LLMEngine.step()`：

```python
seqs, is_prefill = self.scheduler.schedule()
token_ids = self.model_runner.call("run", seqs, is_prefill)
self.scheduler.postprocess(seqs, token_ids, is_prefill)
```

因此，batch 的生命周期只有一个 scheduling step。下一轮执行时，batch 可以因为新请求、
请求完成、KV cache 不足或请求抢占而发生变化。

## 2. 请求状态和调度队列

每个请求被包装为一个 [`Sequence`](./nanovllm/engine/sequence.py)，状态有三种：

```python
class SequenceStatus(Enum):
    WAITING = auto()
    RUNNING = auto()
    FINISHED = auto()
```

调度器维护两个队列：

- `waiting`：需要执行 prompt prefill，或者被抢占后等待重新计算的请求。
- `running`：prefill 已完成，可以逐 token decode 的请求。
- `FINISHED`：已经生成结束，不再位于调度队列中。

状态转换如下：

```text
                 prefill 完成
WAITING --------------------------> RUNNING
   ^                                  |
   |                                  | EOS / max_tokens
   | KV cache 不足，发生抢占           v
   +------------------------------ FINISHED
```

## 3. 动态组成 Prefill Batch

核心调度逻辑位于
[`Scheduler.schedule()`](./nanovllm/engine/scheduler.py)。

调度器优先从 `waiting` 队列选择需要 prefill 的请求：

```python
while self.waiting and len(scheduled_seqs) < self.max_num_seqs:
```

prefill batch 同时受到两个配置限制：

- `max_num_seqs`：一轮最多调度多少条序列，默认 512。
- `max_num_batched_tokens`：一轮最多处理多少个 prompt token，默认 16384。

配置定义在 [`Config`](./nanovllm/config.py) 中。

### 3.1 Chunked Prefill

如果某个 prompt 大于本轮剩余的 token budget，调度器会只处理其中一部分：

```python
seq.num_scheduled_tokens = min(num_tokens, remaining)
```

但是当前实现只允许 batch 中第一条序列进行 chunked prefill：

```python
if remaining < num_tokens and scheduled_seqs:
    break
```

未完成 prefill 的请求不会采样输出 token：

```python
if is_prefill and seq.num_cached_tokens < seq.num_tokens:
    continue
```

它会留在 `waiting` 队列中，下一轮继续处理剩余 prompt。

### 3.2 Prefill 完成

当整个 prompt 都已经被调度后，请求会从 `waiting` 转入 `running`：

```python
seq.status = SequenceStatus.RUNNING
self.waiting.popleft()
self.running.append(seq)
```

从下一次 decode 调度开始，该请求可以与其他不同长度的请求组成动态 batch。

## 4. Decode Batch

只有本轮没有调度到 prefill 请求时，调度器才会进入 decode 阶段：

```python
while self.running and len(scheduled_seqs) < self.max_num_seqs:
```

每个 running sequence 每轮只计算一个 token：

```python
seq.num_scheduled_tokens = 1
seq.is_prefill = False
scheduled_seqs.append(seq)
```

例如当前有三条序列：

| 请求 | 当前上下文长度 | 本轮模型输入 |
|---|---:|---|
| A | 128 | A 的最后一个 token |
| B | 600 | B 的最后一个 token |
| C | 35 | C 的最后一个 token |

虽然三条请求的上下文长度不同，它们仍然可以组成同一个 decode batch。历史 token
不需要重新作为模型输入，因为对应的 K/V 已经保存在 paged KV cache 中。

[`ModelRunner.prepare_decode()`](./nanovllm/engine/model_runner.py) 为每条序列准备：

```python
input_ids.append(seq.last_token)
positions.append(len(seq) - 1)
context_lens.append(len(seq))
slot_mapping.append(...)
```

其中：

- `input_ids`：每条序列最后一个尚未写入 KV cache 的 token。
- `positions`：该 token 在序列中的位置。
- `context_lens`：每条序列当前的上下文长度。
- `slot_mapping`：新生成的 K/V 应写入的物理 cache slot。
- `block_tables`：每条序列的逻辑 KV block 到物理 block 的映射。

## 5. Paged KV Cache

每条 [`Sequence`](./nanovllm/engine/sequence.py) 都有独立的 `block_table`：

```python
self.block_table = []
```

例如：

```text
Sequence A block_table = [7, 2, 18]
Sequence B block_table = [5]
Sequence C block_table = [11, 9]
```

序列的 KV cache 在逻辑上连续，但物理上可以分布在 GPU cache 的任意 block 中。

`ModelRunner.prepare_block_tables()` 会把不同长度的 block table padding 成矩阵：

```text
A: [ 7,  2, 18]
B: [ 5, -1, -1]
C: [11,  9, -1]
```

decode attention 调用
[`flash_attn_with_kvcache()`](./nanovllm/layers/attention.py)：

```python
o = flash_attn_with_kvcache(
    q.unsqueeze(1),
    k_cache,
    v_cache,
    cache_seqlens=context.context_lens,
    block_table=context.block_tables,
    softmax_scale=self.scale,
    causal=True,
)
```

FlashAttention 根据每条请求自己的 `cache_seqlens` 和 `block_table` 找到历史 K/V。
因此，不需要把不同长度请求的完整上下文 padding 成相同长度，这使得不同长度请求可以
高效地共享同一个 decode batch。

## 6. KV Cache 的分配、扩展和释放

KV cache 被切分成固定大小的 block，默认每个 block 对应 256 个 token。

### 6.1 Prefill 时分配

[`BlockManager.allocate()`](./nanovllm/engine/block_manager.py) 为新请求分配物理
block，并建立该请求的 `block_table`。

### 6.2 Decode 时扩展

调度 decode 前会调用：

```python
self.block_manager.may_append(seq)
```

当当前 token 已经进入新的逻辑 block 时，`BlockManager` 会分配一个新的物理 block。

### 6.3 请求结束时立即释放

`Scheduler.postprocess()` 检测 EOS 和 `max_tokens`：

```python
if (not seq.ignore_eos and token_id == self.eos) \
        or seq.num_completion_tokens == seq.max_tokens:
    seq.status = SequenceStatus.FINISHED
    self.block_manager.deallocate(seq)
    self.running.remove(seq)
```

请求结束后：

1. 从 `running` 队列移除。
2. KV block 引用计数减一。
3. 无引用的 block 返回 free list。
4. 后续请求可以立即复用这些 block 和空出的 batch 位置。

这是 continuous batching 中“请求动态退出、资源动态复用”的关键。

## 7. KV Cache 不足时的抢占

decode 调度前会检查是否有足够空间容纳新的 K/V：

```python
while not self.block_manager.can_append(seq):
```

如果空间不足，调度器会抢占 `running` 队列尾部的请求：

```python
self.preempt(self.running.pop())
```

`Scheduler.preempt()` 会执行：

```python
seq.status = SequenceStatus.WAITING
seq.is_prefill = True
self.block_manager.deallocate(seq)
self.waiting.appendleft(seq)
```

因此当前实现使用的是 recompute 策略：

1. 释放被抢占请求的 KV cache。
2. 将请求重新放入 `waiting` 队列。
3. 后续重新执行 prefill。

它没有实现将 KV cache swap 到 CPU。若完整 block 仍可从 prefix cache 命中，
重新 prefill 时可能复用部分前缀。

## 8. Prefix Cache

[`BlockManager.hash_blocks()`](./nanovllm/engine/block_manager.py) 会按完整 token
block 计算链式 hash：

```text
hash(block 0)
hash(hash(block 0), block 1)
hash(hash(block 1), block 2)
```

新请求进入时，`BlockManager.can_allocate()` 检查是否已经存在相同的前缀 block。
命中后：

```python
seq.num_cached_tokens = num_cached_blocks * self.block_size
```

prefill 只提交尚未缓存的 token。如果存在缓存前缀，
[`Attention.forward()`](./nanovllm/layers/attention.py) 会通过 `block_table`
同时访问已缓存 K/V：

```python
if context.block_tables is not None:
    k, v = k_cache, v_cache
```

Prefix cache 不是 continuous batching 本身，但可以降低新请求进入 batch 时的
prefill 开销。

## 9. `num_kv_heads` 与 Continuous Batching 的关系

[`Attention.num_kv_heads`](./nanovllm/layers/attention.py) 不是调度参数，它描述
模型的 GQA/MQA 结构。

在 [`Qwen3Attention`](./nanovllm/models/qwen3.py) 中，每个 tensor-parallel rank
分别持有一部分 query heads 和 KV heads：

```python
self.num_heads = self.total_num_heads // tp_size
self.num_kv_heads = self.total_num_kv_heads // tp_size
```

Q、K、V 被 reshape 为：

```python
q = q.view(-1, self.num_heads, self.head_dim)
k = k.view(-1, self.num_kv_heads, self.head_dim)
v = v.view(-1, self.num_kv_heads, self.head_dim)
```

当 `num_heads > num_kv_heads` 时，多个 query head 共享较少的 K/V head，这就是
Grouped Query Attention。

它与 continuous batching 的间接关系是 KV cache 容量。每个 KV cache block 的
字节数在 `ModelRunner.allocate_kv_cache()` 中计算：

```python
block_bytes = (
    2
    * num_layers
    * block_size
    * num_kv_heads
    * head_dim
    * dtype.itemsize
)
```

其中 `2` 分别代表 K cache 和 V cache。因此：

```text
num_kv_heads 越少
    -> 每个 token 的 KV cache 越小
    -> 同样显存可以分配更多 KV block
    -> 可以同时保留更多 running sequence
    -> continuous batching 的并发容量更高
```

`Attention` 虽然保存了 `self.num_kv_heads`，但 `forward()` 没有直接使用这个成员。
FlashAttention 可以从 `k`、`v`、`k_cache` 和 `v_cache` 的 tensor shape 推断
KV head 数量。

## 10. 当前实现的特点和限制

这个工程实现的是：

> Iteration-level continuous batching + paged KV cache + chunked prefill +
> prefix cache + recompute preemption。

相对于完整的在线 vLLM serving 系统，它有以下简化：

1. **Prefill 和 decode 不会混在同一次 forward 中。**  
   `schedule()` 只要成功选出 prefill 请求就立即返回；只有没有 prefill batch 时才会
   调度 decode。

2. **Prefill 优先。**  
   如果 `waiting` 队列持续存在可调度请求，已经位于 `running` 队列的 decode 请求可能
   被延后。

3. **`generate()` 是离线接口。**  
   它首先把当前调用传入的所有 prompt 加入队列。底层 `add_request()`、`step()` 和动态
   队列已经具备 iteration-level batching 基础，但项目没有提供持续接收网络请求的
   HTTP serving 层。

4. **抢占采用 recompute。**  
   被抢占请求的 KV cache 被释放，之后重新 prefill，而不是 swap 到 CPU。

5. **Decode 每轮重新组 batch。**  
   已完成请求立即退出，空出的 batch 位置和 KV block 可以在后续轮次中被其他请求使用。

## 11. 总结

Nano-vLLM 的 continuous batching 由 `Scheduler` 在每个模型执行 step 动态重组请求，
并通过 paged KV cache 让上下文长度不同的序列共享同一个 decode batch。请求完成时会
立即退出 batch 并释放 KV block；显存不足时可以抢占请求并通过 recompute 恢复。

`num_kv_heads` 不负责 batch 调度，但它决定每个 token 的 KV cache 显存成本，因此会
直接影响可以同时保留的序列数量和 continuous batching 的并发能力。
