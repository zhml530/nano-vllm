"use strict";

const BLOCK_SIZE = 4;
const LAYERS = 3;
const BLOCKS = 8;
const PROMPT = ["请", "解释", "一下", "什么", "是", "缓存"];
const GENERATED = ["缓存", "保存", "历史", "计算", "结果"];
const INITIAL_FREE = [5, 1, 6, 0, 2, 3, 4, 7];

function buildFrames() {
  const frames = [];
  const state = {
    tokens: [], table: [], free: [...INITIAL_FREE], hashes: [],
    memory: Array.from({ length: LAYERS }, () => Array(BLOCKS * BLOCK_SIZE).fill(null)),
    cached: 0, status: "尚未提交", phase: "init", activeLayer: -1,
    writes: [], reads: [], inputs: [], selected: 0, round: 0,
  };
  const save = (phase, title, body, source, code, io) => {
    Object.assign(state, { phase, title, body, source, code, io });
    frames.push(JSON.parse(JSON.stringify(state)));
  };
  const clearActivity = () => {
    state.activeLayer = -1;
    state.writes = [];
    state.reads = [];
  };
  const allocate = () => {
    if (!state.free.length) throw new Error("演示所需块数超过内存池容量");
    state.table.push(state.free.shift());
  };
  const hashFullBlocks = () => {
    state.hashes = state.table.slice(0, Math.floor(state.cached / BLOCK_SIZE));
  };
  const forward = (isPrefill) => {
    for (let layer = 0; layer < LAYERS; layer++) {
      clearActivity();
      state.activeLayer = layer;
      for (const position of state.inputs) {
        const slot = state.table[Math.floor(position / BLOCK_SIZE)] * BLOCK_SIZE + position % BLOCK_SIZE;
        state.memory[layer][slot] = position;
        state.writes.push(slot);
      }
      if (!isPrefill) {
        state.reads = Array.from({ length: state.tokens.length }, (_, position) =>
          state.table[Math.floor(position / BLOCK_SIZE)] * BLOCK_SIZE + position % BLOCK_SIZE);
      }
      save(isPrefill ? "prefill" : "decode",
        `${isPrefill ? "Prefill" : `Decode ${state.round}`} · 第 ${layer} 层写入与 attention`,
        isPrefill
          ? `第 ${layer} 层同时处理 6 个输入，生成该层独立的 Q/K/V，并把 K/V 写入橙色格。本次没有已缓存前缀，attention 直接使用本轮产生的 K/V 张量，按因果掩码计算，不从分页池读取。然后 hidden states 传给下一层。`
          : `只为位置 ${state.selected} 的一个新输入计算 Q/K/V。先写入橙色格，再用这个 Q 读取第 ${layer} 层蓝框内的位置 0～${state.selected} 的 KV（包含刚写入的自己）。历史 token 的 K/V 不重新计算。`,
        "layers/attention.py",
        isPrefill
          ? "store_kvcache(k, v, k_cache, v_cache, slot_mapping)\nflash_attn_varlen_func(q, k, v, causal=True,\n                       block_table=None)"
          : "store_kvcache(k, v, k_cache, v_cache, slot_mapping)\nflash_attn_with_kvcache(q, k_cache, v_cache,\n    cache_seqlens=context_lens, block_table=block_tables)",
        isPrefill
          ? "Prefill 的多个 query 并行，但各自只能看自己与更早的位置；不能看到未来 token。"
          : `context_lens = [${state.tokens.length}]；block_tables = [[${state.table.join(", ")}]]。各层地址编号相同，KV 数值不同。`);
    }
    clearActivity();
    state.cached += state.inputs.length;
    hashFullBlocks();
    const outputIndex = state.tokens.length - PROMPT.length;
    state.tokens.push(GENERATED[outputIndex]);
    state.selected = state.tokens.length - 1;
    save(isPrefill ? "prefill" : "decode",
      `采样输出 ${outputIndex + 1}：「${GENERATED[outputIndex]}」`,
      `所有层执行完毕，用最后输入位置的 logits 采样新 token。postprocess 把缓存计数更新为 ${state.cached}，登记已写满块的前缀 hash，并将输出加入序列。新输出位于 ${state.selected}，还没有自己的 KV。${outputIndex === GENERATED.length - 1 ? "已达到 max_tokens = 5，接下来立即释放；这两帧拆开只是为了看清源码中的先后顺序。" : "它将在下一轮 decode 作为唯一的新输入。"}`,
      "engine/scheduler.py",
      "block_manager.hash_blocks(seq)\nseq.num_cached_tokens += seq.num_scheduled_tokens\nseq.append_token(token_id)",
      "绿色 token：全部层已有 KV；橙色虚线 token：刚采样，尚未作为输入执行。");
  };
  save("init", "先准备显存池，不是先来一个请求",
    "ModelRunner 在加载与 warmup 后预分配 KV cache，并把每层 K/V 切片绑定给 attention。这里展示一个已回收、可重新使用的池：8 个编号的空闲顺序不是连续的，格子中的旧内容省略。整个池的容量是 8 × 4 = 32 个 token，不再乘层数。",
    "engine/model_runner.py",
    "kv_cache = torch.empty(2, layers, blocks, block_size,\n                       num_kv_heads, head_dim)\nmodule.k_cache = kv_cache[0, layer_id]\nmodule.v_cache = kv_cache[1, layer_id]",
    "右图为便于比较，把实际张量中分开的 K、V 区域并排展示；每一层都有相同数量的块。");
  state.tokens = [...PROMPT];
  state.status = "WAITING";
  save("allocate", "请求进入等待队列",
    "Tokenizer 的结果在此简化为 6 个示意 token。Sequence 已经持有 token_ids，但 block_table 还是空的。加入 waiting 队列不会计算 KV，也不会重新申请 GPU 显存。",
    "engine/llm_engine.py", "seq = Sequence(token_ids, sampling_params)\nscheduler.add(seq)",
    "本演示只跟踪请求 A；没有 prefix-cache 命中或其他活跃请求。");
  while (state.table.length < Math.ceil(PROMPT.length / BLOCK_SIZE)) allocate();
  state.status = "RUNNING";
  save("allocate", "为整个 prompt 预留 2 个块",
    "ceil(6 / 4) = 2。Scheduler 让 BlockManager 从空闲队列取出 5、1，写入页表 [5, 1]。前 4 个 token 属于物理块 5，后 2 个属于物理块 1。所有层同时预留对应编号，但 KV 此刻仍未计算。",
    "engine/block_manager.py", "block_id = free_block_ids.popleft()\nused_block_ids.add(block_id)\nseq.block_table.append(block_id)",
    "used = {5, 1}，不是每层各维护一份 used。深色块边框仅表示该空间已被请求占用。");
  state.inputs = PROMPT.map((_, index) => index);
  save("prefill", "准备 Prefill：一次送入全部 6 个 token",
    "ModelRunner 把逻辑 token 位置翻译为物理 slot。slot_mapping = [20, 21, 22, 23, 4, 5]。映射有跳跃完全正常：请求的逻辑顺序连续，不代表物理块相邻。",
    "engine/model_runner.py", "slot = block_table[position // block_size] * block_size\n       + position % block_size\nset_context(True, ..., slot_mapping, ..., block_tables=None)",
    "没有 prefix cache 时，cu_seqlens_q 和 cu_seqlens_k 相同，prefill 不传分页读取表。");
  forward(true);
  for (let round = 1; round < GENERATED.length; round++) {
    state.round = round;
    state.inputs = [state.tokens.length - 1];
    state.selected = state.tokens.length - 1;
    const needsBlock = state.table.length < Math.ceil(state.tokens.length / BLOCK_SIZE);
    if (needsBlock) allocate();
    const slot = state.table.at(-1) * BLOCK_SIZE + state.selected % BLOCK_SIZE;
    save("decode", needsBlock ? `Decode ${round} · 跨块边界，追加 block 6` : `Decode ${round} · 复用块内空位`,
      `把上一轮采样的「${state.tokens.at(-1)}」（位置 ${state.selected}）作为输入。${needsBlock
        ? "已有 2 个块只覆盖位置 0～7，现在位置 8 需要第三块。may_append 从空闲池取出 6，页表变成 [5, 1, 6]，没有扩大显存池。"
        : "现有尾块还有容量，不用申请新 block ID。"} 本轮只送入这一个 token，而不是重新送入整个 prompt。`,
      needsBlock ? "engine/block_manager.py" : "engine/model_runner.py",
      needsBlock
        ? "if len(seq) % block_size == 1:\n    seq.block_table.append(self._allocate_block())"
        : "input_ids.append(seq.last_token)\nslot_mapping.append(block_table[-1] * block_size\n                    + last_block_num_tokens - 1)",
      `input_ids = [位置 ${state.selected}]，slot_mapping = [${slot}]。下一步才真正计算并写入 KV。`);
    forward(false);
  }
  const released = [...state.table].reverse();
  state.free.push(...released);
  state.table = [];
  state.cached = 0;
  state.status = "FINISHED";
  state.inputs = [];
  save("release", "请求结束：归还编号，显存池留下",
    "已经生成 5 个输出。deallocate 逆序归还块 6、1、5，并清空请求页表、归零 num_cached_tokens。GPU 张量没有释放，旧 KV 也没有清零。完整块的 hash 仍可用于后续前缀复用；重新分配为新内容时才淘汰旧映射并覆盖数据。",
    "engine/block_manager.py",
    "for block_id in reversed(seq.block_table):\n    block.ref_count -= 1\n    if block.ref_count == 0: free_block_ids.append(block_id)\nseq.num_cached_tokens = 0\nseq.block_table.clear()",
    "最后一个输出（位置 10）没有写入 KV，也不需要写入：请求已经结束。空闲不等于内容为空。");
  return frames;
}

if (typeof module !== "undefined" && module.exports) {
  module.exports = { buildFrames, BLOCK_SIZE, LAYERS, BLOCKS, PROMPT, GENERATED, INITIAL_FREE };
}

if (typeof document !== "undefined") {
  const frames = buildFrames();
  const $ = (id) => document.getElementById(id);
  let index = 0;
  let selected = 0;
  let timer = null;
  const chapters = [...document.querySelectorAll("[data-chapter]")];
  const text = (id, value) => { $(id).textContent = value; };
  const progress = $("progress");
  progress.max = String(frames.length - 1);

  function stop() {
    clearInterval(timer);
    timer = null;
    text("play", "▶ 自动播放");
    $("play").setAttribute("aria-pressed", "false");
  }

  function render() {
    const frame = frames[index];
    progress.value = String(index);
    progress.setAttribute("aria-valuetext", `第 ${index + 1} 步：${frame.title}`);
    text("step-count", `${index + 1} / ${frames.length}`);
    text("step-number", String(index + 1).padStart(2, "0"));
    text("step-title", frame.title);
    text("step-body", frame.body);
    text("status", frame.status);
    text("token-count", frame.tokens.length);
    text("cached-count", frame.cached);
    $("prev").disabled = index === 0;
    $("next").disabled = index === frames.length - 1;
    chapters.forEach((button) => {
      const active = button.dataset.chapter === frame.phase;
      button.classList.toggle("active", active);
      if (active) button.setAttribute("aria-current", "step");
      else button.removeAttribute("aria-current");
    });
    const allLayersContain = (position) => frame.memory.every((layer) => layer.includes(position));
    for (const [id, start, end] of [["prompt", 0, PROMPT.length], ["output", PROMPT.length, frame.tokens.length]]) {
      $(id).replaceChildren();
      frame.tokens.slice(start, end).forEach((value, offset) => {
        const position = start + offset;
        const button = document.createElement("button");
        const cached = allLayersContain(position);
        button.className = `token ${cached ? "cached" : position >= PROMPT.length ? "pending" : ""} ${position === selected ? "selected" : ""}`;
        button.setAttribute("aria-label", `位置 ${position}：${value}，${cached ? "全部层已有 KV" : "尚未在全部层写入 KV"}`);
        button.setAttribute("aria-pressed", String(position === selected));
        button.append(document.createTextNode(value));
        const number = document.createElement("small");
        number.textContent = `p${position}`;
        button.append(number);
        button.addEventListener("click", () => {
          stop();
          selected = position;
          render();
          document.querySelector(`#${id} .token.selected`).focus();
        });
        $(id).append(button);
      });
      if (!$(id).childElementCount) $(id).textContent = id === "prompt" ? "等待请求…" : "尚未采样";
    }
    $("block-table").innerHTML = frame.table.length
      ? frame.table.map((block, logical) => `<div class="table-row"><span>逻辑块 ${logical} · p${logical * BLOCK_SIZE}–${(logical + 1) * BLOCK_SIZE - 1}</span><span class="block-id">→ #${block}</span></div>`).join("")
      : '<p class="mono small">[] — 尚未分配 / 已归还</p>';
    text("used", `{ ${frame.table.join(", ")} }`);
    text("free", `[ ${frame.free.join(", ")} ]`);
    text("hashes", frame.hashes.length ? frame.hashes.map((id) => `#${id}`).join(", ") : "无");
    text("pool-count", `${frame.table.length} / ${BLOCKS} 占用`);
    const block = frame.table[Math.floor(selected / BLOCK_SIZE)];
    const selectedSlot = block === undefined ? -1 : block * BLOCK_SIZE + selected % BLOCK_SIZE;
    $("memory").innerHTML = `<div class="block-heads"><span>物理 ID</span>${Array.from({ length: BLOCKS }, (_, id) => `<span>#${id}</span>`).join("")}</div>` +
      frame.memory.map((layer, layerId) => `<div class="layer ${frame.activeLayer === layerId ? "active" : ""}"><div class="layer-grid"><div class="layer-label">L${layerId}${frame.activeLayer === layerId ? " ◀" : ""}</div>` +
        Array.from({ length: BLOCKS }, (_, id) => {
          const owned = frame.table.includes(id);
          const retained = !owned && layer.slice(id * BLOCK_SIZE, (id + 1) * BLOCK_SIZE).some((value) => value !== null);
          return `<div class="block ${owned ? "owned" : ""} ${retained ? "retained" : ""}" aria-label="第 ${layerId} 层 block ${id}，${owned ? "请求 A 占用" : "空闲"}">` +
            ["K", "V"].map((kind) => `<div class="kv-row"><span class="kv-label">${kind}</span>` +
              Array.from({ length: BLOCK_SIZE }, (_, offset) => {
                const slot = id * BLOCK_SIZE + offset;
                const value = layer[slot];
                const active = frame.activeLayer === layerId;
                return `<span class="cell ${value !== null ? "filled" : ""} ${active && frame.reads.includes(slot) ? "read" : ""} ${active && frame.writes.includes(slot) ? "write" : ""} ${slot === selectedSlot ? "inspect" : ""}" title="L${layerId} ${kind} block ${id}, offset ${offset}, slot ${slot}: ${value === null ? "无本请求 KV" : `token p${value}`}">${value === null ? "·" : value}</span>`;
              }).join("") + "</div>").join("") + "</div>";
        }).join("") + "</div></div>").join("");
    text("memory-caption", frame.phase === "release" ? "淡色格：空闲块中保留的旧内容，不再归请求 A 所有。" : frame.activeLayer >= 0 ? `当前执行 L${frame.activeLayer}；每层使用相同页表，但内容各不相同。` : "显存池大小固定；分配与归还只改变块的使用权。");
    text("selected-token", frame.tokens.length ? `p${selected} · ${frame.tokens[selected]}` : "尚无 token");
    text("formula", !frame.tokens.length ? "等待请求进入…" : block === undefined
      ? `position = ${selected}\nlogical_block = floor(${selected} / 4) = ${Math.floor(selected / BLOCK_SIZE)}\n当前页表没有对应物理块`
      : `position = ${selected}\nlogical_block = floor(${selected} / 4) = ${Math.floor(selected / BLOCK_SIZE)}\nblock_id = table[${Math.floor(selected / BLOCK_SIZE)}] = ${block}\noffset = ${selected} % 4 = ${selected % BLOCK_SIZE}\nslot = ${block} × 4 + ${selected % BLOCK_SIZE} = ${selectedSlot}\n每层：K/V[${block}, ${selected % BLOCK_SIZE}, :, :]`);
    text("address-note", frame.phase === "release" ? "页表已清空，不能再用它访问已归还块；图中只保留了物理残留内容。" : selectedSlot === -1 ? "若这个 token 已采样，scheduler 会在下一轮为其预留块；只有编号分配完成后才能寻址。" : "地址可用不代表 KV 已写入。kernel 的元素偏移 = slot × (num_kv_heads × head_dim)。图中黑色下划线标记所选 slot。");
    const sourceLink = $("source-link");
    sourceLink.textContent = `nanovllm/${frame.source}`;
    sourceLink.href = `../nanovllm/${frame.source}`;
    text("source-code", frame.code);
    text("io-note", frame.io);
  }
  function go(value) {
    index = Math.max(0, Math.min(frames.length - 1, value));
    selected = frames[index].selected;
    if (index === frames.length - 1) stop();
    render();
  }
  function start() {
    if (index === frames.length - 1) go(0);
    timer = setInterval(() => go(index + 1), Number($("speed").value));
    text("play", "Ⅱ 暂停播放");
    $("play").setAttribute("aria-pressed", "true");
  }
  $("play").addEventListener("click", () => timer === null ? start() : stop());
  $("prev").addEventListener("click", () => { stop(); go(index - 1); });
  $("next").addEventListener("click", () => { stop(); go(index + 1); });
  $("reset").addEventListener("click", () => { stop(); go(0); });
  progress.addEventListener("input", () => { stop(); go(Number(progress.value)); });
  $("speed").addEventListener("change", () => { if (timer !== null) { stop(); start(); } });
  chapters.forEach((button) => button.addEventListener("click", () => {
    stop();
    go(frames.findIndex((frame) => frame.phase === button.dataset.chapter));
  }));
  document.addEventListener("visibilitychange", () => { if (document.hidden) stop(); });
  render();
}
