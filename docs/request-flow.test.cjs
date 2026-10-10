const test = require("node:test");
const assert = require("node:assert/strict");
const { buildFrames, BLOCK_SIZE, BLOCKS, LAYERS, PROMPT, GENERATED, INITIAL_FREE } = require("./request-flow.js");

test("block ownership partitions the fixed pool at every frame", () => {
  for (const frame of buildFrames()) {
    const ids = [...frame.table, ...frame.free];
    assert.equal(ids.length, BLOCKS);
    assert.equal(new Set(ids).size, BLOCKS);
    assert.ok(ids.every((id) => id >= 0 && id < BLOCKS));
    assert.equal(frame.memory.length, LAYERS);
    assert.ok(frame.memory.every((layer) => layer.length === BLOCKS * BLOCK_SIZE));
  }
});

test("non-contiguous page tables address the same tokens in all layers", () => {
  for (const frame of buildFrames()) {
    for (const layer of frame.memory) {
      layer.forEach((position, slot) => {
        if (position === null || frame.phase === "release") return;
        assert.equal(slot, frame.table[Math.floor(position / BLOCK_SIZE)] * BLOCK_SIZE + position % BLOCK_SIZE);
      });
    }
    if (frame.activeLayer >= 0) {
      assert.ok(frame.writes.every((slot) => frame.memory[frame.activeLayer][slot] !== null));
    }
  }
});

test("prefill writes all prompt tokens but reads direct K/V, then samples first output", () => {
  const frames = buildFrames();
  const layers = frames.filter((frame) => frame.phase === "prefill" && frame.activeLayer >= 0);
  assert.equal(layers.length, LAYERS);
  for (const frame of layers) {
    assert.deepEqual(frame.writes, [20, 21, 22, 23, 4, 5]);
    assert.deepEqual(frame.reads, []);
    assert.equal(frame.cached, 0);
  }
  const sampled = frames.find((frame) => frame.tokens.length === PROMPT.length + 1);
  assert.equal(sampled.cached, PROMPT.length);
  assert.ok(sampled.memory.every((layer) => !layer.includes(PROMPT.length)));
});

test("decode only writes its new input and reads all past plus current KV", () => {
  const frames = buildFrames();
  const steps = frames.filter((frame) => frame.phase === "decode" && frame.activeLayer >= 0);
  assert.equal(steps.length, (GENERATED.length - 1) * LAYERS);
  for (const frame of steps) {
    assert.equal(frame.inputs.length, 1);
    assert.equal(frame.writes.length, 1);
    assert.equal(frame.reads.length, frame.tokens.length);
    assert.deepEqual(frame.reads.map((slot) => frame.memory[frame.activeLayer][slot]),
      Array.from({ length: frame.tokens.length }, (_, i) => i));
    assert.equal(frame.memory[frame.activeLayer][frame.writes[0]], frame.tokens.length - 1);
  }
});

test("a sampled boundary token is allocated only at its next decode", () => {
  const frames = buildFrames();
  const pending = frames.find((frame) => frame.tokens.length === 9);
  assert.deepEqual(pending.table, [5, 1]);
  assert.equal(pending.cached, 8);
  const next = frames[frames.indexOf(pending) + 1];
  assert.deepEqual(next.table, [5, 1, 6]);
  assert.ok(next.memory.every((layer) => !layer.includes(8)));
});

test("release preserves GPU contents and full-block hashes, not the request table", () => {
  const frames = buildFrames();
  const last = frames.at(-1);
  const before = frames.at(-2);
  assert.equal(last.status, "FINISHED");
  assert.equal(last.tokens.length, PROMPT.length + GENERATED.length);
  assert.deepEqual(last.table, []);
  assert.equal(last.cached, 0);
  assert.deepEqual(last.memory, before.memory);
  assert.deepEqual(last.hashes, [5, 1]);
  assert.deepEqual(last.free, [...INITIAL_FREE.slice(3), 6, 1, 5]);
  assert.ok(last.memory.every((layer) => layer.filter((value) => value !== null).length === 10));
  assert.ok(last.memory.every((layer) => !layer.includes(10)));
  assert.equal(frames[0].tokens.length, 0);
});
