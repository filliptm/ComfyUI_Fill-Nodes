import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";
import vm from "node:vm";

const source = await readFile(
  new URL("../web/nodes/ksamplers/FL_KsamplerContextWindow.js", import.meta.url),
  "utf8",
);

test("context-window progress uses stable string node keys", () => {
  assert.match(source, /const nodeKey = \(value\) => String\(value\);/);
  assert.match(source, /INSTANCES\.set\(nodeKey\(node\.id\), inst\);/);
  assert.match(source, /INSTANCES\.get\(nodeKey\(detail\.node\)\)/);
  assert.doesNotMatch(source, /parseInt\(detail\.node/);
});

function harness() {
  class Element {
    constructor() { this.style = {}; this.children = []; this.roles = new Map(); this.events = {}; }
    appendChild(child) { this.children.push(child); }
    querySelector(selector) {
      if (!this.roles.has(selector)) this.roles.set(selector, new Element());
      return this.roles.get(selector);
    }
    addEventListener(name, callback) { this.events[name] = callback; }
    setAttribute(name, value) { this[name] = value; }
    remove() { this.removed = true; }
  }
  const handlers = new Map();
  const timers = new Map();
  const app = { registerExtension(extension) { this.extension = extension; } };
  const context = vm.createContext({
    app,
    api: { addEventListener: (name, callback) => handlers.set(name, callback) },
    document: { createElement: () => new Element(), getElementById: () => null, head: new Element() },
    setTimeout(callback) { const id = timers.size + 1; timers.set(id, callback); return id; },
    clearTimeout(id) { timers.delete(id); },
  });
  vm.runInContext(source.replace(/^import .*;\r?$/gm, "").replace(/^export /gm, ""), context);
  const defaults = vm.runInContext("({...ADVANCED_DEFAULTS})", context);
  const widgets = Object.entries({ seed: 7, steps: 20, context_length: 81, context_overlap: 30, ...defaults }).map(([name, value]) => ({
    name, value, type: "number", computeSize: () => [200, 20], options: {},
  }));
  const node = {
    id: "subgraph:7", constructor: { comfyClass: "FL_KsamplerContextWindow" }, properties: {},
    widgets, inputs: widgets.map((w) => ({name: w.name, link: null, widget: {name: w.name}})), size: [330, 730], setDirtyCanvas() {}, graph: { change() {} },
    addDOMWidget(name, type, container, options) {
      const widget = { name, type, container, options };
      this.widgets.push(widget);
      return widget;
    },
    computeSize() { return [330, 100 + this.widgets.filter((w) => !w.hidden).length * 24]; },
    setSize(size) { this.size = size; },
  };
  app.extension.nodeCreated(node);
  const init = () => { for (const [id, callback] of timers) { timers.delete(id); callback(); } };
  const widget = (name) => widgets.find((w) => w.name === name);
  const role = (name) => widget("context_progress").container.children[0].querySelector(`[data-role="${name}"]`);
  return { context, node, widget, role, init, handlers, timers };
}

test("advanced collapse preserves widget values and serialization order", () => {
  const h = harness();
  const before = h.node.widgets.map((w) => [w.name, w.value]);
  h.init();
  const compactHeight = h.node.size[1];
  assert.equal(h.widget("temporal_unit").hidden, true);
  assert.equal(h.widget("context_length").hidden, undefined);
  h.role("advanced").events.click();
  assert.equal(h.widget("temporal_unit").type, "number");
  assert.equal(h.widget("temporal_unit").hidden, undefined);
  assert.equal(h.widget("context_stride").hidden, true);
  assert.ok(h.node.size[1] > compactHeight);
  h.role("advanced").events.click();
  assert.deepEqual(h.node.widgets.map((w) => [w.name, w.value]), before);
  assert.equal(h.node.size[1], compactHeight);
  assert.ok(compactHeight < 730);
});

test("schedule-dependent controls follow the actual ComfyUI enum values", () => {
  const h = harness();
  h.init();
  h.role("advanced").events.click();
  const schedule = h.widget("context_schedule");
  schedule.value = "standard_uniform";
  schedule.callback();
  assert.equal(h.widget("context_stride").hidden, undefined);
  assert.equal(h.widget("closed_loop").hidden, true);
  schedule.value = "looped_uniform";
  schedule.callback();
  assert.equal(h.widget("closed_loop").hidden, undefined);
  schedule.value = "batched";
  schedule.callback();
  assert.equal(h.widget("context_overlap").hidden, true);
  assert.equal(h.widget("context_overlap").value, 30);
});

test("saved legacy values survive configure and connected advanced sockets stay present", () => {
  const h = harness();
  h.init();
  h.node.id = "nested:42";
  h.widget("temporal_unit").value = "video_frames_4n_plus_1";
  h.node.inputs.find((input) => input.name === "context_stride").link = 3;
  h.node.onConfigure();
  assert.equal(h.widget("temporal_unit").value, "video_frames_4n_plus_1");
  assert.equal(h.node.inputs.find((input) => input.name === "context_stride").link, 3);
  assert.match(h.role("advanced").textContent, /2 overrides/);
  h.handlers.get("fl_context_window_progress")({ detail: { node: "nested:42", status: "resolved", profile: "ltx", video_frames: 81, latent_length: 11 } });
  assert.match(h.role("model").textContent, /LTX.*81 frames \/ 11 latents/);
  h.node.onConnectionsChange();
  assert.match(h.role("model").textContent, /Settings changed/);
});

test("progress accepts string execution IDs and removal cancels pending initialization", () => {
  const h = harness();
  h.init();
  h.handlers.get("executing")({ detail: "subgraph:7" });
  assert.equal(h.role("percent").textContent, "0%");
  h.handlers.get("execution_error")({ detail: { node_id: "subgraph:7", exception_message: "Invalid overlap" } });
  assert.equal(h.role("percent").textContent, "error");
  const pending = harness();
  pending.widget("context_progress").onRemove();
  assert.equal(pending.timers.size, 0);
});
