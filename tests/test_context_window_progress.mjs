import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

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
