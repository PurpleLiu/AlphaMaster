"use strict";

const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");

const appJs = fs.readFileSync(
  path.resolve(__dirname, "../../web/static/app.js"),
  "utf8",
);
const match = appJs.match(/function createSerialTaskQueue\(\) \{[\s\S]*?\n\}/);
assert.ok(match, "app.js 必須提供可測試的 Telegram 儲存序列佇列");

const context = { Promise };
vm.createContext(context);
vm.runInContext(`${match[0]}\nglobalThis.enqueue = createSerialTaskQueue();`, context);

async function main() {
  const events = [];
  let releaseFirst;
  const first = context.enqueue(
    () => new Promise((resolve) => {
      events.push("first-start");
      releaseFirst = () => {
        events.push("first-end");
        resolve();
      };
    }),
  );
  const second = context.enqueue(() => {
    events.push("second-start");
  });

  await Promise.resolve();
  assert.deepEqual(events, ["first-start"]);
  releaseFirst();
  await Promise.all([first, second]);
  assert.deepEqual(events, ["first-start", "first-end", "second-start"]);
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
