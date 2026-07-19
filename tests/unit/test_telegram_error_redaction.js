"use strict";

const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");

const token = "BOT_TOKEN_SENTINEL_7e57b1";
const chatId = "CHAT_ID_SENTINEL_7e57b1";
const appJs = fs.readFileSync(
  path.resolve(__dirname, "../../web/static/app.js"),
  "utf8",
);

const start = appJs.indexOf("function formatApiError(");
const end = appJs.indexOf("async function fetchJSON(");
assert.ok(start >= 0 && end > start, "Telegram 錯誤淨化 helper 必須位於 fetchJSON 前");
assert.ok(appJs.includes("function isTelegramCredentialPath("));

const logged = [];
const context = {
  logClientError: async (...args) => logged.push(args),
  Promise,
};
vm.createContext(context);
vm.runInContext(
  `${appJs.slice(start, end)}\nglobalThis.formatApiError = formatApiError;\nglobalThis.reportRtTelegramError = reportRtTelegramError;`,
  context,
);
context.logClientError = async (...args) => logged.push(args);

const maliciousResponse = {
  detail: `token=${token}; chat_id=${chatId}`,
  traceback: `https://api.telegram.org/bot${token}/sendMessage`,
};
const safe = context.formatApiError(maliciousResponse, 422, "/api/realtime/telegram");
assert.equal(safe, "Telegram 設定格式不正確，請檢查欄位格式。");
assert.ok(!safe.includes(token));
assert.ok(!safe.includes(chatId));

const hint = {
  textContent: "",
  classList: { remove() {}, add() {} },
};
context.reportRtTelegramError(hint, "測試訊息");
assert.ok(!hint.textContent.includes(token));
assert.ok(!hint.textContent.includes(chatId));
assert.ok(!JSON.stringify(logged).includes(token));
assert.ok(!JSON.stringify(logged).includes(chatId));

for (const name of ["loadRtTelegramSettings", "saveRtTelegramSettings", "testRtTelegram"]) {
  const fnStart = appJs.indexOf(`async function ${name}(`);
  const next = appJs.indexOf("\nasync function ", fnStart + 1);
  const source = appJs.slice(fnStart, next < 0 ? undefined : next);
  assert.ok(!source.includes("e.message"), `${name} 不得將伺服器錯誤原文顯示或寫入日誌`);
  assert.ok(source.includes("reportRtTelegramError"), `${name} 必須使用 Telegram 安全錯誤處理`);
}
