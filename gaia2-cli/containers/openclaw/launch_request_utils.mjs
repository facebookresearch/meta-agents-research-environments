// Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
// This source code is licensed under the terms described in the LICENSE file in
// the root directory of this source tree.
export function isGoogleGenerativeAiCall(method, urlStr) {
  return (
    method === "POST" &&
    urlStr.includes("/models/") &&
    /:(generateContent|streamGenerateContent)(?:[?#]|$)/.test(urlStr)
  );
}

// OpenClaw's openai-completions provider turns a forwarded `delta.reasoning`
// into a thinking block, re-sends it on every later request and counts it in
// its tool-result context guard, even when the served chat template never
// renders past reasoning. GAIA2_OPENCLAW_FORWARD_REASONING=0 (or false/off/no)
// keeps it away from OpenClaw; the raw response and trace still carry it.
export function forwardedReasoning(msg, env) {
  const flag = String(env?.GAIA2_OPENCLAW_FORWARD_REASONING ?? "").trim().toLowerCase();
  if (["0", "false", "off", "no"].includes(flag)) return "";
  return typeof msg?.reasoning === "string" ? msg.reasoning : "";
}

export function classifyLlmRequest(url, init) {
  const urlStr = typeof url === "string" ? url : url?.toString?.() || "";
  const method = init?.method?.toUpperCase() || "";
  const isCompletions =
    method === "POST" && urlStr.includes("/chat/completions");
  const isResponses = method === "POST" && urlStr.includes("/responses");
  const isAnthropicMessages =
    method === "POST" && urlStr.includes("/v1/messages");
  const isGoogleGenerativeAi = isGoogleGenerativeAiCall(method, urlStr);

  return {
    urlStr,
    isCompletions,
    isResponses,
    isAnthropicMessages,
    isGoogleGenerativeAi,
    isLLMCall:
      isCompletions ||
      isResponses ||
      isAnthropicMessages ||
      isGoogleGenerativeAi,
  };
}
