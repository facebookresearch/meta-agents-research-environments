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
