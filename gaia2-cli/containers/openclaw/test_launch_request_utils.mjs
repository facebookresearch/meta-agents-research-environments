// Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
// This source code is licensed under the terms described in the LICENSE file in
// the root directory of this source tree.
import assert from "node:assert/strict";

import {
  classifyLlmRequest,
  isGoogleGenerativeAiCall,
} from "./launch_request_utils.mjs";

assert.equal(
  isGoogleGenerativeAiCall(
    "POST",
    "https://provider.example/v1beta/models/gemini-3.1-pro:generateContent"
  ),
  true
);
assert.equal(
  isGoogleGenerativeAiCall(
    "POST",
    "https://provider.example/v1beta/models/gemini-3.1-pro:streamGenerateContent?alt=sse"
  ),
  true
);
assert.equal(
  isGoogleGenerativeAiCall(
    "POST",
    "https://provider.example/v1beta/models/gemini-3.1-pro:countTokens"
  ),
  false
);
assert.equal(
  isGoogleGenerativeAiCall(
    "GET",
    "https://provider.example/v1beta/models/gemini-3.1-pro:generateContent"
  ),
  false
);

const googleRequest = classifyLlmRequest(
  "https://provider.example/v1/models/gemini-3.1-pro:streamGenerateContent?alt=sse",
  { method: "POST" }
);
assert.equal(googleRequest.isGoogleGenerativeAi, true);
assert.equal(googleRequest.isLLMCall, true);

const openaiResponsesRequest = classifyLlmRequest(
  "https://provider.example/v1/responses",
  { method: "POST" }
);
assert.equal(openaiResponsesRequest.isResponses, true);
assert.equal(openaiResponsesRequest.isLLMCall, true);

const anthropicRequest = classifyLlmRequest(
  "https://provider.example/v1/messages",
  { method: "POST" }
);
assert.equal(anthropicRequest.isAnthropicMessages, true);
assert.equal(anthropicRequest.isLLMCall, true);

const completionsRequest = classifyLlmRequest(
  "https://provider.example/v1/chat/completions",
  { method: "POST" }
);
assert.equal(completionsRequest.isCompletions, true);
assert.equal(completionsRequest.isLLMCall, true);

const nonLlmRequest = classifyLlmRequest(
  "https://provider.example/v1/files",
  { method: "POST" }
);
assert.equal(nonLlmRequest.isLLMCall, false);
