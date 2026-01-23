// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the terms described in the LICENSE file in
// the root directory of this source tree.

import { cleanup, render, screen } from "@testing-library/react";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import AppWrapper from "../App";
import { NotificationsContext } from "../contexts/NotificationsContextProvider";

vi.mock("../relay/RelayEnvironment");

// Mock localStorage
const localStorageMock = {
  getItem: vi.fn(),
  setItem: vi.fn(),
  removeItem: vi.fn(),
  clear: vi.fn(),
};

beforeEach(() => {
  Object.defineProperty(window, "localStorage", {
    value: localStorageMock,
    writable: true,
  });
});

afterEach(() => {
  cleanup();
  vi.clearAllMocks();
});

it("renders Meta Agents Research Environments app", async () => {
  const { container } = render(
    <NotificationsContext.Provider
      value={{ snackPack: [], notify: vi.fn(), clear: vi.fn() }}
    >
      <AppWrapper />
    </NotificationsContext.Provider>,
  );
  // App should render (either in loading state or loaded state)
  expect(container).toBeDefined();
  // Check that something is rendered (loading text or the app content)
  const loadingText = screen.queryByText("Loading...");
  const serverStatus = screen.queryByTestId("server-status");
  expect(loadingText !== null || serverStatus !== null).toBe(true);
});
