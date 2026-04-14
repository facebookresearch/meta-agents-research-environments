# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
"""Core event loop: scenario simulation engine shared across gaia2-cli and the GAIA2 framework."""

from gaia2_core.event_loop.processor import EventProcessor

__all__ = ["EventProcessor"]
