# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Smoke tests for omnigaia data models."""

from __future__ import annotations

import unittest


class TestImportable(unittest.TestCase):
    """Verify that the core data models are importable."""

    def test_import(self) -> None:
        import omnigaia  # noqa: F401

        self.assertTrue(True)


if __name__ == "__main__":
    unittest.main()
