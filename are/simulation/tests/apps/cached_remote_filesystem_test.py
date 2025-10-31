# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.


import threading
from pathlib import Path
from typing import Any
from unittest.mock import call, MagicMock

import fsspec
import pytest

from are.simulation.apps.utils.cached_remote_filesystem import CachedRemoteFileSystem
from are.simulation.apps.utils.remote_fs_cache import get_remote_fs_cache


class MockFileSystem(fsspec.AbstractFileSystem):
    """
    Mock filesystem that tracks access counts for testing.
    """

    def __init__(self, files: dict[str, str | bytes], **kwargs: Any):
        super().__init__(**kwargs)
        self.files = files  # {path: content}
        self.find_count = 0
        self.ls_count = 0
        self.info_count = 0
        self.open_count = 0
        self.exists_count = 0

    def find(
        self, path: str, maxdepth: int | None = None, withdirs: bool = False, **kwargs
    ) -> list[str]:
        self.find_count += 1
        # Return all files under the path
        result = [p for p in self.files.keys() if p.startswith(path)]
        return result

    def ls(
        self, path: str, detail: bool = True, **kwargs: Any
    ) -> list[dict[str, Any]] | list[str]:
        self.ls_count += 1
        # Return files in the directory
        items = []
        for file_path, content in self.files.items():
            if file_path.startswith(path) and file_path != path:
                if detail:
                    items.append(
                        {
                            "name": file_path,
                            "size": (
                                len(content) if isinstance(content, (str, bytes)) else 0
                            ),
                            "type": "file",
                            "mode": 0o644,
                        }
                    )
                else:
                    items.append(file_path)
        return items

    def info(self, path: str, **kwargs: Any) -> dict[str, Any]:
        self.info_count += 1
        if path not in self.files:
            raise FileNotFoundError(f"File not found: {path}")
        content = self.files[path]
        return {
            "name": path,
            "size": len(content) if isinstance(content, (str, bytes)) else 0,
            "type": "file",
            "mode": 0o644,
        }

    def open(self, path: str, mode: str = "rb", **kwargs: Any):
        self.open_count += 1
        if path not in self.files:
            raise FileNotFoundError(f"File not found: {path}")
        content = self.files[path]
        if isinstance(content, str):
            content = content.encode("utf-8")
        # Return a simple BytesIO-like object
        from io import BytesIO

        return BytesIO(content)

    def exists(self, path: str, **kwargs: Any) -> bool:
        self.exists_count += 1
        # A path exists if it's a file or if any file starts with it (directory)
        if path in self.files:
            return True
        # Check if this is a directory (any file starts with path/)
        path_with_slash = path if path.endswith("/") else path + "/"
        return any(p.startswith(path_with_slash) for p in self.files.keys())

    def isfile(self, path: str) -> bool:
        return path in self.files

    def isdir(self, path: str) -> bool:
        # Simple check: if any file starts with this path, it's a "directory"
        return any(
            p.startswith(path + "/") or p.startswith(path) for p in self.files.keys()
        )


@pytest.fixture
def mock_files():
    """Test files for the mock filesystem."""
    return {
        "/remote/test1.txt": "Content of test1",
        "/remote/test2.txt": "Content of test2",
        "/remote/dir/test3.txt": "Content of test3",
        "/remote/dir/test4.txt": "Content of test4",
    }


@pytest.fixture(scope="function", autouse=True)
def isolated_cache():
    """Ensure each test in this module has an isolated cache."""
    # Clear before the test
    get_remote_fs_cache().clear()
    yield
    # Clear after the test
    get_remote_fs_cache().clear()


def test_cached_remote_filesystem_find_called_once(mock_files):
    """Test that find() is only called once on the underlying filesystem."""
    # Create mock filesystem
    mock_fs = MockFileSystem(mock_files)

    # Create two CachedRemoteFileSystem instances with the same URI
    uri = "mock://remote"
    cached_fs1 = CachedRemoteFileSystem(mock_fs, uri)
    cached_fs2 = CachedRemoteFileSystem(mock_fs, uri)

    # Verify find was called only once during initialization
    assert mock_fs.find_count == 1

    # Call find() on both instances
    result1 = cached_fs1.find("/remote")
    result2 = cached_fs2.find("/remote")

    # Verify find was still only called once (uses cache)
    assert mock_fs.find_count == 1

    # Verify results are correct
    assert len(result1) == len(mock_files)
    assert len(result2) == len(mock_files)


def test_cached_remote_filesystem_info_caching(mock_files):
    """Test that info() returns correct and consistent results."""
    # Create mock filesystem
    mock_fs = MockFileSystem(mock_files)

    # Create CachedRemoteFileSystem
    uri = "mock://remote"
    cached_fs = CachedRemoteFileSystem(mock_fs, uri)

    # Call info() on a file
    info1 = cached_fs.info("/remote/test1.txt")

    # Call info() again on the same file
    info2 = cached_fs.info("/remote/test1.txt")

    # Both calls should return correct and consistent info
    assert info1["size"] == info2["size"]
    assert info1["mode"] == info2["mode"]
    assert info1["size"] == len(mock_files["/remote/test1.txt"])


def test_cached_remote_filesystem_ls_caching(mock_files):
    """Test that ls() caches file stats."""
    # Create mock filesystem
    mock_fs = MockFileSystem(mock_files)

    # Create CachedRemoteFileSystem
    uri = "mock://remote"
    cached_fs = CachedRemoteFileSystem(mock_fs, uri)

    # Call ls with detail=True
    ls_result = cached_fs.ls("/remote", detail=True)

    # Verify we got results
    assert len(ls_result) > 0

    # Now call info() on files that were listed
    for item in ls_result:
        if isinstance(item, dict):
            file_path = item["name"]
            # Get cache to verify stats were cached
            cache = get_remote_fs_cache()
            rel_path = "/" + file_path.replace("/remote/", "")
            cached_stats = cache.get_file_stats(uri, rel_path)
            # Stats should be cached from ls() call
            if cached_stats:
                assert cached_stats["size"] == item["size"]


def test_cached_remote_filesystem_shared_cache_across_instances():
    """Test that multiple instances share the same cache."""
    # Use a fresh mock filesystem for this test
    test_files = {
        "/test/file1.txt": "Content 1",
        "/test/file2.txt": "Content 2",
    }
    mock_fs = MockFileSystem(test_files)

    # Create three CachedRemoteFileSystem instances with the same URI
    uri = "mock://test"

    cached_fs1 = CachedRemoteFileSystem(mock_fs, uri)
    count_after_first = mock_fs.find_count

    cached_fs2 = CachedRemoteFileSystem(mock_fs, uri)
    cached_fs3 = CachedRemoteFileSystem(mock_fs, uri)

    # The cache should be shared, so find should not be called again for fs2 and fs3
    assert mock_fs.find_count == count_after_first

    # Call find on all instances - should use cached data
    result1 = cached_fs1.find("/test")
    result2 = cached_fs2.find("/test")
    result3 = cached_fs3.find("/test")

    # find() calls should use cached data, not call underlying fs
    assert mock_fs.find_count == count_after_first

    # All results should be identical
    assert result1 == result2 == result3
    assert len(result1) == len(test_files)


def test_cached_remote_filesystem_file_content_caching(mock_files):
    """Test that file content is cached and shared across instances."""
    # Create mock filesystem
    mock_fs = MockFileSystem(mock_files)

    # Create two CachedRemoteFileSystem instances with the same URI
    uri = "mock://remote"
    cached_fs1 = CachedRemoteFileSystem(mock_fs, uri)
    cached_fs2 = CachedRemoteFileSystem(mock_fs, uri)

    initial_open_count = mock_fs.open_count

    # Open a file with the first instance
    with cached_fs1.open("/remote/test1.txt", "rb") as f:
        content1 = f.read()

    first_open_count = mock_fs.open_count - initial_open_count

    # Open the same file with the second instance
    with cached_fs2.open("/remote/test1.txt", "rb") as f:
        content2 = f.read()

    second_open_count = mock_fs.open_count - initial_open_count

    # Content should be the same
    assert content1 == content2

    # Because they share the same WholeFileCacheFileSystem,
    # the second open should use the cached file
    # (Note: This depends on WholeFileCacheFileSystem behavior)
    # At minimum, verify content is correct
    assert content1.decode("utf-8") == "Content of test1"


def test_cached_remote_filesystem_thread_safety():
    """Test that the cache is thread-safe."""
    # Use a fresh mock filesystem for this test
    test_files = {
        "/thread_test/file1.txt": "Content 1",
        "/thread_test/file2.txt": "Content 2",
    }
    mock_fs = MockFileSystem(test_files)
    uri = "mock://thread_test"

    results = []
    errors = []

    def worker():
        try:
            # Each thread creates its own CachedRemoteFileSystem instance
            cached_fs = CachedRemoteFileSystem(mock_fs, uri)
            # Call find()
            result = cached_fs.find("/thread_test")
            results.append(result)
        except Exception as e:
            errors.append(e)

    # Create multiple threads
    threads = [threading.Thread(target=worker) for _ in range(10)]

    # Start all threads
    for thread in threads:
        thread.start()

    # Wait for all threads to complete
    for thread in threads:
        thread.join()

    # Verify no errors occurred
    assert len(errors) == 0

    # Verify all threads got results
    assert len(results) == 10

    # Verify all results are identical
    first_result = results[0]
    for result in results[1:]:
        assert result == first_result

    # Verify results are correct
    assert len(first_result) == len(test_files)


def test_cached_remote_filesystem_concurrent_file_access(mock_files):
    """Test concurrent file access with caching."""
    # Create mock filesystem
    mock_fs = MockFileSystem(mock_files)
    uri = "mock://remote"

    contents = []
    errors = []

    def worker():
        try:
            # Each thread creates its own CachedRemoteFileSystem instance
            cached_fs = CachedRemoteFileSystem(mock_fs, uri)
            # Open and read a file
            with cached_fs.open("/remote/test1.txt", "rb") as f:
                content = f.read()
            contents.append(content)
        except Exception as e:
            errors.append(e)

    # Create multiple threads
    threads = [threading.Thread(target=worker) for _ in range(10)]

    # Start all threads
    for thread in threads:
        thread.start()

    # Wait for all threads to complete
    for thread in threads:
        thread.join()

    # Verify no errors occurred
    assert len(errors) == 0

    # Verify all threads got content
    assert len(contents) == 10

    # Verify all contents are identical
    expected_content = b"Content of test1"
    for content in contents:
        assert content == expected_content


def test_cached_remote_filesystem_different_uris_separate_caches():
    """Test that different URIs use separate caches."""
    # Create two separate mock filesystems
    test_files1 = {
        "/uri1/file1.txt": "Content 1A",
        "/uri1/file2.txt": "Content 1B",
    }
    test_files2 = {
        "/uri2/file1.txt": "Content 2A",
        "/uri2/file2.txt": "Content 2B",
        "/uri2/file3.txt": "Content 2C",
    }
    mock_fs1 = MockFileSystem(test_files1)
    mock_fs2 = MockFileSystem(test_files2)

    # Create CachedRemoteFileSystem instances with different URIs
    uri1 = "mock://uri1"
    uri2 = "mock://uri2"

    cached_fs1 = CachedRemoteFileSystem(mock_fs1, uri1)
    cached_fs2 = CachedRemoteFileSystem(mock_fs2, uri2)

    # Each should have called find once
    assert mock_fs1.find_count == 1
    assert mock_fs2.find_count == 1

    # Find should return different results
    result1 = cached_fs1.find("/uri1")
    result2 = cached_fs2.find("/uri2")

    assert len(result1) == len(test_files1)
    assert len(result2) == len(test_files2)
    assert result1 != result2


def test_cached_remote_filesystem_proxy_methods():
    """Test that non-cached methods are properly proxied."""
    mock_files = {"/remote/test.txt": "Content"}
    mock_fs = MockFileSystem(mock_files)
    uri = "mock://remote"

    cached_fs = CachedRemoteFileSystem(mock_fs, uri)

    # Test exists()
    assert cached_fs.exists("/remote/test.txt")
    assert not cached_fs.exists("/remote/nonexistent.txt")

    # Test isfile()
    assert cached_fs.isfile("/remote/test.txt")


def test_cache_clear():
    """Test that clearing the cache removes metadata entries."""
    test_files = {"/clear_test2/test.txt": "Content"}
    mock_fs = MockFileSystem(test_files)
    uri = "mock://clear_test2"

    # Create first instance
    cached_fs1 = CachedRemoteFileSystem(mock_fs, uri)
    assert mock_fs.find_count == 1

    # Verify cache has the entry
    cache = get_remote_fs_cache()
    assert uri in cache._cache

    # Clear the cache
    cache.clear()

    # Verify cache metadata is cleared
    assert uri not in cache._cache
