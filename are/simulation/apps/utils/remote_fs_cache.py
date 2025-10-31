# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.


import logging
import os
import tempfile
import threading
from typing import Any

from fsspec import AbstractFileSystem
from fsspec.core import url_to_fs
from fsspec.implementations.cached import WholeFileCacheFileSystem

logger = logging.getLogger(__name__)


class RemoteFsCache:
    """
    A global cache for remote filesystem listings and file stats.

    This cache is designed to reduce the number of calls to remote filesystems
    (e.g., S3, HuggingFace) when running multiple scenarios that share the same
    remote filesystem path. It caches:

    1. File/directory listings (from find() calls)
    2. File stats (size, mode) loaded lazily

    The cache is thread-safe and shared across all FallbackFileSystem instances.
    """

    def __init__(self):
        self._lock = threading.RLock()
        # Cache structure:
        # {
        #   "remote_uri": {
        #     "fs": AbstractFileSystem,  # Raw filesystem
        #     "cached_fs": WholeFileCacheFileSystem,  # Cached filesystem wrapper
        #     "root": str,
        #     "file_list": set[str],  # Set of relative paths
        #     "stats": {  # Cached stats for individual files
        #       "relative_path": {"size": int, "mode": int}
        #     }
        #   }
        # }
        self._cache: dict[str, dict[str, Any]] = {}
        self._cache_storage = os.path.join(tempfile.gettempdir(), "are_remote_fs_cache")

    def get_or_create_fs_entry(
        self,
        remote_uri: str,
        fs: AbstractFileSystem | None = None,
        root: str | None = None,
    ) -> tuple[AbstractFileSystem, str, set[str]]:
        """
        Get or create a cache entry for the given remote URI.

        Returns a tuple of (filesystem, root_path, file_list).
        If the entry doesn't exist, it will be created by scanning the remote filesystem.

        :param remote_uri: The remote filesystem URI (e.g., "s3://bucket/path" or "hf://...")
        :param fs: Optional filesystem instance (if None, will be resolved from URI)
        :param root: Optional root path (if None, will be resolved from URI)
        :return: Tuple of (filesystem, root_path, set of relative paths)
        """
        with self._lock:
            if remote_uri in self._cache:
                entry = self._cache[remote_uri]
                return entry["fs"], entry["root"], entry["file_list"]

            # Create new entry
            logger.info(f"Initializing remote filesystem cache for: {remote_uri}")

            # Use provided fs/root or resolve from URI
            if fs is None or root is None:
                fs, root = url_to_fs(remote_uri)

            # Scan the remote filesystem once to get all files
            file_list = set()
            try:
                if fs.exists(root):
                    all_paths = fs.find(root, withdirs=True, detail=False)
                    for path_info in all_paths:
                        rel_path = "/" + os.path.relpath(path_info, root)
                        if rel_path != "/.":
                            file_list.add(rel_path)
                    logger.info(
                        f"Cached {len(file_list)} files from remote filesystem: {remote_uri}"
                    )
                else:
                    logger.warning(f"Remote path does not exist: {remote_uri}")
            except Exception as e:
                logger.error(
                    f"Failed to scan remote filesystem {remote_uri}: {e}", exc_info=True
                )

            # Store in cache
            self._cache[remote_uri] = {
                "fs": fs,
                "root": root,
                "file_list": file_list,
                "stats": {},
            }

            return fs, root, file_list

    def get_file_stats(self, remote_uri: str, rel_path: str) -> dict[str, Any] | None:
        """
        Get cached file stats for a relative path, or None if not cached.

        :param remote_uri: The remote filesystem URI
        :param rel_path: The relative path (e.g., "/path/to/file")
        :return: Dict with "size" and "mode" keys, or None if not cached
        """
        with self._lock:
            if remote_uri not in self._cache:
                return None
            return self._cache[remote_uri]["stats"].get(rel_path)

    def set_file_stats(
        self, remote_uri: str, rel_path: str, size: int, mode: int
    ) -> None:
        """
        Cache file stats for a relative path.

        :param remote_uri: The remote filesystem URI
        :param rel_path: The relative path (e.g., "/path/to/file")
        :param size: File size in bytes
        :param mode: File mode (permissions)
        """
        with self._lock:
            if remote_uri not in self._cache:
                logger.warning(
                    f"Attempted to cache stats for uninitialized URI: {remote_uri}"
                )
                return
            self._cache[remote_uri]["stats"][rel_path] = {"size": size, "mode": mode}

    def clear(self) -> None:
        """Clear all cached data."""
        with self._lock:
            # Clear the in-memory cache
            # Note: WholeFileCacheFileSystem disk cache persists
            self._cache.clear()
            logger.info("Cleared remote filesystem cache")

    def get_cached_filesystem(self, remote_uri: str) -> AbstractFileSystem:
        """
        Get a cached filesystem for the given remote URI.

        This returns a WholeFileCacheFileSystem that wraps the raw filesystem,
        providing file content caching. The cached filesystem is shared across
        all callers for the same remote URI.

        :param remote_uri: The remote filesystem URI
        :return: Cached filesystem instance
        """
        with self._lock:
            # Ensure the cache entry exists
            self.get_or_create_fs_entry(remote_uri)

            # Check if we already have a cached filesystem
            if "cached_fs" in self._cache[remote_uri]:
                return self._cache[remote_uri]["cached_fs"]

            # Create a new cached filesystem
            raw_fs = self._cache[remote_uri]["fs"]
            cached_fs = WholeFileCacheFileSystem(
                fs=raw_fs,
                cache_storage=self._cache_storage,
            )
            self._cache[remote_uri]["cached_fs"] = cached_fs

            logger.info(
                f"Created shared cached filesystem for {remote_uri} "
                f"at {self._cache_storage}"
            )

            return cached_fs

    def get_cache_stats(self) -> dict[str, dict[str, int]]:
        """
        Get statistics about the cache contents.

        :return: Dict mapping URIs to stats like file count and cached stats count
        """
        with self._lock:
            stats = {}
            for uri, entry in self._cache.items():
                stats[uri] = {
                    "file_count": len(entry["file_list"]),
                    "cached_stats_count": len(entry["stats"]),
                }
            return stats


# Global singleton instance
_global_cache = RemoteFsCache()


def get_remote_fs_cache() -> RemoteFsCache:
    """Get the global RemoteFsCache instance."""
    return _global_cache
