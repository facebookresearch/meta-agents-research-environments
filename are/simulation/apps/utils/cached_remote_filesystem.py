# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.


import logging
import os
from typing import Any

from fsspec import AbstractFileSystem

from are.simulation.apps.utils.remote_fs_cache import get_remote_fs_cache

logger = logging.getLogger(__name__)


class CachedRemoteFileSystem(AbstractFileSystem):
    """
    A filesystem wrapper that caches listings and stats for remote filesystems.

    This filesystem wraps any fsspec filesystem and provides global caching
    for file listings and stats. It's designed to reduce API calls to remote
    filesystems (e.g., S3, HuggingFace) when running multiple scenarios that
    share the same remote path.

    The cache is global and thread-safe, shared across all instances of this
    filesystem with the same remote URI.

    Key Features:
        * Global caching of file listings (from find() calls)
        * Global caching of file stats (size, mode) loaded lazily
        * Thread-safe cache shared across all instances
        * Transparent pass-through for all other operations
    """

    def __init__(
        self,
        fs: AbstractFileSystem,
        remote_uri: str,
        **kwargs: Any,
    ):
        """
        Initialize the CachedRemoteFileSystem.

        :param fs: The underlying remote filesystem to wrap
        :param remote_uri: The URI of the remote filesystem (for cache key)
        :param kwargs: Additional arguments to pass to AbstractFileSystem
        """
        super().__init__(**kwargs)
        self.remote_uri = remote_uri
        self.cache = get_remote_fs_cache()

        # Extract root path from URI
        # For "mock://remote" the root should be "/remote"
        if "://" in remote_uri:
            root = "/" + remote_uri.split("://", 1)[1]
        else:
            root = remote_uri if remote_uri.startswith("/") else "/" + remote_uri

        # Pre-populate metadata cache on initialization
        # Pass the filesystem directly to avoid URL resolution
        self.cache.get_or_create_fs_entry(self.remote_uri, fs=fs, root=root)

        # Get the shared cached filesystem from the global cache
        # This ensures all instances with the same remote_uri share
        # the same file cache, avoiding redundant downloads
        self.fs = self.cache.get_cached_filesystem(remote_uri)

        logger.debug(f"Initialized CachedRemoteFileSystem for {remote_uri}")

    def find(
        self,
        path: str,
        maxdepth: int | None = None,
        withdirs: bool = False,
        detail: bool = False,
        **kwargs: Any,
    ) -> list[str] | list[dict[str, Any]]:
        """
        List all files below path using cached data when possible.

        :param path: The path to search from
        :param maxdepth: Maximum depth to search
        :param withdirs: Include directories in results
        :param detail: If True, return list of dicts with file info
        :param kwargs: Additional arguments
        :return: List of file paths or list of dicts if detail=True
        """
        # Get cached data (doesn't call find() again if already cached)
        if self.remote_uri in self.cache._cache:
            entry = self.cache._cache[self.remote_uri]
            root = entry["root"]
            cached_files = entry["file_list"]

            # Convert cached relative paths to absolute paths
            result = []
            for rel_path in cached_files:
                abs_path = os.path.join(root, rel_path.lstrip("/"))
                # Filter by path prefix if specified
                if abs_path.startswith(path):
                    result.append(abs_path)

            logger.debug(f"find() returned {len(result)} cached files for {path}")
            if detail:
                # Return detailed info for each file
                detailed_result: list[dict[str, Any]] = []
                for file_path in result:
                    try:
                        info = self.info(file_path)
                        detailed_result.append(info)
                    except Exception:
                        pass
                return detailed_result
            return result
        else:
            # Fall back to underlying filesystem if not cached
            result_from_fs = self.fs.find(
                path, maxdepth=maxdepth, withdirs=withdirs, detail=detail, **kwargs
            )
            # Type narrowing for return value
            if isinstance(result_from_fs, dict):
                # If it's a dict, we need to handle it - but find() should return list
                return []
            return result_from_fs

    def ls(
        self, path: str, detail: bool = True, **kwargs: Any
    ) -> list[dict[str, Any]] | list[str]:
        """
        List directory contents, with optional caching of stats.

        :param path: Path to list
        :param detail: If True, return detailed information
        :param kwargs: Additional arguments
        :return: List of files/directories
        """
        # Use the underlying filesystem for ls
        results = self.fs.ls(path, detail=detail, **kwargs)

        if detail:
            # Cache stats for files
            for item in results:
                if isinstance(item, dict) and item.get("type") == "file":
                    try:
                        # Get relative path for caching
                        _, root, _ = self.cache.get_or_create_fs_entry(self.remote_uri)
                        rel_path = "/" + os.path.relpath(item["name"], root)

                        # Cache the stats
                        self.cache.set_file_stats(
                            self.remote_uri,
                            rel_path,
                            item.get("size", 0),
                            item.get("mode", 0o644),
                        )
                    except Exception as e:
                        logger.debug(f"Failed to cache stats for {item['name']}: {e}")

        return results

    def info(self, path: str, **kwargs: Any) -> dict[str, Any]:
        """
        Get file info, using cached stats when available.

        :param path: Path to get info for
        :param kwargs: Additional arguments
        :return: File info dictionary
        """
        # Try to get cached stats first
        try:
            _, root, _ = self.cache.get_or_create_fs_entry(self.remote_uri)
            rel_path = "/" + os.path.relpath(path, root)
            cached_stats = self.cache.get_file_stats(self.remote_uri, rel_path)

            if cached_stats:
                # We have cached stats, but we still need other metadata
                # Get full info from filesystem
                info = self.fs.info(path, **kwargs)

                # Update with cached stats (avoid re-fetching if expensive)
                info["size"] = cached_stats["size"]
                info["mode"] = cached_stats["mode"]

                logger.debug(f"Used cached stats for {path}")
                return info
        except Exception as e:
            logger.debug(f"Failed to use cached stats for {path}: {e}")

        # Fall back to direct filesystem call
        info = self.fs.info(path, **kwargs)

        # Cache the stats for future use
        try:
            _, root, _ = self.cache.get_or_create_fs_entry(self.remote_uri)
            rel_path = "/" + os.path.relpath(path, root)
            self.cache.set_file_stats(
                self.remote_uri, rel_path, info.get("size", 0), info.get("mode", 0o644)
            )
        except Exception as e:
            logger.debug(f"Failed to cache stats for {path}: {e}")

        return info

    # Proxy all other methods to the underlying filesystem
    def __getattr__(self, attr: str) -> Any:
        """Proxy all other methods to the underlying filesystem."""
        return getattr(self.fs, attr)

    def open(
        self,
        path: str,
        mode: str = "rb",
        block_size: int | None = None,
        cache_options: dict | None = None,
        compression: str | None = None,
        **kwargs: Any,
    ) -> Any:
        """
        Open a file.

        :param path: Path to open
        :param mode: Mode to open in
        :param block_size: Block size for reading
        :param cache_options: Cache options
        :param compression: Compression codec
        :param kwargs: Additional arguments
        :return: File handle
        """
        return self.fs.open(
            path,
            mode=mode,
            block_size=block_size,
            cache_options=cache_options,
            compression=compression,
            **kwargs,
        )
