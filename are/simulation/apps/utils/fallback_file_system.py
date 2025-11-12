# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.


import logging
import os
from typing import Any, TypedDict

from fsspec import AbstractFileSystem
from fsspec.core import url_to_fs

logger = logging.getLogger(__name__)

DEFAULT_MODE = 0o644


class LazyFallbackEntry(TypedDict):
    """Registry entry for files that haven't had their stats loaded yet."""

    loaded: bool  # Always False for lazy entries


class LoadedFallbackEntry(TypedDict):
    """Registry entry for files that have had their stats loaded."""

    size: int
    mode: int
    loaded: bool  # Always True for loaded entries


# Union type for fallback_stats values
FallbackEntry = LazyFallbackEntry | LoadedFallbackEntry


class FallbackFileSystem(AbstractFileSystem):
    """
    A filesystem wrapper that provides fallback functionality.

    This filesystem wraps another filesystem and provides a fallback mechanism
    where files are copied from a backup location to the underlying filesystem
    only when they are accessed. This allows for lazy loading of files from a backup.

    The fallback system uses stats to proxy ls info/stats without having to copy the file over.
    This is also used to keep track of which files still need to be recovered.

    Key Features:
        * Lazy loading of files from a backup location
        * Transparent proxying of filesystem operations to the underlying filesystem
        * Tracking of files that still need to be recovered
    """

    fs: AbstractFileSystem
    base_root: str
    fallback_fs: AbstractFileSystem | None = None
    fallback_root: str | None = None
    fallback_stats: dict[str, FallbackEntry] = {}

    def __init__(
        self,
        fs: AbstractFileSystem,
        base_root: str,
    ):
        """
        Initialize the FallbackFileSystem. You need to use set_fallback_root to set the fallback explicitly.

        :param fs: The underlying filesystem to wrap
        :param base_root: The directory of the underlying filesystem that will benefit from the fallback
        :param fallback_root: The root directory of the fallback filesystem
        """
        super().__init__()
        self.fs = fs
        self.base_root = base_root

    def _rel_path(self, path: str) -> str:
        """
        Converts a path in base_root to be relative.

        :param path: Path to convert
        :return: Relative path
        """
        return ("/" + os.path.relpath(path, self.base_root)).replace("//", "/")

    def _get_fallback_stats_for_item(
        self, path: str, item: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Updates an item dictionary with stats from the fallback_stats dictionary if applicable.
        This is used by both ls and info methods to provide accurate stats for files that haven't been loaded yet.
        Uses lazy loading - only fetches stats from fallback when actually needed.

        :param path: Path to the file
        :param item: Item dictionary to update
        :return: Updated item dictionary
        """
        # If this is a file that hasn't been loaded yet but we have a registry entry for it
        rel_path = self._rel_path(path)
        if (
            self.fs.exists(path)
            and self.fs.isfile(path)
            and item["size"] == 0
            and rel_path in self.fallback_stats
        ):
            # Check if we need to lazy load the stats
            fallback_entry = self.fallback_stats[rel_path]

            # If this is a lazy entry (only has "loaded" marker), fetch stats now
            if "loaded" in fallback_entry and not fallback_entry["loaded"]:
                if self.fallback_root is not None and self.fallback_fs is not None:
                    try:
                        fallback_path = os.path.join(
                            self.fallback_root, rel_path.lstrip("/")
                        )
                        if self.fallback_fs.exists(
                            fallback_path
                        ) and self.fallback_fs.isfile(fallback_path):
                            info = self.fallback_fs.info(fallback_path)

                            # Update the registry with actual stats
                            self.fallback_stats[rel_path] = LoadedFallbackEntry(
                                size=info["size"],
                                mode=info.get("mode", DEFAULT_MODE),
                                loaded=True,
                            )

                            logger.debug(f"Lazy loaded stats for {rel_path}")
                        else:
                            # File doesn't exist in fallback, remove from registry
                            del self.fallback_stats[rel_path]
                            return item
                    except Exception as e:
                        logger.warning(f"Failed to lazy load stats for {rel_path}: {e}")
                        # Remove the entry if we can't get stats
                        del self.fallback_stats[rel_path]
                        return item
                else:
                    # No fallback available, remove from registry
                    del self.fallback_stats[rel_path]
                    return item

            # Update with the real stats from fallback (now loaded)
            fallback_stat = self.fallback_stats[rel_path]
            if "size" in fallback_stat and "mode" in fallback_stat:
                item["size"] = fallback_stat["size"]
                item["mode"] = fallback_stat["mode"]

        return item

    def _ensure_file_loaded(self, path: str) -> None:
        """
        Ensures that a file is loaded from the fallback_root if it exists there.
        This is called before any operation that reads file content.
        Only loads files that are in the fallback_stats dictionary to prevent
        recovering files that were removed with rm.

        :param path: Path to the file to ensure is loaded
        """
        if (
            not self.fallback_root
            or not self.fallback_fs
            or not self.fallback_fs.exists(self.fallback_root)
        ):
            return

        if path == ".":
            return  # This is the root directory

        rel_path = self._rel_path(path)

        # Only load files that are in the fallback_stats dictionary
        # This prevents recovering files that were removed with rm
        if rel_path in self.fallback_stats:
            # Check if the file is empty in the underlying filesystem (placeholder)
            if self.fs.exists(path) and self.fs.info(path)["size"] == 0:
                # Get the fallback path
                fallback_path = os.path.join(self.fallback_root, rel_path.lstrip("/"))
                # Only copy if the fallback file exists
                if self.fallback_fs.exists(fallback_path) and self.fallback_fs.isfile(
                    fallback_path
                ):
                    # Copy the file from fallback_root to the underlying filesystem
                    self.fs.makedirs(os.path.dirname(path), exist_ok=True)
                    with self.fallback_fs.open(fallback_path, "rb") as f:
                        self.fs.write_bytes(path, f.read())

                    logger.debug(f"Lazily loaded file from fallback: {path}")
                    # Remove from fallback_stats after loading
                    self.fallback_stats.pop(rel_path, None)

    def set_fallback_root(
        self,
        fallback_root: str,
        expected_paths: set[str] | None = None,
    ) -> None:
        """
        Set the fallback root directory, populate fallback stats, and create placeholders.
        Automatically resets placeholders and stats.
        Raises an exception if the underlying filesystem contains non-empty files
        that would be overwritten by the fallback.

        :param fallback_root: The root directory or URI of the fallback filesystem
        :param expected_paths: Set of paths to populate stats for
        :raises ValueError: If the underlying filesystem contains non-empty files that
                           would be overwritten by the fallback
        """
        # Parse the fallback_root URI to get a filesystem and path
        raw_fs, self.fallback_root = url_to_fs(fallback_root)

        # Wrap remote filesystems with CachedRemoteFileSystem for performance
        if "://" in fallback_root:
            # This is a remote filesystem - wrap it with caching
            from are.simulation.apps.utils.cached_remote_filesystem import (
                CachedRemoteFileSystem,
            )

            self.fallback_fs = CachedRemoteFileSystem(raw_fs, fallback_root)
            logger.info(
                f"Wrapped remote filesystem {fallback_root} with CachedRemoteFileSystem"
            )
        else:
            # Local filesystem - use as-is
            self.fallback_fs = raw_fs

        assert self.fallback_fs is not None
        assert self.fallback_root is not None

        # If no expected paths are provided, we can't check for conflicts
        if expected_paths is None:
            expected_paths = set()
            # If fallback_root exists, scan it to find all potential paths
            if self.fallback_root and self.fallback_fs.exists(self.fallback_root):
                found_paths = self.fallback_fs.find(
                    self.fallback_root, withdirs=True, detail=False
                )
                # Ensure we're working with a list of strings
                if isinstance(found_paths, list):
                    for path_info in found_paths:
                        # path_info should be a string when detail=False
                        if isinstance(path_info, str):
                            # Get the path relative to fallback_root
                            rel_path = os.path.relpath(path_info, self.fallback_root)
                            if rel_path == ".":
                                continue
                            rel_path = "/" + rel_path
                            expected_paths.add(rel_path)

        # Check for conflicts - non-empty files in the underlying filesystem
        # that would be overwritten by the fallback
        conflicts = []
        for path in expected_paths:
            if path == "/":
                continue

            # Calculate the path in the base filesystem
            if self.base_root == "/":
                fs_path = path
            else:
                # Remove leading slash for proper joining with base_root
                rel_path = path.lstrip("/")
                fs_path = os.path.join(self.base_root, rel_path)

            # Check if the file exists and is not empty
            if (
                self.fs.exists(fs_path)
                and self.fs.isfile(fs_path)
                and self.fs.info(fs_path)["size"] > 0
            ):
                conflicts.append(path)

        # If conflicts are found, raise an exception
        if conflicts:
            raise ValueError(
                f"Cannot set fallback root: The following files in the underlying filesystem "
                f"would be overwritten by the fallback: {', '.join(conflicts)}"
            )

        # Clear existing fallback stats
        self.fallback_stats = {}

        # If fallback_root is not set or doesn't exist, we're done
        if self.fallback_root is None or not self.fallback_fs.exists(
            self.fallback_root
        ):
            return

        # Create placeholders and populate fallback stats
        self._create_placeholders_internal(expected_paths)

    def _create_placeholders_internal(self, expected_paths: set[str]) -> None:
        """
        Create placeholder files and directories using lazy approach.
        Only discovers which files exist without getting their stats (much faster).

        :param expected_paths: Set of paths that we expect to see in the base_root and should be backed up by fallback_root. Paths are relative to base_root and fallback_root.
        """
        if self.fallback_root is None or self.fallback_fs is None:
            return

        # Phase 1: Lightweight discovery - use find() to get all existing files
        try:
            all_paths = self.fallback_fs.find(
                self.fallback_root, withdirs=True, detail=False
            )
            existing_files = set()

            for path_info in all_paths:
                rel_path = "/" + os.path.relpath(path_info, self.fallback_root)
                if rel_path != "/.":
                    existing_files.add(rel_path)

            logger.debug(
                f"Found {len(existing_files)} existing files in fallback using find()"
            )

        except Exception as e:
            logger.warning(
                f"Failed to use find() for fallback discovery, falling back to individual checks: {e}"
            )
            # Fallback to the old approach if find() fails
            existing_files = set()
            for rel_path in expected_paths:
                fallback_path = os.path.join(self.fallback_root, rel_path.lstrip("/"))
                try:
                    if self.fallback_fs.exists(fallback_path):
                        existing_files.add(rel_path)
                except Exception as inner_e:
                    logger.warning(
                        f"Failed to check existence of {fallback_path}: {inner_e}"
                    )

        # Phase 2: Create placeholders and lightweight registry
        for rel_path in expected_paths:
            # Calculate the path in the base filesystem
            fs_path = os.path.join(self.base_root, rel_path.lstrip("/"))
            fallback_path = os.path.join(self.fallback_root, rel_path.lstrip("/"))

            # Only create placeholders for files that exist in the fallback
            if rel_path in existing_files:
                # Create an empty placeholder file in the base filesystem
                try:
                    if self.fallback_fs.isdir(fallback_path):
                        self.fs.makedirs(fs_path, exist_ok=True)
                    else:
                        self.fs.makedirs(os.path.dirname(fs_path), exist_ok=True)
                        with self.fs.open(fs_path, "w") as _:
                            pass

                        # Create lightweight registry entry (no stats yet - lazy loading)
                        self.fallback_stats[rel_path] = LazyFallbackEntry(loaded=False)

                except Exception as e:
                    logger.warning(f"Failed to create placeholder for {fs_path}: {e}")

        logger.debug(
            f"Created {len(self.fallback_stats)} placeholder entries for lazy loading"
        )

    # Proxy methods with fallback functionality

    def open(
        self,
        path: str,
        mode: str = "rb",
        block_size: int | None = None,
        cache_options: dict | None = None,
        compression: str | None = None,
        **kwargs: Any,
    ):
        """
        Open a file for reading or writing.

        :param path: Path to the file to open
        :param mode: Mode to open the file in
        :param block_size: Size of blocks to read/write
        :param cache_options: Cache options
        :param compression: Compression format
        :param kwargs: Additional arguments to pass to the file system
        :return: File handle to the opened file
        """
        rel_path = self._rel_path(path)
        # If opening for reading, ensure the file is loaded from fallback if needed
        if ("r" in mode or "a" in mode) and "w" not in mode:
            self._ensure_file_loaded(path)
        # If opening for writing, remove from fallback_stats to prevent reloading
        elif "w" in mode and rel_path in self.fallback_stats:
            # Remove from fallback_stats since we're overwriting the file
            self.fallback_stats.pop(rel_path, None)
            logger.debug(f"Removed {rel_path} from fallback_stats due to write mode")

        return self.fs.open(
            path,
            mode=mode,
            block_size=block_size,
            cache_options=cache_options,
            compression=compression,
            **kwargs,
        )

    def cat(
        self, path: str, recursive: bool = False, on_error: str = "raise", **kwargs: Any
    ) -> bytes | str:
        """
        Read the contents of a file.

        :param path: Path to the file to read
        :param recursive: If True, recursively read files in directories
        :param on_error: What to do on error ('raise' or 'omit')
        :param kwargs: Additional arguments to pass to the file system
        :return: Contents of the file
        """
        # Ensure the file is loaded from fallback if needed
        self._ensure_file_loaded(path)
        # Explicitly cast the return value to ensure it matches the expected return type
        result = self.fs.cat(path, recursive=recursive, on_error=on_error, **kwargs)
        if isinstance(result, (bytes, str)):
            return result
        # This should not happen, but just in case
        raise TypeError(f"Unexpected return type from cat: {type(result)}")

    def cat_file(
        self,
        path: str,
        start: int | None = None,
        end: int | None = None,
        **kwargs: Any,
    ) -> bytes | str:
        """
        Read the contents of a file.

        :param path: Path to the file to read
        :param start: Start byte position
        :param end: End byte position
        :param kwargs: Additional arguments to pass to the file system
        :return: Contents of the file
        """
        # Ensure the file is loaded from fallback if needed
        self._ensure_file_loaded(path)

        # Use a direct approach to ensure we get bytes or str
        with self.fs.open(path, "rb", **kwargs) as f:
            if start is not None:
                f.seek(start)
            if end is not None and start is not None:
                return f.read(end - start)
            elif end is not None:
                return f.read(end)
            else:
                return f.read()

    def ls(
        self, path: str = ".", detail: bool = False, **kwargs: dict[str, Any]
    ) -> list[str | dict[str, Any]]:
        """
        List the contents of a directory.

        :param path: Path to list, defaults to '.'
        :param detail: If True, return detailed information about each file
        :param kwargs: Additional arguments to pass to the file system
        :return: List of files and directories in the directory
        """
        results = self.fs.ls(path, detail=detail, **kwargs)

        # If detail is True, update stats from fallback if needed
        if detail:
            new_results = []
            for item in results:
                if isinstance(item, dict) and item.get("type") == "file":
                    item_path = item["name"]
                    item = self._get_fallback_stats_for_item(item_path, item)
                new_results.append(item)
            return new_results

        return results

    def info(self, path: str, **kwargs: dict[str, Any]) -> dict[str, Any]:
        """
        Get information about a file or directory.

        :param path: Path to the file or directory to get information about
        :param kwargs: Additional arguments to pass to the file system
        :return: Information about the file or directory
        """
        # Get basic info from the file system
        if self.fs.exists(path) and self.fs.isfile(path):
            # For files that need to be loaded, we have two options:
            # 1. If we have fallback stats, use them without loading the file
            info = self.fs.info(path, **kwargs)
            if info["size"] == 0 and self._rel_path(path) in self.fallback_stats:
                return self._get_fallback_stats_for_item(path, info)

            # 2. Otherwise, the file is not fallbacked anymore
            return info

        # For directories, just get the info
        return self.fs.info(path, **kwargs)

    def mv(
        self,
        path1: str,
        path2: str,
        recursive: bool = False,
        maxdepth: int | None = None,
        **kwargs: dict[str, Any],
    ) -> None:
        """
        Move a file or directory.

        :param path1: Path to the file or directory to move
        :param path2: Path to the destination
        :param recursive: If True, move directories recursively
        :param maxdepth: Maximum depth to recurse when recursive=True
        :param kwargs: Additional arguments to pass to the file system
        """
        # Handle different cases based on whether it's a file or directory
        if self.fs.isfile(path1):
            # For a single file, ensure it's loaded if it's in fallback_stats
            if self._rel_path(path1) in self.fallback_stats:
                self._ensure_file_loaded(path1)

            # Perform the move operation
            self.fs.mv(path1, path2, recursive=False, **kwargs)

        elif recursive and self.fs.isdir(path1):
            # For recursive directory moves, we need to handle all files in the directory

            rel_path1 = self._rel_path(path1)

            # Ensure all files in the directory that are in fallback_stats are loaded
            for fallback_path in list(self.fallback_stats.keys()):
                if fallback_path.startswith(rel_path1):
                    # Ensure the file is loaded from fallback if needed
                    self._ensure_file_loaded(
                        os.path.join(self.base_root, fallback_path.lstrip("/"))
                    )

            # Perform the move operation
            self.fs.mv(path1, path2, recursive=True, maxdepth=maxdepth, **kwargs)

        else:
            # For non-recursive directory moves or other cases, just delegate to the underlying filesystem
            self.fs.mv(path1, path2, recursive=recursive, maxdepth=maxdepth, **kwargs)

    def rm(
        self, path: str, recursive: bool = False, maxdepth: int | None = None
    ) -> None:
        """
        Remove a file or directory.

        :param path: Path to the file or directory to remove
        :param recursive: If True, remove the directory and all its contents
        :param maxdepth: Maximum depth to recurse when recursive=True
        """
        was_dir = self.fs.isdir(path)
        # Remove the file or directory from the filesystem
        self.fs.rm(path, recursive=recursive, maxdepth=maxdepth)

        rel_path = self._rel_path(path)

        # Also remove any corresponding entries from fallback_stats
        if recursive and was_dir:
            # If removing a directory recursively, remove all entries that start with this path
            for key in list(self.fallback_stats.keys()):
                if key.startswith(rel_path):
                    del self.fallback_stats[key]
        else:
            # If removing a single file, just remove that entry
            if rel_path in self.fallback_stats:
                del self.fallback_stats[rel_path]

    # proxy the rest directly

    def mkdir(self, path, create_parents=True, **kwargs):
        """
        Create directory entry at path.

        For systems that don't have true directories, may create an for
        this instance only and not touch the real filesystem.

        :param path: location
        :param create_parents: if True, this is equivalent to ``makedirs``
        :param kwargs: may be permissions, etc.
        """
        self.fs.mkdir(
            path, create_parents=create_parents, **kwargs
        )  # not necessary to implement, may not have directories

    def makedirs(self, path, exist_ok=False):
        """
        Recursively make directories.

        Creates directory at path and any intervening required directories.
        Raises exception if, for instance, the path already exists but is a
        file.

        :param path: leaf directory name
        :param exist_ok: If False, will error if the target already exists
        """
        self.fs.makedirs(path, exist_ok=exist_ok)

    def rmdir(self, path):
        """
        Remove a directory, if empty.

        :param path: Path to the directory to remove
        """
        self.fs.rmdir(path)

    # everything else we can proxy automatically
    def __getattr__(self, attr):
        return getattr(self.fs, attr)
