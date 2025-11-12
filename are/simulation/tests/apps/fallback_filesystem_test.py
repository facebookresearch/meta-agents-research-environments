# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.


from pathlib import Path

import fsspec
import pytest

from are.simulation.apps.utils.fallback_file_system import (
    DEFAULT_MODE,
    FallbackFileSystem,
)
from are.simulation.apps.utils.remote_fs_cache import get_remote_fs_cache

# Test files to create in the main filesystem and fallback
TEST_FILES = {
    "/test1.txt": "Test file content 1 💵",
    "/test2.txt": "Test file content 2 💵💵",
    "/images/photo.jpg": b"Binary image content",
    "/docs/readme.md": "# Documentation\nThis is a test markdown file.",
    "/docs/reports/report.txt": "Report content\nWith multiple lines",
}


@pytest.fixture(autouse=True)
def clear_remote_fs_cache():
    """Clear the global remote filesystem cache before each test to prevent race conditions."""
    cache = get_remote_fs_cache()
    cache.clear()
    # Also clear the cache storage directory to prevent stale files
    import shutil

    cache_storage = cache._cache_storage
    if Path(cache_storage).exists():
        shutil.rmtree(cache_storage)
        Path(cache_storage).mkdir(parents=True, exist_ok=True)
    yield
    # Clean up after the test as well
    cache.clear()
    if Path(cache_storage).exists():
        shutil.rmtree(cache_storage)


@pytest.fixture
def memory_fs():
    return fsspec.filesystem(
        "memory"
    )  # memory is a global fs, so creation here should be found later on


@pytest.fixture
def fallback_dir(memory_fs):
    """Set up the fallback directory with test files."""

    fallback_path = Path("/fallback")

    memory_fs.makedirs(fallback_path, exist_ok=True)
    # Create test files in the fallback directory
    for path, content in TEST_FILES.items():
        full_path = fallback_path / path.lstrip("/")
        memory_fs.makedirs(str(full_path.parent), exist_ok=True)

        if isinstance(content, bytes):
            memory_fs.write_bytes(str(full_path), content)
        else:
            memory_fs.write_text(str(full_path), content, encoding="utf-8")

    return fallback_path


@pytest.fixture
def fallback_fs(tmp_path: Path, fallback_dir: Path):
    """Create a FallbackFileSystem instance with a LocalFileSystem as the underlying filesystem."""

    real_dir = tmp_path / "real"
    real_dir.mkdir(parents=True, exist_ok=True)

    # Create the base filesystem
    base_fs = fsspec.filesystem("file")

    # Create the FallbackFileSystem
    fs = FallbackFileSystem(
        base_fs,
        base_root=str(real_dir),
    )

    # Set the fallback root (which now automatically creates placeholders)
    expected_paths = set(TEST_FILES.keys())
    # Use str(fallback_dir) directly since fallback_dir already has the leading slash
    fs.set_fallback_root(f"memory:/{fallback_dir}", expected_paths)

    return fs, real_dir, fallback_dir


def test_fallback_fs_initialization(fallback_fs, memory_fs):
    """Test that the FallbackFileSystem is properly initialized with placeholders using lazy loading."""
    fs, real_dir, fallback_dir = fallback_fs

    # With lazy loading, fallback_stats should contain lightweight registry entries
    for path, content in TEST_FILES.items():
        # Check that files are empty placeholders in the main directory
        main_file_path = real_dir / path.lstrip("/")
        assert main_file_path.stat().st_size == 0

        # Check that the file has a lazy loading entry in fallback_stats
        assert path in fs.fallback_stats
        assert fs.fallback_stats[path] == {"loaded": False}

    # Verify that we have the expected number of entries
    assert len(fs.fallback_stats) == len(TEST_FILES)


def test_fallback_fs_info_and_ls(fallback_fs, memory_fs):
    """Test that info and ls methods return correct stats before loading."""
    fs, real_dir, fallback_dir = fallback_fs

    # Test path for info
    test_path = str(real_dir / "images/photo.jpg")
    fallback_path = fallback_dir / "images/photo.jpg"

    # Get stats directly from the fallback file using pathlib
    stat_result = memory_fs.info(fallback_path)

    # Check that info returns the correct stats before loading
    info_result = fs.info(test_path)

    # Compare the size and mode which should come from fallback_stats
    assert info_result["size"] == stat_result["size"]
    assert info_result["mode"] == stat_result.get("mode", DEFAULT_MODE)

    # Test path for ls
    images_dir = str(real_dir / "images")

    # Check that ls with detail=True returns the correct stats before loading
    ls_results = fs.ls(images_dir, detail=True)

    # Find the file in the results
    test_file_info = next(
        (item for item in ls_results if item["name"] == test_path), None
    )

    # Compare the file info with stats from pathlib
    assert test_file_info is not None
    assert test_file_info["size"] == stat_result["size"]
    assert test_file_info["mode"] == stat_result.get("mode", DEFAULT_MODE)


def test_fallback_fs_lazy_loading(fallback_fs):
    """Test that files are loaded correctly when accessed."""
    fs, real_dir, _fallback_dir = fallback_fs

    file = "test1.txt"

    # Test path
    test_path_obj = real_dir / file
    test_path = str(test_path_obj)
    expected_content = TEST_FILES["/" + file]

    # Verify file exists initially as a placeholder
    assert fs.exists(test_path)
    assert "/" + file in fs.fallback_stats
    assert test_path_obj.stat().st_size == 0

    # Access the file to trigger lazy loading
    with fs.open(test_path, "r") as f:
        content = f.read()

    # Verify content is correct
    assert content == expected_content

    # Verify the file is now loaded (no longer empty)
    assert test_path_obj.stat().st_size > 0

    # Verify the file is removed from fallback_stats after loading
    assert "/" + file not in fs.fallback_stats


def test_fallback_fs_file_removal(fallback_fs):
    """Test that individual files removed with rm are not recovered from the fallback system."""
    fs, real_dir, _fallback_dir = fallback_fs

    file = "test2.txt"

    # Test path
    test_path_obj = real_dir / file
    test_path = str(test_path_obj)

    # Verify file exists initially
    assert fs.exists(test_path)
    assert "/" + file in fs.fallback_stats

    # Remove the file
    fs.rm(test_path)

    # Verify it's removed from the filesystem
    assert not fs.exists(test_path)

    # Verify it's also removed from fallback_stats
    assert "/" + file not in fs.fallback_stats

    # Try to access the file to see if it gets recovered
    # This should raise a FileNotFoundError since the file should not be recovered
    with pytest.raises(FileNotFoundError):
        with fs.open(test_path, "r") as f:
            f.read()


def test_fallback_fs_directory_removal(fallback_fs):
    """Test that directories removed recursively with rm are not recovered from the fallback system."""
    fs, real_dir, _fallback_dir = fallback_fs

    file = "docs/readme.md"

    # Test paths
    dir_path_obj = real_dir / "docs"
    dir_path = str(dir_path_obj)
    file_path_obj = real_dir / file
    file_path = str(file_path_obj)

    # Verify file exists initially
    assert fs.exists(file_path)
    assert "/" + file in fs.fallback_stats

    # Remove a directory recursively
    fs.rm(dir_path, recursive=True)

    # Verify it's removed from the filesystem
    assert not fs.exists(dir_path)
    assert not fs.exists(file_path)

    # Verify it's also removed from fallback_stats
    assert "/" + file not in fs.fallback_stats

    # Try to access the file to see if it gets recovered
    with pytest.raises(FileNotFoundError):
        with fs.open(file_path, "r") as f:
            f.read()


def test_fallback_fs_move(fallback_fs):
    """Test that the mv function loads files from fallback before moving them."""
    fs, real_dir, _fallback_dir = fallback_fs

    file = "test1.txt"

    # Test paths
    test_path_obj = real_dir / file
    test_path = str(test_path_obj)
    new_path_obj = real_dir / "moved_test1.txt"
    new_path = str(new_path_obj)

    # Verify file exists initially as a placeholder
    assert fs.exists(test_path)
    assert "/" + file in fs.fallback_stats
    assert test_path_obj.stat().st_size == 0

    # Move the file
    fs.mv(test_path, new_path)

    # Verify the original file no longer exists
    assert not fs.exists(test_path)

    # Verify the file exists at the new location
    assert fs.exists(new_path)

    # Verify the file was loaded during the move operation
    # (The mv method should ensure the file is loaded before moving)
    assert new_path_obj.stat().st_size > 0

    # Verify the fallback_stats entry was updated
    assert "/" + file not in fs.fallback_stats

    # If we access the file at the new location, it should already be loaded
    with fs.open(new_path, "r") as f:
        content = f.read()
    assert content == TEST_FILES["/test1.txt"]


def test_fallback_fs_write_before_load(fallback_fs):
    """Test that writing to a file before recovering it from fallback prevents fallback loading."""
    fs, real_dir, _fallback_dir = fallback_fs

    file = "test2.txt"

    # Test paths
    test_path_obj = real_dir / file
    test_path = str(test_path_obj)

    # Verify file exists initially as a placeholder
    assert fs.exists(test_path)
    assert "/" + file in fs.fallback_stats
    assert test_path_obj.stat().st_size == 0

    # Write new content to the file
    new_content = b"This is new content that should replace the fallback content"
    with fs.open(test_path, "wb") as f:
        f.write(new_content)

    # Verify the file is removed from fallback_stats
    assert "/" + file not in fs.fallback_stats

    # Verify the file contains only the new content
    with fs.open(test_path, "rb") as f:
        file_content = f.read()
    assert file_content == new_content

    # Verify the file size matches the new content size
    assert test_path_obj.stat().st_size == len(new_content)

    # Try the same with open() in write mode for a text file
    text_file = "docs/readme.md"
    text_path_obj = real_dir / text_file
    text_path = str(text_path_obj)

    # Verify file exists initially as a placeholder
    assert fs.exists(text_path)
    assert "/" + text_file in fs.fallback_stats
    assert text_path_obj.stat().st_size == 0

    # Write new content using open()
    new_text_content = "New markdown content that should replace fallback"
    with fs.open(text_path, "w") as f:
        f.write(new_text_content)

    # Verify the file is removed from fallback_stats
    assert "/" + text_file not in fs.fallback_stats

    # Verify the file contains only the new content
    with fs.open(text_path, "r") as f:
        file_text_content = f.read()
    assert file_text_content == new_text_content

    # Verify the file size matches the new content size
    assert text_path_obj.stat().st_size == len(new_text_content)


def test_fallback_fs_recursive_directory_move(fallback_fs):
    """Test that the mv function correctly handles recursive directory moves with fallback files."""
    fs, real_dir, _fallback_dir = fallback_fs

    # Setup paths
    docs_dir_obj = real_dir / "docs"
    docs_dir = str(docs_dir_obj)
    docs_file = "docs/readme.md"
    docs_file_path_obj = real_dir / docs_file
    docs_file_path = str(docs_file_path_obj)
    nested_file = "docs/reports/report.txt"
    nested_file_path_obj = real_dir / nested_file
    nested_file_path = str(nested_file_path_obj)
    new_docs_dir_obj = real_dir / "moved_docs"
    new_docs_dir = str(new_docs_dir_obj)
    new_docs_file_path_obj = real_dir / "moved_docs/readme.md"
    new_docs_file_path = str(new_docs_file_path_obj)
    new_nested_file_path_obj = real_dir / "moved_docs/reports/report.txt"
    new_nested_file_path = str(new_nested_file_path_obj)

    # Verify files exist initially as placeholders
    assert fs.exists(docs_file_path)
    assert fs.exists(nested_file_path)
    assert "/" + docs_file in fs.fallback_stats
    assert "/" + nested_file in fs.fallback_stats
    assert docs_file_path_obj.stat().st_size == 0
    assert nested_file_path_obj.stat().st_size == 0

    # Move the directory recursively
    fs.mv(docs_dir, new_docs_dir, recursive=True)

    # Verify the original directory no longer exists
    assert not fs.exists(docs_dir)
    assert not fs.exists(docs_file_path)
    assert not fs.exists(nested_file_path)

    # Verify the directory exists at the new location
    assert fs.exists(new_docs_dir)
    assert fs.exists(new_docs_file_path)
    assert fs.exists(new_nested_file_path)

    # Verify the files were loaded during the move operation
    assert new_docs_file_path_obj.stat().st_size > 0
    assert new_nested_file_path_obj.stat().st_size > 0

    # Verify the fallback_stats entries were updated
    assert "/" + docs_file not in fs.fallback_stats
    assert "/" + nested_file not in fs.fallback_stats

    # Access the files at the new location to ensure they were properly loaded
    with fs.open(new_docs_file_path, "r") as f:
        content = f.read()
    assert content == TEST_FILES["/docs/readme.md"]

    with fs.open(new_nested_file_path, "r") as f:
        content = f.read()
    assert content == TEST_FILES["/docs/reports/report.txt"]


def test_set_fallback_root_resets_stats(tmp_path):
    """Test that set_fallback_root resets placeholders and stats."""
    # Create directories for testing
    real_dir = tmp_path / "real"
    real_dir.mkdir()

    initial_fallback_dir = tmp_path / "initial_fallback"
    initial_fallback_dir.mkdir()

    new_fallback_dir = tmp_path / "new_fallback"
    new_fallback_dir.mkdir()

    # Create the base filesystem
    base_fs = fsspec.filesystem("file")

    # Create the FallbackFileSystem
    fs = FallbackFileSystem(
        base_fs,
        real_dir,
    )

    file = "test.txt"

    # Create a test file in the initial fallback directory
    test_file = initial_fallback_dir / file
    test_file.write_text("Initial content")

    # Set the initial fallback root (which now automatically creates placeholders)
    fs.set_fallback_root(str(initial_fallback_dir))

    # Verify fallback_stats contains the test file
    assert "/" + file in fs.fallback_stats

    new_file = "new_test.txt"

    # Create a different test file in the new fallback directory
    new_test_file = new_fallback_dir / new_file
    new_test_file.write_text("New content")

    # Set the new fallback root (which now automatically creates placeholders)
    fs.set_fallback_root(str(new_fallback_dir))

    # Verify fallback_stats has been reset and contains only the new file
    assert "/" + file not in fs.fallback_stats
    assert "/" + new_file in fs.fallback_stats


def test_set_fallback_root_raises_on_conflict(tmp_path):
    """Test that set_fallback_root raises an exception if there are conflicts."""
    # Create directories for testing
    real_dir = tmp_path / "real"
    real_dir.mkdir()

    fallback_dir = tmp_path / "fallback"
    fallback_dir.mkdir()

    # Create the base filesystem
    base_fs = fsspec.filesystem("file")

    # Create the FallbackFileSystem
    fs = FallbackFileSystem(
        base_fs,
        base_root=real_dir,
    )

    # Create a file in the real directory that would conflict with fallback
    conflict_file_path = real_dir / "conflict.txt"
    conflict_file_path.write_text("Existing content")

    # Create a test file in the fallback directory with the same name
    fallback_file = fallback_dir / "conflict.txt"
    fallback_file.write_text("Fallback content")

    # Attempt to set the fallback root, which should raise an exception
    # (now automatically creates placeholders)
    with pytest.raises(ValueError) as excinfo:
        fs.set_fallback_root(str(fallback_dir))

    # Verify the exception message mentions the conflicting file
    assert "conflict.txt" in str(excinfo.value)
    assert "would be overwritten" in str(excinfo.value)
