# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
"""
Standalone CLI for the Gaia2 Files (filesystem) app.

Binary name: ``cloud-drive``
App class name (for events): ``SandboxLocalFileSystem``

Operates directly on the real filesystem at FS_ROOT (default:
``$GAIA2_STATE_DIR/filesystem/``).  No JSON state file is loaded at runtime —
the directory tree *is* the state.
"""

import os
import re
import sys

import click

from gaia2_cli.shared import (
    build_schema,
    cli_error,
    json_output,
    log_action,
    set_app,
    set_log_class,
)

APP_NAME = "Files"

# ---------------------------------------------------------------------------
# Filesystem root
# ---------------------------------------------------------------------------

_FS_ROOT: str | None = None


def _get_fs_root() -> str:
    global _FS_ROOT
    if _FS_ROOT is None:
        env = os.environ.get("GAIA2_FS_ROOT")
        if env:
            _FS_ROOT = os.path.abspath(env)
        else:
            state_dir = os.environ.get("GAIA2_STATE_DIR", "/workspace/state")
            _FS_ROOT = os.path.abspath(os.path.join(state_dir, "filesystem"))
    return _FS_ROOT


# ---------------------------------------------------------------------------
# Path validation
# ---------------------------------------------------------------------------


def _validate_path(path: str) -> str:
    """Resolve *path* relative to FS_ROOT and reject escapes."""
    fs_root = _get_fs_root()

    if path == "/":
        return fs_root

    # Tilde expansion → home/userhome/
    path = re.sub(r"^~/?", "home/userhome/", path)
    path = re.sub(r"^~([^/]+)/?", r"home/\1/", path)

    # Strip leading slash so join works correctly
    if path.startswith("/"):
        path = path.lstrip("/")

    full = os.path.abspath(os.path.join(fs_root, path))

    if not (full == fs_root or full.startswith(fs_root + os.sep)):
        cli_error(f"Operation not allowed outside sandbox: {path}")

    return full


def _rel_path(full_path: str) -> str:
    """Convert an absolute path back to a sandbox-relative path with leading /."""
    fs_root = _get_fs_root()
    rel = os.path.relpath(full_path, fs_root)
    if rel == ".":
        return "/"
    return "/" + rel


# ---------------------------------------------------------------------------
# CLI group
# ---------------------------------------------------------------------------


@click.group(context_settings={"terminal_width": 10000, "max_content_width": 10000})
def cli():
    """CloudDrive - Gaia2 filesystem app CLI."""
    set_app(APP_NAME)
    set_log_class("SandboxLocalFileSystem")


# ---------------------------------------------------------------------------
# cat (READ)
# ---------------------------------------------------------------------------


@cli.command("cat")
@click.option("--path", "file_path", required=True, help="Path to the file to read.")
def cat_cmd(file_path):
    """Read the contents of a file and print to stdout."""
    real = _validate_path(file_path)

    if not os.path.exists(real):
        cli_error(f"File not found: {file_path}")
    if os.path.isdir(real):
        cli_error(f"Is a directory: {file_path}")

    try:
        with open(real, "r") as f:
            content = f.read()
    except UnicodeDecodeError:
        cli_error(f"Cannot read binary file as text: {file_path}")

    # Print raw content to stdout (like Unix cat)
    sys.stdout.write(content)

    # Log truncated preview to events
    preview = content[:200]
    if len(content) > 200:
        preview += "..."
    log_action("cat", {"path": file_path}, ret=preview)


# ---------------------------------------------------------------------------
# ls (READ)
# ---------------------------------------------------------------------------


@cli.command("ls")
@click.option(
    "--path", "dir_path", required=True, help="Path to the directory to list."
)
@click.option(
    "--detail", is_flag=True, default=False, help="Show detailed info for each entry."
)
def ls_cmd(dir_path, detail):
    """List the contents of a directory."""
    real = _validate_path(dir_path)

    if not os.path.exists(real):
        cli_error(f"Directory not found: {dir_path}")
    if not os.path.isdir(real):
        cli_error(f"Not a directory: {dir_path}")

    entries = sorted(os.listdir(real))

    if detail:
        result = []
        for name in entries:
            full = os.path.join(real, name)
            stat = os.stat(full)
            result.append(
                {
                    "name": _rel_path(full),
                    "size": stat.st_size,
                    "type": "directory" if os.path.isdir(full) else "file",
                }
            )
    else:
        result = [_rel_path(os.path.join(real, name)) for name in entries]

    log_action("ls", {"path": dir_path, "detail": detail}, ret=result)
    json_output(result)


# ---------------------------------------------------------------------------
# exists (READ)
# ---------------------------------------------------------------------------


@cli.command("exists")
@click.option("--path", "file_path", required=True, help="Path to check.")
def exists_cmd(file_path):
    """Check if a file or directory exists."""
    real = _validate_path(file_path)
    result = os.path.exists(real)

    log_action("exists", {"path": file_path}, ret=result)
    json_output(result)


# ---------------------------------------------------------------------------
# info (READ)
# ---------------------------------------------------------------------------


@cli.command("info")
@click.option("--path", "file_path", required=True, help="Path to get info about.")
def info_cmd(file_path):
    """Get metadata about a file or directory."""
    real = _validate_path(file_path)

    if not os.path.exists(real):
        cli_error(f"Path not found: {file_path}")

    stat = os.stat(real)
    result = {
        "name": _rel_path(real),
        "size": stat.st_size,
        "type": "directory" if os.path.isdir(real) else "file",
        "created": stat.st_ctime,
        "modified": stat.st_mtime,
    }

    log_action("info", {"path": file_path}, ret=result)
    json_output(result)


# ---------------------------------------------------------------------------
# tree (READ)
# ---------------------------------------------------------------------------


@cli.command("tree")
@click.option(
    "--path",
    "dir_path",
    default="/",
    help="Path to start the tree from (default: /).",
)
def tree_cmd(dir_path):
    """Generate a directory tree listing."""
    real = _validate_path(dir_path)

    if not os.path.exists(real):
        cli_error(f"Path not found: {dir_path}")
    if not os.path.isdir(real):
        cli_error(f"Not a directory: {dir_path}")

    def _build_tree(current_path, level):
        items = sorted(os.listdir(current_path))
        lines = []
        for i, name in enumerate(items):
            indent = "    " * level
            connector = "└── " if i == len(items) - 1 else "├── "
            lines.append(f"{indent}{connector}{name}")
            full = os.path.join(current_path, name)
            if os.path.isdir(full):
                lines.extend(_build_tree(full, level + 1))
        return lines

    root_name = os.path.basename(real) or "/"
    tree_str = root_name + "\n" + "\n".join(_build_tree(real, 0))
    if _build_tree(real, 0):
        tree_str += "\n"

    print(tree_str, end="")

    log_action("tree", {"path": dir_path}, ret=tree_str[:500])


# ---------------------------------------------------------------------------
# mkdir (WRITE)
# ---------------------------------------------------------------------------


@cli.command("mkdir")
@click.option(
    "--path", "dir_path", required=True, help="Path of the directory to create."
)
@click.option(
    "--create-parents",
    is_flag=True,
    default=False,
    help="Create parent directories if they do not exist.",
)
def mkdir_cmd(dir_path, create_parents):
    """Create a directory."""
    real = _validate_path(dir_path)

    if create_parents:
        os.makedirs(real, exist_ok=True)
    else:
        parent = os.path.dirname(real)
        if not os.path.isdir(parent):
            cli_error(f"Parent directory does not exist: {os.path.dirname(dir_path)}")
        os.mkdir(real)

    log_action(
        "mkdir", {"path": dir_path, "create_parents": create_parents}, write=True
    )
    json_output({"status": "success", "path": _rel_path(real)})


# ---------------------------------------------------------------------------
# makedirs (WRITE)
# ---------------------------------------------------------------------------


@cli.command("makedirs")
@click.option(
    "--path", "dir_path", required=True, help="Path of nested directories to create."
)
@click.option(
    "--exist-ok",
    is_flag=True,
    default=False,
    help="Do not raise an error if the directory already exists.",
)
def makedirs_cmd(dir_path, exist_ok):
    """Create nested directories (like mkdir -p)."""
    real = _validate_path(dir_path)

    try:
        os.makedirs(real, exist_ok=exist_ok)
    except FileExistsError:
        cli_error(f"Directory already exists: {dir_path}")

    log_action("makedirs", {"path": dir_path, "exist_ok": exist_ok}, write=True)
    json_output({"status": "success", "path": _rel_path(real)})


# ---------------------------------------------------------------------------
# mv (WRITE)
# ---------------------------------------------------------------------------


@cli.command("mv")
@click.option("--source", "--path1", required=True, help="Source path.")
@click.option("--destination", "--path2", required=True, help="Destination path.")
def mv_cmd(source, destination):
    """Move a file or directory."""
    real_src = _validate_path(source)
    real_dst = _validate_path(destination)

    if not os.path.exists(real_src):
        cli_error(f"Source not found: {source}")

    os.rename(real_src, real_dst)

    log_action("mv", {"path1": source, "path2": destination}, write=True)
    json_output(
        {
            "status": "success",
            "source": _rel_path(real_src),
            "destination": _rel_path(real_dst),
        }
    )


# ---------------------------------------------------------------------------
# rm (WRITE)
# ---------------------------------------------------------------------------


@cli.command("rm")
@click.option("--path", "file_path", required=True, help="Path to remove.")
@click.option(
    "--recursive",
    is_flag=True,
    default=False,
    help="Remove directories and their contents recursively.",
)
def rm_cmd(file_path, recursive):
    """Remove a file or directory."""
    import shutil

    real = _validate_path(file_path)

    if not os.path.exists(real):
        cli_error(f"Path not found: {file_path}")

    if os.path.isdir(real):
        if not recursive:
            cli_error(f"Is a directory (use --recursive): {file_path}")
        shutil.rmtree(real)
    else:
        os.remove(real)

    log_action("rm", {"path": file_path, "recursive": recursive}, write=True)
    json_output({"status": "success", "path": file_path})


# ---------------------------------------------------------------------------
# rmdir (WRITE)
# ---------------------------------------------------------------------------


@cli.command("rmdir")
@click.option(
    "--path", "dir_path", required=True, help="Path of the empty directory to remove."
)
def rmdir_cmd(dir_path):
    """Remove an empty directory."""
    real = _validate_path(dir_path)

    if not os.path.exists(real):
        cli_error(f"Path not found: {dir_path}")
    if not os.path.isdir(real):
        cli_error(f"Not a directory: {dir_path}")

    try:
        os.rmdir(real)
    except OSError:
        cli_error(f"Directory not empty: {dir_path}")

    log_action("rmdir", {"path": dir_path}, write=True)
    json_output({"status": "success", "path": _rel_path(real)})


# ---------------------------------------------------------------------------
# schema
# ---------------------------------------------------------------------------


@cli.command("schema")
def schema():
    """Output machine-readable JSON schema of all commands."""
    json_output(build_schema(cli))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main():
    cli()


if __name__ == "__main__":
    main()
