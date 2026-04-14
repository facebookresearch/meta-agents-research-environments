# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
"""Unit tests for gaia2_cli/apps/files.py."""

import json

import pytest
from conftest import assert_event, make_cli_runner, parse_output, read_events
from gaia2_cli.apps.files import cli

APP = "SandboxLocalFileSystem"


@pytest.fixture
def fs_env(tmp_path, monkeypatch):
    """Set up a filesystem root and GAIA2_STATE_DIR."""
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    fs_root = state_dir / "filesystem"
    fs_root.mkdir()
    monkeypatch.setenv("GAIA2_STATE_DIR", str(state_dir))
    (state_dir / "events.jsonl").touch()

    # Reset the cached FS_ROOT so each test picks up the new env
    import gaia2_cli.apps.files as da

    da._FS_ROOT = None

    return fs_root, state_dir, make_cli_runner()


def _populate(fs_root):
    """Create a small directory tree for testing."""
    docs = fs_root / "Documents"
    docs.mkdir()
    (docs / "hello.txt").write_text("hello world")
    (docs / "data.csv").write_text("a,b,c\n1,2,3\n")
    pics = fs_root / "Pictures"
    pics.mkdir()
    return docs, pics


# ---------------------------------------------------------------------------
# cat
# ---------------------------------------------------------------------------


def test_cat_happy_path(fs_env):
    fs_root, state_dir, runner = fs_env
    _populate(fs_root)

    result = runner.invoke(cli, ["cat", "--path", "/Documents/hello.txt"])
    assert result.exit_code == 0, result.output
    assert result.output == "hello world"


def test_cat_file_not_found(fs_env):
    fs_root, state_dir, runner = fs_env

    result = runner.invoke(cli, ["cat", "--path", "/no_such_file.txt"])
    assert result.exit_code == 1
    assert "File not found" in (result.output + (result.stderr or ""))


def test_cat_is_directory(fs_env):
    fs_root, state_dir, runner = fs_env
    _populate(fs_root)

    result = runner.invoke(cli, ["cat", "--path", "/Documents"])
    assert result.exit_code == 1
    assert "Is a directory" in (result.output + (result.stderr or ""))


def test_cat_event_log(fs_env):
    fs_root, state_dir, runner = fs_env
    _populate(fs_root)

    runner.invoke(cli, ["cat", "--path", "/Documents/hello.txt"])
    events = read_events(state_dir)
    assert len(events) == 1
    assert_event(events[0], APP, "cat", write=False)
    assert events[0]["args"]["path"] == "/Documents/hello.txt"


# ---------------------------------------------------------------------------
# ls
# ---------------------------------------------------------------------------


def test_ls_simple(fs_env):
    fs_root, state_dir, runner = fs_env
    _populate(fs_root)

    result = runner.invoke(cli, ["ls", "--path", "/"])
    assert result.exit_code == 0, result.output
    out = parse_output(result)
    assert "/Documents" in out
    assert "/Pictures" in out


def test_ls_detail(fs_env):
    fs_root, state_dir, runner = fs_env
    _populate(fs_root)

    result = runner.invoke(cli, ["ls", "--path", "/Documents", "--detail"])
    assert result.exit_code == 0, result.output
    out = parse_output(result)
    assert len(out) == 2
    names = {e["name"] for e in out}
    assert "/Documents/data.csv" in names
    assert "/Documents/hello.txt" in names
    for entry in out:
        assert "size" in entry
        assert "type" in entry


def test_ls_not_found(fs_env):
    fs_root, state_dir, runner = fs_env

    result = runner.invoke(cli, ["ls", "--path", "/nonexistent"])
    assert result.exit_code == 1


# ---------------------------------------------------------------------------
# exists
# ---------------------------------------------------------------------------


def test_exists_true(fs_env):
    fs_root, state_dir, runner = fs_env
    _populate(fs_root)

    result = runner.invoke(cli, ["exists", "--path", "/Documents/hello.txt"])
    assert result.exit_code == 0
    assert parse_output(result) is True


def test_exists_false(fs_env):
    fs_root, state_dir, runner = fs_env

    result = runner.invoke(cli, ["exists", "--path", "/no_such_file"])
    assert result.exit_code == 0
    assert parse_output(result) is False


# ---------------------------------------------------------------------------
# info
# ---------------------------------------------------------------------------


def test_info_file(fs_env):
    fs_root, state_dir, runner = fs_env
    _populate(fs_root)

    result = runner.invoke(cli, ["info", "--path", "/Documents/hello.txt"])
    assert result.exit_code == 0
    out = parse_output(result)
    assert out["type"] == "file"
    assert out["size"] == len("hello world")
    assert out["name"] == "/Documents/hello.txt"


def test_info_directory(fs_env):
    fs_root, state_dir, runner = fs_env
    _populate(fs_root)

    result = runner.invoke(cli, ["info", "--path", "/Documents"])
    assert result.exit_code == 0
    out = parse_output(result)
    assert out["type"] == "directory"


def test_info_not_found(fs_env):
    fs_root, state_dir, runner = fs_env

    result = runner.invoke(cli, ["info", "--path", "/nope"])
    assert result.exit_code == 1


# ---------------------------------------------------------------------------
# tree
# ---------------------------------------------------------------------------


def test_tree(fs_env):
    fs_root, state_dir, runner = fs_env
    _populate(fs_root)

    result = runner.invoke(cli, ["tree", "--path", "/"])
    assert result.exit_code == 0
    assert "Documents" in result.output
    assert "hello.txt" in result.output
    assert "Pictures" in result.output


# ---------------------------------------------------------------------------
# mkdir
# ---------------------------------------------------------------------------


def test_mkdir(fs_env):
    fs_root, state_dir, runner = fs_env

    result = runner.invoke(cli, ["mkdir", "--path", "/NewDir"])
    assert result.exit_code == 0, result.output
    assert (fs_root / "NewDir").is_dir()

    events = read_events(state_dir)
    assert len(events) == 1
    assert_event(events[0], APP, "mkdir", write=True)


def test_mkdir_no_parent(fs_env):
    fs_root, state_dir, runner = fs_env

    result = runner.invoke(cli, ["mkdir", "--path", "/a/b/c"])
    assert result.exit_code == 1
    assert "Parent directory does not exist" in (result.output + (result.stderr or ""))


def test_mkdir_create_parents(fs_env):
    fs_root, state_dir, runner = fs_env

    result = runner.invoke(cli, ["mkdir", "--path", "/a/b/c", "--create-parents"])
    assert result.exit_code == 0
    assert (fs_root / "a" / "b" / "c").is_dir()


# ---------------------------------------------------------------------------
# makedirs
# ---------------------------------------------------------------------------


def test_makedirs(fs_env):
    fs_root, state_dir, runner = fs_env

    result = runner.invoke(cli, ["makedirs", "--path", "/x/y/z"])
    assert result.exit_code == 0
    assert (fs_root / "x" / "y" / "z").is_dir()


def test_makedirs_exist_ok(fs_env):
    fs_root, state_dir, runner = fs_env
    (fs_root / "existing").mkdir()

    result = runner.invoke(cli, ["makedirs", "--path", "/existing", "--exist-ok"])
    assert result.exit_code == 0

    result = runner.invoke(cli, ["makedirs", "--path", "/existing"])
    assert result.exit_code == 1


# ---------------------------------------------------------------------------
# mv
# ---------------------------------------------------------------------------


def test_mv(fs_env):
    fs_root, state_dir, runner = fs_env
    _populate(fs_root)

    result = runner.invoke(
        cli,
        [
            "mv",
            "--source",
            "/Documents/hello.txt",
            "--destination",
            "/Documents/moved.txt",
        ],
    )
    assert result.exit_code == 0, result.output
    assert not (fs_root / "Documents" / "hello.txt").exists()
    assert (fs_root / "Documents" / "moved.txt").read_text() == "hello world"


def test_mv_path_aliases(fs_env):
    fs_root, state_dir, runner = fs_env
    _populate(fs_root)

    result = runner.invoke(
        cli,
        [
            "mv",
            "--path1",
            "/Documents/hello.txt",
            "--path2",
            "/Documents/moved.txt",
        ],
    )
    assert result.exit_code == 0, result.output
    assert not (fs_root / "Documents" / "hello.txt").exists()
    assert (fs_root / "Documents" / "moved.txt").read_text() == "hello world"


def test_mv_not_found(fs_env):
    fs_root, state_dir, runner = fs_env

    result = runner.invoke(
        cli, ["mv", "--source", "/nope.txt", "--destination", "/dest.txt"]
    )
    assert result.exit_code == 1


# ---------------------------------------------------------------------------
# rm
# ---------------------------------------------------------------------------


def test_rm_file(fs_env):
    fs_root, state_dir, runner = fs_env
    _populate(fs_root)

    result = runner.invoke(cli, ["rm", "--path", "/Documents/hello.txt"])
    assert result.exit_code == 0
    assert not (fs_root / "Documents" / "hello.txt").exists()


def test_rm_directory_without_recursive(fs_env):
    fs_root, state_dir, runner = fs_env
    _populate(fs_root)

    result = runner.invoke(cli, ["rm", "--path", "/Documents"])
    assert result.exit_code == 1
    assert "use --recursive" in (result.output + (result.stderr or ""))


def test_rm_directory_recursive(fs_env):
    fs_root, state_dir, runner = fs_env
    _populate(fs_root)

    result = runner.invoke(cli, ["rm", "--path", "/Documents", "--recursive"])
    assert result.exit_code == 0
    assert not (fs_root / "Documents").exists()


# ---------------------------------------------------------------------------
# rmdir
# ---------------------------------------------------------------------------


def test_rmdir(fs_env):
    fs_root, state_dir, runner = fs_env
    (fs_root / "EmptyDir").mkdir()

    result = runner.invoke(cli, ["rmdir", "--path", "/EmptyDir"])
    assert result.exit_code == 0, result.output
    assert not (fs_root / "EmptyDir").exists()

    events = read_events(state_dir)
    assert len(events) == 1
    assert_event(events[0], APP, "rmdir", write=True)


def test_rmdir_not_empty(fs_env):
    fs_root, state_dir, runner = fs_env
    _populate(fs_root)

    result = runner.invoke(cli, ["rmdir", "--path", "/Documents"])
    assert result.exit_code == 1
    assert "not empty" in (result.output + (result.stderr or "")).lower()


# ---------------------------------------------------------------------------
# Path traversal / validation
# ---------------------------------------------------------------------------


def test_path_traversal_rejected(fs_env):
    fs_root, state_dir, runner = fs_env

    result = runner.invoke(cli, ["cat", "--path", "/../../etc/passwd"])
    assert result.exit_code == 1
    assert "not allowed outside sandbox" in (result.output + (result.stderr or ""))


def test_path_traversal_dotdot_in_middle(fs_env):
    fs_root, state_dir, runner = fs_env
    _populate(fs_root)

    result = runner.invoke(cli, ["cat", "--path", "/Documents/../../../etc/passwd"])
    assert result.exit_code == 1


def test_tilde_resolves_to_home(fs_env):
    fs_root, state_dir, runner = fs_env
    home = fs_root / "home" / "userhome"
    home.mkdir(parents=True)
    (home / "notes.txt").write_text("my notes")

    result = runner.invoke(cli, ["cat", "--path", "~/notes.txt"])
    assert result.exit_code == 0
    assert "my notes" in result.output


# ---------------------------------------------------------------------------
# schema
# ---------------------------------------------------------------------------


def test_schema(fs_env):
    fs_root, state_dir, runner = fs_env

    result = runner.invoke(cli, ["schema"])
    assert result.exit_code == 0
    out = parse_output(result)
    commands = {entry["command"] for entry in out}
    assert "cat" in commands
    assert "ls" in commands
    assert "tree" in commands
    assert "rmdir" in commands
    assert "schema" not in commands


# ---------------------------------------------------------------------------
# gaia2-init filesystem creation
# ---------------------------------------------------------------------------


def test_init_filesystem_from_scenario(tmp_path, monkeypatch):
    """Test that gaia2-init creates the filesystem directory from scenario state."""
    from gaia2_cli.init_cmd import main as init_main

    # Create a minimal scenario JSON with a Files app
    scenario = {
        "apps": [
            {
                "name": "SandboxLocalFileSystem",
                "app_state": {
                    "files": {
                        "name": "tmppom6e92_",
                        "type": "directory",
                        "children": [
                            {
                                "name": "Documents",
                                "type": "directory",
                                "children": [
                                    {"name": "report.txt", "type": "file"},
                                    {
                                        "name": "backed.txt",
                                        "type": "file",
                                        "real_path": "demo_filesystem/Documents/backed.txt",
                                    },
                                ],
                            },
                            {"name": "Pictures", "type": "directory", "children": []},
                        ],
                    }
                },
            }
        ]
    }

    scenario_file = tmp_path / "scenario.json"
    scenario_file.write_text(json.dumps(scenario))

    # Create a backing store with one file
    backing = tmp_path / "backing"
    (backing / "demo_filesystem" / "Documents").mkdir(parents=True)
    (backing / "demo_filesystem" / "Documents" / "backed.txt").write_text(
        "backed content"
    )

    state_dir = tmp_path / "state"

    runner = make_cli_runner()
    result = runner.invoke(
        init_main,
        [
            "--scenario",
            str(scenario_file),
            "--state-dir",
            str(state_dir),
            "--fs-backing-dir",
            str(backing),
        ],
    )
    assert result.exit_code == 0, result.output

    fs_root = state_dir / "filesystem"
    assert fs_root.is_dir()
    assert (fs_root / "Documents").is_dir()
    assert (fs_root / "Pictures").is_dir()
    assert (fs_root / "Documents" / "report.txt").exists()
    # report.txt has no real_path → empty placeholder
    assert (fs_root / "Documents" / "report.txt").stat().st_size == 0
    # backed.txt should have real content
    assert (fs_root / "Documents" / "backed.txt").read_text() == "backed content"


def test_init_filesystem_no_backing_dir(tmp_path, monkeypatch):
    """Without a backing dir, all files become empty placeholders."""
    from gaia2_cli.init_cmd import main as init_main

    scenario = {
        "apps": [
            {
                "name": "VirtualFileSystem",
                "app_state": {
                    "files": {
                        "name": "",
                        "type": "directory",
                        "children": [
                            {
                                "name": "data.csv",
                                "type": "file",
                                "real_path": "demo_filesystem/data.csv",
                            }
                        ],
                    }
                },
            }
        ]
    }

    scenario_file = tmp_path / "scenario.json"
    scenario_file.write_text(json.dumps(scenario))
    state_dir = tmp_path / "state"

    # Ensure env var is not set
    monkeypatch.delenv("GAIA2_FS_BACKING_DIR", raising=False)

    runner = make_cli_runner()
    result = runner.invoke(
        init_main,
        ["--scenario", str(scenario_file), "--state-dir", str(state_dir)],
    )
    assert result.exit_code == 0, result.output

    fs_root = state_dir / "filesystem"
    assert (fs_root / "data.csv").exists()
    assert (fs_root / "data.csv").stat().st_size == 0
