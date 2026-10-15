# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
"""Static checks of the image Dockerfiles (gaia2-cli base and every runtime).

Ways a rebuild of the same commit has broken in practice, each caught here
without building anything:

- COPY keeps the modes of the build context. A checkout made under umask 0007
  has 0660 files, which the in-image gaia2 and agent users cannot read (the
  OpenClaw build then failed running /etc/profile.d/openclaw-setup.sh as
  agent). Every file copied from the context must therefore set its mode.
- ``litellm>=1.0`` floats: a fresh build picked up litellm 1.105.0, whose
  import builds an httpx SSL context. The base image pins every package.
"""

from __future__ import annotations

import re
import shlex
import tomllib
from pathlib import Path

import pytest
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

GAIA2_CLI_DIR = Path(__file__).resolve().parents[2]  # the image build context
BASE_DOCKERFILE = GAIA2_CLI_DIR / "cli" / "Dockerfile"
SCAFFOLD_DOCKERFILES = sorted((GAIA2_CLI_DIR / "containers").glob("*/Dockerfile"))
DOCKERFILES = [BASE_DOCKERFILE, *SCAFFOLD_DOCKERFILES]
CONSTRAINTS = GAIA2_CLI_DIR / "cli" / "image-constraints.txt"
CONSTRAINTS_IN_IMAGE = "/tmp/image-constraints.txt"


def _ids(paths: list[Path]) -> list[str]:
    return [str(p.relative_to(GAIA2_CLI_DIR)) for p in paths]


def _instructions(dockerfile: Path) -> list[tuple[str, str]]:
    """Return ``(KEYWORD, arguments)`` per instruction, continuations joined."""
    instructions: list[tuple[str, str]] = []
    pending = ""
    for raw in dockerfile.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.endswith("\\"):
            pending += line[:-1] + " "
            continue
        keyword, _, args = (pending + line).partition(" ")
        instructions.append((keyword.upper(), args.strip()))
        pending = ""
    assert not pending, f"{dockerfile}: dangling line continuation"
    return instructions


def _copies(dockerfile: Path) -> list[tuple[dict[str, str], list[str], str]]:
    """Return ``(flags, sources, destination)`` for each COPY/ADD."""
    copies = []
    for keyword, args in _instructions(dockerfile):
        if keyword not in ("COPY", "ADD"):
            continue
        assert not args.startswith("["), f"{dockerfile}: JSON-form {keyword}"
        tokens = shlex.split(args)
        flags: dict[str, str] = {}
        while tokens and tokens[0].startswith("--"):
            name, _, value = tokens.pop(0)[2:].partition("=")
            flags[name] = value
        *sources, destination = tokens
        copies.append((flags, sources, destination))
    return copies


def _context_copies(dockerfile: Path):
    """COPY/ADD instructions that read the build context (not another stage)."""
    return [c for c in _copies(dockerfile) if "from" not in c[0]]


def _mode(flags: dict[str, str]) -> int:
    value = flags.get("chmod", "")
    # buildah 1.23 (podman 3.4) rejects symbolic modes such as a+rX.
    assert re.fullmatch(r"0?[0-7]{3}", value), (
        f"--chmod={value!r}: use an explicit numeric mode such as 0644"
    )
    return int(value, 8)


def test_all_images_are_checked() -> None:
    assert _ids(SCAFFOLD_DOCKERFILES) == [
        "containers/hermes/Dockerfile",
        "containers/mini-swe/Dockerfile",
        "containers/openclaw/Dockerfile",
        "containers/oracle/Dockerfile",
    ]


@pytest.mark.parametrize("dockerfile", DOCKERFILES, ids=_ids(DOCKERFILES))
class TestCopiedFileModes:
    def test_every_context_copy_sets_its_mode(self, dockerfile: Path) -> None:
        missing = [
            " ".join(sources)
            for flags, sources, _ in _context_copies(dockerfile)
            if "chmod" not in flags
        ]
        assert not missing, (
            "COPY without --chmod keeps the checkout's modes (0660 under "
            f"umask 0007): {missing}"
        )

    def test_copied_files_are_readable_by_every_user(self, dockerfile: Path) -> None:
        for flags, sources, destination in _context_copies(dockerfile):
            mode = _mode(flags)
            assert mode & 0o444 == 0o444, f"{sources} -> {destination}: {oct(mode)}"
            assert mode & 0o022 == 0, f"{sources} must not be group/world-writable"
            if any((GAIA2_CLI_DIR / s).is_dir() for s in sources):
                # --chmod applies to the copied directories too.
                assert mode & 0o111 == 0o111, f"{sources}: directories need +x"

    def test_copied_sources_exist(self, dockerfile: Path) -> None:
        for _, sources, _ in _context_copies(dockerfile):
            for source in sources:
                assert (GAIA2_CLI_DIR / source).exists(), source


@pytest.mark.parametrize(
    "dockerfile", SCAFFOLD_DOCKERFILES, ids=_ids(SCAFFOLD_DOCKERFILES)
)
class TestRuntimeImages:
    def test_init_entrypoint_is_executable(self, dockerfile: Path) -> None:
        # The runner and FEH's apptainer launcher exec it directly.
        modes = [
            _mode(flags)
            for flags, _, destination in _context_copies(dockerfile)
            if destination == "/opt/gaia2-init-entrypoint.sh"
        ]
        assert len(modes) == 1
        assert modes[0] & 0o111 == 0o111

    def test_tls_env_helper_is_shipped(self, dockerfile: Path) -> None:
        entrypoint = dockerfile.parent / "gaia2-init-entrypoint.sh"
        assert ". /opt/gaia2-tls-env.sh" in entrypoint.read_text()
        assert (["shared/gaia2-tls-env.sh"], "/opt/gaia2-tls-env.sh") in [
            (sources, destination)
            for _, sources, destination in _context_copies(dockerfile)
        ]

    def test_privilege_boundaries_are_unchanged(self, dockerfile: Path) -> None:
        runs = " ; ".join(args for kw, args in _instructions(dockerfile) if kw == "RUN")
        assert "chmod u+s /usr/local/bin/gaia2-exec" in runs
        assert "chmod 700 /var/gaia2" in runs
        assert "chmod 700 /opt/gaia2_filesystem" in runs
        for _, _, destination in _copies(dockerfile):
            assert not destination.startswith("/var/gaia2"), destination


# ---------------------------------------------------------------------------
# Python package pins of the base image
# ---------------------------------------------------------------------------


def _pins() -> dict[str, str]:
    pins: dict[str, str] = {}
    for line in CONSTRAINTS.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        match = re.fullmatch(r"([A-Za-z0-9][A-Za-z0-9._-]*)==([^\s;=]+)", line)
        assert match, f"not an exact pin: {line!r}"
        name = canonicalize_name(match.group(1))
        assert name not in pins, f"{name} pinned twice"
        pins[name] = match.group(2)
    return pins


def _pip_installs() -> list[str]:
    commands = []
    for keyword, args in _instructions(BASE_DOCKERFILE):
        if keyword == "RUN":
            commands += [c.strip() for c in args.split("&&") if "pip install" in c]
    return commands


def test_constraints_pin_exact_versions() -> None:
    pins = _pins()
    assert pins["litellm"] == "1.105.0"
    # The rest of the judge stack floats just the same without pins.
    for name in ("httpx", "openai", "pydantic", "click"):
        assert name in pins


def test_every_base_image_pip_install_uses_the_constraints() -> None:
    installs = _pip_installs()
    assert len(installs) == 3  # gaia2-core, gaia2-cli[judge], huggingface_hub
    for command in installs:
        assert f"-c {CONSTRAINTS_IN_IMAGE}" in command, command


def test_constraints_are_copied_before_the_first_install() -> None:
    instructions = _instructions(BASE_DOCKERFILE)
    copy_at = next(
        i
        for i, (kw, args) in enumerate(instructions)
        if kw == "COPY"
        and args.endswith(f"cli/image-constraints.txt {CONSTRAINTS_IN_IMAGE}")
    )
    install_at = next(
        i
        for i, (kw, args) in enumerate(instructions)
        if kw == "RUN" and "pip install" in args
    )
    assert copy_at < install_at


def test_pins_satisfy_the_declared_requirements() -> None:
    pins = _pins()
    cli = tomllib.loads((GAIA2_CLI_DIR / "cli" / "pyproject.toml").read_text())
    core = tomllib.loads((GAIA2_CLI_DIR / "core" / "pyproject.toml").read_text())
    requirements = [
        *core["project"]["dependencies"],
        *cli["project"]["dependencies"],
        *cli["project"]["optional-dependencies"]["judge"],
    ]
    for spec in requirements:
        requirement = Requirement(spec)
        name = canonicalize_name(requirement.name)
        if name in ("gaia2-core", "gaia2-cli"):
            continue  # installed from the build context
        assert name in pins, f"{name} is required but not pinned"
        assert requirement.specifier.contains(pins[name], prereleases=True), (
            f"{name}=={pins[name]} does not satisfy {spec!r}"
        )
