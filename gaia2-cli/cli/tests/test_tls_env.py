# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
"""Tests for shared/gaia2-tls-env.sh and its use by the init entrypoints.

Launchers that forward the host environment can hand the container a CA
bundle variable naming a host-only file (a login node's
SSL_CERT_FILE=/var/facebook/x509_identities/tls.crt). litellm 1.105.0 then
raised FileNotFoundError while being imported, gaia2-eventd died creating the
judge, and the run idled out 600 s later. The entrypoints must hand only
usable bundles to the processes they start.

The scripts run with bash. "Root" is faked with ``id``/``su``/``chown`` shims:
the fake ``su`` answers ``test -r`` the way an unrelated user would (other
read bit only) and records any other command with its environment.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

GAIA2_CLI_DIR = Path(__file__).resolve().parents[2]
LIB = GAIA2_CLI_DIR / "shared" / "gaia2-tls-env.sh"
ENTRYPOINTS = sorted((GAIA2_CLI_DIR / "containers").glob("*/gaia2-init-entrypoint.sh"))
VARS = (
    "SSL_CERT_FILE",
    "REQUESTS_CA_BUNDLE",
    "CURL_CA_BUNDLE",
    "SSL_CERT_DIR",
    "NODE_EXTRA_CA_CERTS",
)
UNSET = "<unset>"
HOST_ONLY = "/var/facebook/x509_identities/tls.crt-gaia2-test-missing"

pytestmark = pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")

_SHIMS = {
    "id": """#!/bin/bash
case "$1" in
    -u) echo 0 ;;
    gaia2|agent) exit 0 ;;
    *) exec /usr/bin/id "$@" ;;
esac
""",
    # su -s SHELL USER -c CMD
    "su": """#!/bin/bash
user="$3"; cmd="$5"
echo "$user" >> "$FAKE_ROOT_LOG/su-users"
case "$cmd" in
    "test -r "*)
        exec /bin/bash -c 'test() { [ "$1" = -r ] && [ -n "$(find -L "$2" -maxdepth 0 -perm -0004 2>/dev/null)" ]; }; '"$cmd" ;;
esac
printf '%s' "$cmd" > "$FAKE_ROOT_LOG/su-command"
env > "$FAKE_ROOT_LOG/su-env"
""",
    "chown": "#!/bin/bash\nexit 0\n",
}


@pytest.fixture
def ca(tmp_path: Path) -> dict[str, Path]:
    """CA files: the image bundle, a readable host bundle, odd non-bundles."""
    certs = tmp_path / "certs"
    certs.mkdir()
    image = certs / "ca-certificates.crt"
    image.write_text("-----BEGIN CERTIFICATE-----\nimage\n")
    host = certs / "gaia2-host-ca-bundle.crt"
    host.write_text("-----BEGIN CERTIFICATE-----\nhost\n")
    private = certs / "private-bundle.crt"  # mounted from a umask 0007 checkout
    private.write_text("-----BEGIN CERTIFICATE-----\nprivate\n")
    empty = certs / "empty.crt"
    empty.touch()
    not_pem = certs / "passwd"
    not_pem.write_text("root:x:0:0:root:/root:/bin/bash\n")
    not_pem.chmod(0o644)
    for path, mode in ((image, 0o644), (host, 0o644), (private, 0o660)):
        path.chmod(mode)
    return {
        "image": image,
        "host": host,
        "private": private,
        "empty": empty,
        "not_pem": not_pem,
        "dir": certs,
    }


@pytest.fixture
def fake_root(tmp_path: Path) -> Path:
    """A PATH directory whose shims make the scripts believe they run as root."""
    shims = tmp_path / "shims"
    shims.mkdir()
    for name, body in _SHIMS.items():
        (shims / name).write_text(body)
        (shims / name).chmod(0o755)
    (tmp_path / "log").mkdir()
    return shims


def _bash(script: str, env: dict[str, str], shims: Path | None, tmp_path: Path):
    path = f"{shims}:/usr/bin:/bin" if shims else "/usr/bin:/bin"
    return subprocess.run(
        ["bash", "-c", script],
        env={"PATH": path, "FAKE_ROOT_LOG": str(tmp_path / "log"), **env},
        capture_output=True,
        text=True,
        check=False,
    )


def _sanitize(
    env: dict[str, str],
    ca: dict[str, Path],
    tmp_path: Path,
    shims: Path | None = None,
) -> tuple[dict[str, str], str]:
    """Run sanitize_tls_env as the entrypoints do; return the env and stderr."""
    script = (
        f"set -eo pipefail\n. {LIB}\nsanitize_tls_env\n"
        f'for v in {" ".join(VARS)}; do printf \'%s=%s\\n\' "$v" "${{!v-{UNSET}}}"; done\n'
    )
    env = {"GAIA2_IMAGE_CA_BUNDLE": str(ca["image"]), **env}
    result = _bash(script, env, shims, tmp_path)
    assert result.returncode == 0, result.stderr
    values = dict(line.split("=", 1) for line in result.stdout.splitlines())
    return values, result.stderr


class TestSanitizeTlsEnv:
    @pytest.mark.parametrize(
        "var", ["SSL_CERT_FILE", "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE"]
    )
    def test_host_only_bundle_is_replaced_by_the_image_bundle(self, var, ca, tmp_path):
        values, stderr = _sanitize({var: HOST_ONLY}, ca, tmp_path)

        assert values[var] == str(ca["image"])
        assert f"{var}={HOST_ONLY} is not a CA bundle the gaia2 and agent" in stderr

    def test_stage0_login_node_environment(self, ca, tmp_path):
        """The fair-sc login environment that killed gaia2-eventd in Stage 0."""
        values, _ = _sanitize(
            {
                "SSL_CERT_FILE": HOST_ONLY,
                "NODE_EXTRA_CA_CERTS": "/var/facebook/rootcanal/ca.pem-gaia2-test-missing",
            },
            ca,
            tmp_path,
        )

        assert values["SSL_CERT_FILE"] == str(ca["image"])
        assert values["NODE_EXTRA_CA_CERTS"] == UNSET

    def test_readable_bundles_are_kept(self, ca, tmp_path):
        env = {
            "SSL_CERT_FILE": str(ca["host"]),
            "REQUESTS_CA_BUNDLE": str(ca["host"]),
            "CURL_CA_BUNDLE": str(ca["image"]),
            "SSL_CERT_DIR": str(ca["dir"]),
            "NODE_EXTRA_CA_CERTS": str(ca["host"]),
        }

        values, stderr = _sanitize(env, ca, tmp_path)

        assert values == env
        assert stderr == ""

    def test_unset_variables_stay_unset(self, ca, tmp_path):
        values, stderr = _sanitize({}, ca, tmp_path)

        assert values == dict.fromkeys(VARS, UNSET)
        assert stderr == ""

    @pytest.mark.parametrize("kind", ["empty", "not_pem", "dir"])
    def test_non_bundles_are_replaced(self, kind, ca, tmp_path):
        values, _ = _sanitize({"SSL_CERT_FILE": str(ca[kind])}, ca, tmp_path)

        assert values["SSL_CERT_FILE"] == str(ca["image"])

    def test_unset_when_the_image_bundle_is_missing_too(self, ca, tmp_path):
        ca["image"].unlink()

        values, stderr = _sanitize({"SSL_CERT_FILE": HOST_ONLY}, ca, tmp_path)

        assert values["SSL_CERT_FILE"] == UNSET
        assert "unset it" in stderr

    def test_missing_cert_dir_is_unset(self, ca, tmp_path):
        values, _ = _sanitize({"SSL_CERT_DIR": HOST_ONLY}, ca, tmp_path)

        assert values["SSL_CERT_DIR"] == UNSET

    def test_as_root_asks_whether_gaia2_and_agent_can_read(
        self, ca, tmp_path, fake_root
    ):
        values, _ = _sanitize(
            {
                "SSL_CERT_FILE": str(ca["private"]),  # 0660: only root could read it
                "REQUESTS_CA_BUNDLE": str(ca["host"]),
            },
            ca,
            tmp_path,
            shims=fake_root,
        )

        assert values["SSL_CERT_FILE"] == str(ca["image"])
        assert values["REQUESTS_CA_BUNDLE"] == str(ca["host"])
        users = (tmp_path / "log" / "su-users").read_text().split()
        assert {"gaia2", "agent"} <= set(users)


# ---------------------------------------------------------------------------
# The entrypoints
# ---------------------------------------------------------------------------


def _ids(paths: list[Path]) -> list[str]:
    return [p.parent.name for p in paths]


def test_every_runtime_has_an_init_entrypoint():
    assert _ids(ENTRYPOINTS) == ["hermes", "mini-swe", "openclaw", "oracle"]


@pytest.mark.parametrize("entrypoint", ENTRYPOINTS, ids=_ids(ENTRYPOINTS))
class TestEntrypointWiring:
    def test_sanitizes_before_starting_anything(self, entrypoint: Path):
        text = entrypoint.read_text()
        source_at = text.index(". /opt/gaia2-tls-env.sh")
        if entrypoint.parent.name == "openclaw":
            # Everything runs from main(); sanitizing must come first there.
            body = text[text.index("\nmain() {") + 1 :]
            first = body.splitlines()[1].strip()
            assert first == "sanitize_tls_env"
        else:
            call_at = text.index("\nsanitize_tls_env\n")
            assert source_at < call_at
            for start in ("su -s", "gaia2-init ", "python3"):
                assert call_at < text.index(start), start

    def test_eventd_uses_the_bundled_litellm_price_map(self, entrypoint: Path):
        text = entrypoint.read_text()
        launch = text[text.index("su -s /usr/bin/bash gaia2 -c") :]
        launch = launch[: launch.index("/usr/local/bin/gaia2-eventd")]
        assert "export LITELLM_LOCAL_MODEL_COST_MAP=True" in launch


def test_openclaw_eventd_starts_with_a_sanitized_environment(ca, tmp_path, fake_root):
    """Run the OpenClaw entrypoint's own functions up to the eventd launch."""
    entrypoint = GAIA2_CLI_DIR / "containers" / "openclaw" / "gaia2-init-entrypoint.sh"
    text = entrypoint.read_text().replace(". /opt/gaia2-tls-env.sh", f". {LIB}")
    assert text.rstrip().endswith('main "$@"')
    functions = tmp_path / "entrypoint-functions.sh"
    functions.write_text(text.rstrip()[: -len('main "$@"')])
    scenario = tmp_path / "scenario.json"
    scenario.write_text("{}")

    script = f"""
. {functions}
export PATH={fake_root}:/usr/bin:/bin  # the entrypoint resets PATH
EVENTD_LOG={tmp_path}/eventd.log
FAKETIME_FILE={tmp_path}/faketime.rc
sanitize_tls_env
start_eventd
"""
    result = _bash(
        script,
        {
            "GAIA2_IMAGE_CA_BUNDLE": str(ca["image"]),
            "GAIA2_SCENARIO": str(scenario),
            "SSL_CERT_FILE": HOST_ONLY,
            "REQUESTS_CA_BUNDLE": HOST_ONLY,
        },
        fake_root,
        tmp_path,
    )

    assert result.returncode == 0, result.stderr
    command = (tmp_path / "log" / "su-command").read_text()
    assert "/usr/local/bin/gaia2-eventd" in command
    assert re.search(r"^\s*export LITELLM_LOCAL_MODEL_COST_MAP=True$", command, re.M)
    child_env = dict(
        line.split("=", 1)
        for line in (tmp_path / "log" / "su-env").read_text().splitlines()
        if "=" in line
    )
    assert child_env["SSL_CERT_FILE"] == str(ca["image"])
    assert child_env["REQUESTS_CA_BUNDLE"] == str(ca["image"])
    assert os.path.exists(child_env["SSL_CERT_FILE"])
