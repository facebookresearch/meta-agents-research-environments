# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms of the Apache 2.0 license
# found in the LICENSE file in the root directory of this source tree.

"""Tests for: the runner extending the in-container no_proxy with the
agent/judge base_url hosts.

Motivation: OpenClaw runs its own in-container TLS MITM proxy. The default
no_proxy is just localhost — so any base_url pointing at a REMOTE host (e.g.
a judge served on another SLURM node) gets tunneled through the
MITM and fails. We need to add those hosts to no_proxy automatically."""

from __future__ import annotations

from gaia2_runner.cli import _build_container_env, _hosts_from_urls


class TestHostsFromUrls:
    def test_extracts_bare_hostname(self):
        assert _hosts_from_urls("http://gpu-node-01:8000/v1") == ["gpu-node-01"]

    def test_skips_localhost_variants(self):
        assert (
            _hosts_from_urls(
                "http://localhost:8000/v1",
                "http://127.0.0.1:8001/v1",
                "http://[::1]:8002/v1",
            )
            == []
        )

    def test_dedupes_when_two_urls_share_a_host(self):
        # agent + judge on same node, different ports.
        assert _hosts_from_urls(
            "http://node-a:8000/v1",
            "http://node-a:8001/v1",
        ) == ["node-a"]

    def test_returns_multiple_distinct_hosts_in_order(self):
        assert _hosts_from_urls(
            "http://node-a:8000/v1",
            "http://node-b:8001/v1",
        ) == ["node-a", "node-b"]

    def test_handles_none_and_empty(self):
        assert _hosts_from_urls(None, "", "  ") == []

    def test_ignores_unparseable_input(self):
        # not a URL at all, just a string — drop it silently.
        assert _hosts_from_urls("not a url") == []


class TestBuildContainerEnvNoProxy:
    def test_no_base_urls_means_no_no_proxy_key(self):
        env = _build_container_env(
            "localhost/gaia2-oc:latest",
            base_url=None,
            thinking="low",
        )
        assert "no_proxy" not in env
        assert "NO_PROXY" not in env

    def test_localhost_base_urls_means_no_no_proxy_key(self):
        env = _build_container_env(
            "localhost/gaia2-oc:latest",
            base_url="http://localhost:8000/v1",
            thinking="low",
            judge_base_url="http://127.0.0.1:8001/v1",
        )
        assert "no_proxy" not in env
        assert "NO_PROXY" not in env

    def test_remote_judge_appends_hostname_to_no_proxy(self):
        env = _build_container_env(
            "localhost/gaia2-oc:latest",
            base_url="http://localhost:8000/v1",
            thinking="low",
            judge_base_url="http://gpu-node-01:8001/v1",
        )
        hosts = set(env["no_proxy"].split(","))
        assert "gpu-node-01" in hosts
        # Always preserve localhost in the seed list.
        assert "localhost" in hosts
        assert env["NO_PROXY"] == env["no_proxy"]

    def test_remote_agent_and_remote_judge_both_added(self):
        env = _build_container_env(
            "localhost/gaia2-oc:latest",
            base_url="http://h100-001:8000/v1",
            thinking="low",
            judge_base_url="http://h100-002:8001/v1",
        )
        hosts = set(env["no_proxy"].split(","))
        assert {"h100-001", "h100-002", "localhost"} <= hosts
