# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.


import time
from unittest.mock import Mock, patch

import pytest

# Mock the scenarios loading to avoid directory dependency
with patch(
    "are.simulation.scenarios.utils.load_utils.load_all_universes",
    return_value={},
):
    from are.simulation.gui.server.session_manager import SessionManager


def test_session_manager_initialization():
    manager = SessionManager(inactivity_limit=300, cleanup_interval=60)
    assert manager.inactivity_limit == 300
    assert manager.cleanup_interval == 60
    assert manager.sessions == {}
    assert manager.cleanup_timer is None
    assert manager.keep_cleanup is False


def test_add_and_get_are_simulation_instance():
    manager = SessionManager(300, 60)
    mock_are_simulation = Mock()

    # Test adding
    manager.add_are_simulation_instance("test_session", mock_are_simulation)
    assert manager.session_exists("test_session")

    # Test retrieving
    retrieved_are_simulation = manager.get_are_simulation_instance("test_session")
    assert retrieved_are_simulation == mock_are_simulation


def test_get_nonexistent_session():
    manager = SessionManager(300, 60)
    with pytest.raises(ValueError):
        manager.get_are_simulation_instance("nonexistent")


def test_session_exists():
    manager = SessionManager(300, 60)
    mock_are_simulation = Mock()

    assert not manager.session_exists("test_session")
    manager.add_are_simulation_instance("test_session", mock_are_simulation)
    assert manager.session_exists("test_session")


def test_get_total_sessions():
    manager = SessionManager(300, 60)
    assert manager.get_total_sessions() == 0

    mock_are_simulation = Mock()
    manager.add_are_simulation_instance("test1", mock_are_simulation)
    manager.add_are_simulation_instance("test2", mock_are_simulation)
    assert manager.get_total_sessions() == 2


def test_cleanup_inactive_are_simulation():
    manager = SessionManager(0, 60)
    mock_are_simulation1 = Mock()

    manager.add_are_simulation_instance("test_session", mock_are_simulation1)

    # Wait a small amount to ensure the session becomes inactive
    time.sleep(0.01)

    manager._cleanup_inactive_are_simulation()

    assert "test_session" not in manager.sessions
    mock_are_simulation1.stop.assert_called_once()


def test_stop_session_manager():
    manager = SessionManager(300, 60)
    mock_are_simulation1 = Mock()
    mock_are_simulation2 = Mock()

    manager.add_are_simulation_instance("session1", mock_are_simulation1)
    manager.add_are_simulation_instance("session2", mock_are_simulation2)

    manager.stop()

    mock_are_simulation1.stop.assert_called_once()
    mock_are_simulation2.stop.assert_called_once()
