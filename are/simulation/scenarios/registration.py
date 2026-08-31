# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.


"""
Registration module for built-in Meta Agents Research Environments scenarios.

This module is responsible for registering all built-in scenarios with the
Meta Agents Research Environments scenario registry. It is loaded via the entry point system when Meta Agents Research Environments starts up.
"""

import importlib
import importlib.util
import logging
import os
import pkgutil
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from are.simulation.scenarios.utils.registry import ScenarioRegistry

logger = logging.getLogger(__name__)


def register_scenarios(registry):
    """
    Register all scenarios in this package with the provided registry.

    Args:
        registry: The ScenarioRegistry instance to register with
    """
    # Simply import the modules containing the scenarios
    # The decorators will handle the registration
    import scenario_flight_customer_service.scenario


def register_builtin_scenarios(registry: "ScenarioRegistry"):
    """
    Register all built-in scenarios with the provided registry.

    This function is called by the Meta Agents Research Environments framework when it discovers the
    built-in scenarios entry point. It imports all scenario modules in the
    are.simulation.scenarios package, which triggers the @register_scenario decorators.

    Args:
        registry: The ScenarioRegistry instance to register with
    """
    logger.info("Registering built-in scenarios")

    # Get the path to the scenarios directory
    scenarios_dir = Path(__file__).parent

    # Track imported modules to avoid duplicates
    imported_modules = set()
    imported_count = 0
    scenarios_to_skip = []

    # Method 1: Use pkgutil.iter_modules to find modules in proper Python packages
    logger.debug("Discovering scenarios using pkgutil.iter_modules")
    _discover_with_pkgutil(scenarios_dir, imported_modules, scenarios_to_skip)

    # Method 2: Use os.walk as a fallback to find Python files in directories
    # that might not be proper packages
    logger.debug("Discovering scenarios using os.walk (fallback method)")
    _discover_with_os_walk(scenarios_dir, imported_modules, scenarios_to_skip)

    imported_count = len(imported_modules)
    logger.info(f"Registered built-in scenarios from {imported_count} modules")


def _discover_with_pkgutil(scenarios_dir, imported_modules, modules_to_skip):
    """
    Discover scenario modules using pkgutil.iter_modules.

    This method finds modules in directories that have __init__.py files
    (proper Python packages).

    Args:
        scenarios_dir: Path to the scenarios directory
        imported_modules: Set to track imported modules
        modules_to_skip: List of modules to skip
    """
    skip = ["utils", "tests"] + (modules_to_skip or [])
    # Walk through all modules in the scenarios package
    for _, name, is_pkg in pkgutil.iter_modules([str(scenarios_dir)]):
        # Skip utility modules and test modules
        if name in skip or name.startswith("_"):
            continue

        # If it's a package, we need to look inside it
        if is_pkg:
            pkg_name = f"are.simulation.scenarios.{name}"
            try:
                # Import the package
                pkg = importlib.import_module(pkg_name)

                # Look for scenario modules in the package
                if pkg.__file__ is None:
                    logger.warning(
                        f"Package {pkg_name} has no __file__ attribute, skipping"
                    )
                    continue
                pkg_path = Path(pkg.__file__).parent
                for _, subname, _ in pkgutil.iter_modules([str(pkg_path)]):
                    # Skip utility modules and test modules
                    if subname in ["utils", "tests"] or subname.startswith("_"):
                        continue

                    # Import the module if it might contain scenarios
                    if "scenario" in subname:
                        module_name = f"{pkg_name}.{subname}"
                        try:
                            if module_name not in imported_modules:
                                importlib.import_module(module_name)
                                imported_modules.add(module_name)
                                logger.debug(f"Imported scenario module: {module_name}")
                        except Exception as e:
                            logger.warning(
                                f"Failed to import scenario module {module_name}: {e}",
                                exc_info=True,
                            )
            except Exception as e:
                logger.warning(
                    f"Failed to import scenario package {pkg_name}: {e}", exc_info=True
                )
        else:
            # Import the module if it might contain scenarios
            if "scenario" in name:
                module_name = f"are.simulation.scenarios.{name}"
                try:
                    if module_name not in imported_modules:
                        importlib.import_module(module_name)
                        imported_modules.add(module_name)
                        logger.debug(f"Imported scenario module: {module_name}")
                except Exception as e:
                    logger.warning(
                        f"Failed to import scenario module {module_name}: {e}",
                        exc_info=True,
                    )


def _discover_with_os_walk(scenarios_dir, imported_modules, modules_to_skip):
    """
    Discover scenario modules using os.walk.

    This method finds Python files in directories that might not have __init__.py files
    (not proper Python packages).

    Args:
        scenarios_dir: Path to the scenarios directory
        imported_modules: Set to track imported modules
        modules_to_skip: List of modules to skip
    """
    skip = ["utils", "tests", "__pycache__"] + (modules_to_skip or [])
    # Walk through all directories and files
    for root, dirs, files in os.walk(scenarios_dir):
        # Skip utility directories and hidden directories
        dirs[:] = [d for d in dirs if d not in skip and not d.startswith(".")]

        # Get the relative path from the scenarios directory
        rel_path = os.path.relpath(root, scenarios_dir)

        # Skip the root directory
        if rel_path == ".":
            continue

        # Convert the relative path to a module path
        if rel_path == "..":
            continue

        module_path = rel_path.replace(os.sep, ".")

        # Look for Python files that might contain scenarios
        for file in files:
            # Skip non-Python files and __init__.py
            if not file.endswith(".py") or file == "__init__.py":
                continue

            # Skip files that don't match our naming convention
            if "scenario" not in file.lower() and not file.endswith("_scenario.py"):
                continue

            # Get the module name
            module_name = f"are.simulation.scenarios.{module_path}.{file[:-3]}"

            # Skip if we've already imported this module
            if module_name in imported_modules:
                continue

            # Import the module
            try:
                # Use importlib.util to import the module from the file path
                file_path = os.path.join(root, file)
                module_spec = importlib.util.spec_from_file_location(
                    module_name, file_path
                )

                if module_spec and module_spec.loader:
                    module = importlib.util.module_from_spec(module_spec)
                    sys.modules[module_name] = module
                    module_spec.loader.exec_module(module)
                    imported_modules.add(module_name)
                    logger.debug(f"Imported scenario module from file: {file_path}")
            except Exception as e:
                logger.warning(
                    f"Failed to import scenario file {file}: {e}", exc_info=True
                )
