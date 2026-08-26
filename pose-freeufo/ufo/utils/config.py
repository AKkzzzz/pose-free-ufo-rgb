# Copyright (C) 2026 Xiaomi Corporation.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied. See the License for the specific language governing
# permissions and limitations under the License.

"""
Configuration utilities for loading and merging JSON configs with command-line arguments.
"""
import json
import argparse
from pathlib import Path
from typing import Any, Dict, Optional


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from JSON file.

    Args:
        config_path: Path to JSON configuration file

    Returns:
        Dictionary containing configuration parameters
    """
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config


def save_config(config: Dict[str, Any], output_path: str):
    """Save configuration to JSON file.

    Args:
        config: Configuration dictionary
        output_path: Path to save JSON file
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(config, f, indent=2)


def merge_config_and_args(parser: argparse.ArgumentParser,
                          config_path: Optional[str] = None,
                          cli_args: Optional[list] = None) -> argparse.Namespace:
    """Merge JSON config with command-line arguments.

    Priority (highest to lowest):
    1. Command-line arguments explicitly provided
    2. JSON config file values
    3. Argument parser defaults

    Args:
        parser: ArgumentParser with all arguments defined
        config_path: Optional path to JSON config file
        cli_args: Optional list of CLI arguments (defaults to sys.argv)

    Returns:
        Namespace with merged configuration
    """
    # Parse command-line arguments
    args = parser.parse_args(cli_args)

    # If config path is provided, load and merge
    if config_path and Path(config_path).exists():
        config = load_config(config_path)

        # Get which args were explicitly set on command line
        # by comparing with defaults
        parser_defaults = {
            action.dest: action.default
            for action in parser._actions
            if action.dest != 'help'
        }

        # Merge: config values override defaults, but CLI args override config
        for key, value in config.items():
            if hasattr(args, key):
                # Only use config value if arg wasn't explicitly set on CLI
                current_value = getattr(args, key)
                if current_value == parser_defaults.get(key):
                    setattr(args, key, value)

    return args


def args_to_dict(args: argparse.Namespace) -> Dict[str, Any]:
    """Convert argparse Namespace to dictionary.

    Args:
        args: Argument namespace

    Returns:
        Dictionary representation
    """
    return vars(args)
