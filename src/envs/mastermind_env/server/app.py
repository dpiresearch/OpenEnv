# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
FastAPI application for the Mastermind Environment.

This module creates an HTTP server that exposes the Mastermind game
over HTTP endpoints, making it compatible with HTTPEnvClient.

Usage:
    # Development (with auto-reload):
    uvicorn envs.mastermind_env.server.app:app --reload --host 0.0.0.0 --port 8000
    
    # Production:
    uvicorn envs.mastermind_env.server.app:app --host 0.0.0.0 --port 8000 --workers 4
    
    # Or run directly:
    python -m envs.mastermind_env.server.app

Environment variables:
    MASTERMIND_CODE_LENGTH: Length of the secret code (default: 4)
    MASTERMIND_NUM_COLORS: Number of possible colors (default: 6)
    MASTERMIND_MAX_ATTEMPTS: Maximum number of attempts (default: 10)
    MASTERMIND_ALLOW_DUPLICATES: Whether to allow duplicate colors (default: true)
"""

import os

from core.env_server import create_app

from ..models import MastermindAction, MastermindObservation
from .mastermind_environment import MastermindEnvironment

# Get configuration from environment variables
code_length = int(os.getenv("MASTERMIND_CODE_LENGTH", "4"))
num_colors = int(os.getenv("MASTERMIND_NUM_COLORS", "6"))
max_attempts = int(os.getenv("MASTERMIND_MAX_ATTEMPTS", "10"))
allow_duplicates = os.getenv("MASTERMIND_ALLOW_DUPLICATES", "true").lower() == "true"

# Create the environment instance
env = MastermindEnvironment(
    code_length=code_length,
    num_colors=num_colors,
    max_attempts=max_attempts,
    allow_duplicates=allow_duplicates,
)

# Create the FastAPI app with web interface and README integration
app = create_app(env, MastermindAction, MastermindObservation, env_name="mastermind_env")


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(app, host="0.0.0.0", port=8000)

