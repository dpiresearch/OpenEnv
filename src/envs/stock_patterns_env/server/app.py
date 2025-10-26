# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
FastAPI application for the Stock Patterns Environment.

This module creates an HTTP server that exposes stock pattern simulation
over HTTP endpoints, making it compatible with HTTPEnvClient.

Usage:
    # Development (with auto-reload):
    uvicorn envs.stock_patterns_env.server.app:app --reload --host 0.0.0.0 --port 8000

    # Production:
    uvicorn envs.stock_patterns_env.server.app:app --host 0.0.0.0 --port 8000 --workers 4

    # Or run directly:
    python -m envs.stock_patterns_env.server.app

Environment variables:
    STOCK_PATTERN_NAME: Pattern name to serve (default: random)
    STOCK_PATTERN_DIFFICULTY: Difficulty level 0.0-1.0 (default: 0.5)
    STOCK_PATTERN_MODE: "trade" or "identify" (default: "trade")
"""

import os

from core.env_server import create_app

from ..models import StockPatternAction, StockPatternObservation
from .stock_patterns_environment import StockPatternEnvironment

# Get configuration from environment variables
pattern_name = os.getenv("STOCK_PATTERN_NAME", None)  # None = random
difficulty = float(os.getenv("STOCK_PATTERN_DIFFICULTY", "0.5"))
mode = os.getenv("STOCK_PATTERN_MODE", "trade")

# Create the environment instance
env = StockPatternEnvironment(
    pattern_name=pattern_name,
    difficulty=difficulty,
    mode=mode,
)

# Create the FastAPI app with web interface and README integration
app = create_app(
    env, 
    StockPatternAction, 
    StockPatternObservation, 
    env_name="stock_patterns_env"
)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)

