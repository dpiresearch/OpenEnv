# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
StockPatternEnv HTTP Client.

This module provides the client for connecting to a Stock Pattern Environment server
over HTTP.
"""

from __future__ import annotations

from typing import Any, Dict, TYPE_CHECKING

from core.client_types import StepResult
from core.http_env_client import HTTPEnvClient

from .models import StockPatternAction, StockPatternObservation, StockPatternState

if TYPE_CHECKING:
    from core.containers.runtime import ContainerProvider


class StockPatternEnv(HTTPEnvClient[StockPatternAction, StockPatternObservation]):
    """
    HTTP client for Stock Pattern Environment.

    This client connects to a StockPatternEnvironment HTTP server and provides
    methods to interact with it: reset(), step(), and state access.

    Example:
        >>> # Connect to a running server
        >>> client = StockPatternEnv(base_url="http://localhost:8000")
        >>> result = client.reset()
        >>> print(result.observation.prices)
        >>>
        >>> # Trade the pattern
        >>> result = client.step(StockPatternAction(action_type="trade", trade_action=1))
        >>> print(result.observation.portfolio_value)
        >>>
        >>> # Identify the pattern
        >>> result = client.step(StockPatternAction(
        ...     action_type="identify",
        ...     pattern_guess="head_and_shoulders"
        ... ))
        >>> print(result.reward)

    Example with Docker:
        >>> # Automatically start container and connect
        >>> client = StockPatternEnv.from_docker_image("stock-patterns-env:latest")
        >>> result = client.reset()
        >>> result = client.step(StockPatternAction(action_type="trade", trade_action=1))
    """

    def _step_payload(self, action: StockPatternAction) -> Dict[str, Any]:
        """
        Convert StockPatternAction to JSON payload for step request.

        Args:
            action: StockPatternAction instance.

        Returns:
            Dictionary representation suitable for JSON encoding.
        """
        return {
            "action_type": action.action_type,
            "trade_action": action.trade_action,
            "pattern_guess": action.pattern_guess,
        }

    def _parse_result(
        self, payload: Dict[str, Any]
    ) -> StepResult[StockPatternObservation]:
        """
        Parse server response into StepResult[StockPatternObservation].

        Args:
            payload: JSON response from server.

        Returns:
            StepResult with StockPatternObservation.
        """
        obs_data = payload.get("observation", {})

        observation = StockPatternObservation(
            prices=obs_data.get("prices", []),
            volumes=obs_data.get("volumes", []),
            current_price=obs_data.get("current_price", 0.0),
            position=obs_data.get("position", 0),
            cash=obs_data.get("cash", 10000.0),
            portfolio_value=obs_data.get("portfolio_value", 10000.0),
            pattern_name=obs_data.get("pattern_name"),
            pattern_progress=obs_data.get("pattern_progress", 0.0),
            available_actions=obs_data.get("available_actions", ["trade", "identify"]),
            done=payload.get("done", False),
            reward=payload.get("reward"),
            metadata=obs_data.get("metadata", {}),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict[str, Any]) -> StockPatternState:
        """
        Parse server response into StockPatternState object.

        Args:
            payload: JSON response from /state endpoint.

        Returns:
            StockPatternState object with environment state information.
        """
        return StockPatternState(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
            pattern_name=payload.get("pattern_name", "unknown"),
            pattern_difficulty=payload.get("pattern_difficulty", 0.5),
            initial_cash=payload.get("initial_cash", 10000.0),
            total_patterns_seen=payload.get("total_patterns_seen", 0),
            correct_identifications=payload.get("correct_identifications", 0),
            total_return=payload.get("total_return", 0.0),
        )

