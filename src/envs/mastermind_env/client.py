# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
MastermindEnv HTTP Client.

This module provides the client for connecting to a Mastermind Environment server
over HTTP.
"""

from __future__ import annotations

from typing import Any, Dict, TYPE_CHECKING

from core.client_types import StepResult

from core.http_env_client import HTTPEnvClient

from .models import MastermindAction, MastermindObservation, MastermindState

if TYPE_CHECKING:
    from core.containers.runtime import ContainerProvider


class MastermindEnv(HTTPEnvClient[MastermindAction, MastermindObservation]):
    """
    HTTP client for Mastermind Environment.
    
    This client connects to a MastermindEnvironment HTTP server and provides
    methods to interact with it: reset(), step(), and state access.
    
    Example:
        >>> # Connect to a running server
        >>> client = MastermindEnv(base_url="http://localhost:8000")
        >>> result = client.reset()
        >>> print(result.observation.attempts_remaining)
        >>>
        >>> # Take an action (make a guess)
        >>> result = client.step(MastermindAction(guess=[0, 1, 2, 3]))
        >>> print(f"Black: {result.observation.black_pegs}, White: {result.observation.white_pegs}")
    
    Example with Docker:
        >>> # Automatically start container and connect
        >>> client = MastermindEnv.from_docker_image("mastermind-env:latest")
        >>> result = client.reset()
        >>> result = client.step(MastermindAction(guess=[0, 1, 2, 3]))
    """
    
    def _step_payload(self, action: MastermindAction) -> Dict[str, Any]:
        """
        Convert MastermindAction to JSON payload for step request.
        
        Args:
            action: MastermindAction instance.
            
        Returns:
            Dictionary representation suitable for JSON encoding.
        """
        return {
            "guess": action.guess,
        }
    
    def _parse_result(
        self, payload: Dict[str, Any]
    ) -> StepResult[MastermindObservation]:
        """
        Parse server response into StepResult[MastermindObservation].
        
        Args:
            payload: JSON response from server.
            
        Returns:
            StepResult with MastermindObservation.
        """
        obs_data = payload.get("observation", {})
        
        observation = MastermindObservation(
            black_pegs=obs_data.get("black_pegs", 0),
            white_pegs=obs_data.get("white_pegs", 0),
            attempts_remaining=obs_data.get("attempts_remaining", 0),
            all_previous_guesses=obs_data.get("all_previous_guesses", []),
            all_previous_feedback=obs_data.get("all_previous_feedback", []),
            game_phase=obs_data.get("game_phase", "playing"),
            done=payload.get("done", False),
            reward=payload.get("reward"),
            metadata=obs_data.get("metadata", {}),
        )
        
        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )
    
    def _parse_state(self, payload: Dict[str, Any]) -> MastermindState:
        """
        Parse server response into MastermindState object.
        
        Args:
            payload: JSON response from /state endpoint.
            
        Returns:
            MastermindState object with environment state information.
        """
        return MastermindState(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
            secret_code=payload.get("secret_code", []),
            code_length=payload.get("code_length", 4),
            num_colors=payload.get("num_colors", 6),
            max_attempts=payload.get("max_attempts", 10),
            current_attempts=payload.get("current_attempts", 0),
            guesses_history=payload.get("guesses_history", []),
            feedback_history=payload.get("feedback_history", []),
            is_solved=payload.get("is_solved", False),
        )

