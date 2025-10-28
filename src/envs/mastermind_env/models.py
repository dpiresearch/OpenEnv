# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for Mastermind Environment.

This module defines the Action, Observation, and State types for the Mastermind game.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from core.env_server import Action, Observation, State


@dataclass
class MastermindAction(Action):
    """
    Action for Mastermind environment.
    
    Attributes:
        guess: List of integers representing color choices (e.g., [0, 1, 2, 3]).
               Each integer is an index into the available colors (0 to num_colors-1).
    """
    guess: List[int]


@dataclass
class MastermindObservation(Observation):
    """
    Observation from Mastermind environment.
    
    This represents the feedback after making a guess.
    
    Attributes:
        black_pegs: Number of correct colors in correct positions.
        white_pegs: Number of correct colors in wrong positions.
        attempts_remaining: Number of attempts left.
        all_previous_guesses: List of all previous guesses made so far.
        all_previous_feedback: List of (black_pegs, white_pegs) for all previous guesses.
        game_phase: String describing the current phase ("playing", "won", "lost").
    """
    black_pegs: int
    white_pegs: int
    attempts_remaining: int
    all_previous_guesses: List[List[int]] = field(default_factory=list)
    all_previous_feedback: List[tuple[int, int]] = field(default_factory=list)
    game_phase: str = "playing"


@dataclass
class MastermindState(State):
    """
    State for Mastermind environment.
    
    Attributes:
        secret_code: The secret code to guess (hidden from agent in production).
        code_length: Length of the secret code (typically 4).
        num_colors: Number of possible colors (typically 6).
        max_attempts: Maximum number of guessing attempts allowed (typically 10).
        current_attempts: Number of attempts used so far.
        guesses_history: List of all guesses made.
        feedback_history: List of all feedback given.
        is_solved: Whether the code has been guessed correctly.
    """
    secret_code: List[int] = field(default_factory=list)
    code_length: int = 4
    num_colors: int = 6
    max_attempts: int = 10
    current_attempts: int = 0
    guesses_history: List[List[int]] = field(default_factory=list)
    feedback_history: List[tuple[int, int]] = field(default_factory=list)
    is_solved: bool = False

