# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Mastermind Environment Server Implementation.

This module implements the classic Mastermind code-breaking game.
The agent tries to guess a secret code within a limited number of attempts,
receiving feedback (black and white pegs) after each guess.
"""

import random
import uuid
from typing import List, Tuple

from core.env_server import Action, Environment, Observation

from ..models import MastermindAction, MastermindObservation, MastermindState


class MastermindEnvironment(Environment):
    """
    Mastermind code-breaking game environment.
    
    In this game:
    - The environment generates a secret code of colored pegs
    - The agent makes guesses to crack the code
    - After each guess, feedback is provided:
      * Black pegs: correct color in correct position
      * White pegs: correct color in wrong position
    - The game ends when the code is cracked or max attempts are reached
    
    Args:
        code_length: Length of the secret code (default: 4)
        num_colors: Number of possible colors (default: 6)
        max_attempts: Maximum number of guessing attempts (default: 10)
        allow_duplicates: Whether duplicate colors are allowed in the code (default: True)
        
    Example:
        >>> env = MastermindEnvironment(code_length=4, num_colors=6)
        >>> obs = env.reset()
        >>> print(obs.attempts_remaining)  # 10
        >>> obs = env.step(MastermindAction(guess=[0, 1, 2, 3]))
        >>> print(f"Black: {obs.black_pegs}, White: {obs.white_pegs}")
    """
    
    def __init__(
        self,
        code_length: int = 4,
        num_colors: int = 6,
        max_attempts: int = 10,
        allow_duplicates: bool = True,
    ):
        """Initialize Mastermind environment."""
        super().__init__()
        
        if code_length < 1:
            raise ValueError("code_length must be at least 1")
        if num_colors < 2:
            raise ValueError("num_colors must be at least 2")
        if max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        if not allow_duplicates and num_colors < code_length:
            raise ValueError(
                f"Without duplicates, num_colors ({num_colors}) must be >= code_length ({code_length})"
            )
        
        self.code_length = code_length
        self.num_colors = num_colors
        self.max_attempts = max_attempts
        self.allow_duplicates = allow_duplicates
        
        # Initialize state
        self._state = MastermindState(
            code_length=code_length,
            num_colors=num_colors,
            max_attempts=max_attempts,
        )
    
    def reset(self) -> Observation:
        """
        Reset the environment and generate a new secret code.
        
        Returns:
            Initial observation with no feedback yet.
        """
        # Generate new secret code
        if self.allow_duplicates:
            secret_code = [random.randint(0, self.num_colors - 1) for _ in range(self.code_length)]
        else:
            # Sample without replacement
            secret_code = random.sample(range(self.num_colors), self.code_length)
        
        # Reset state
        self._state = MastermindState(
            episode_id=str(uuid.uuid4()),
            step_count=0,
            secret_code=secret_code,
            code_length=self.code_length,
            num_colors=self.num_colors,
            max_attempts=self.max_attempts,
            current_attempts=0,
            guesses_history=[],
            feedback_history=[],
            is_solved=False,
        )
        
        # Return initial observation (no feedback yet)
        return MastermindObservation(
            black_pegs=0,
            white_pegs=0,
            attempts_remaining=self.max_attempts,
            all_previous_guesses=[],
            all_previous_feedback=[],
            game_phase="playing",
            done=False,
            reward=0.0,
        )
    
    def step(self, action: Action) -> Observation:
        """
        Execute agent's guess and return feedback.
        
        Args:
            action: MastermindAction containing the guess.
            
        Returns:
            Observation with feedback (black/white pegs) and game state.
            
        Raises:
            ValueError: If action is not a MastermindAction or guess is invalid.
        """
        if not isinstance(action, MastermindAction):
            raise ValueError(f"Expected MastermindAction, got {type(action)}")
        
        # Validate guess
        guess = action.guess
        if len(guess) != self.code_length:
            raise ValueError(
                f"Guess length {len(guess)} doesn't match code length {self.code_length}"
            )
        
        if any(color < 0 or color >= self.num_colors for color in guess):
            raise ValueError(
                f"Invalid color in guess. Colors must be in range [0, {self.num_colors - 1}]"
            )
        
        # Calculate feedback
        black_pegs, white_pegs = self._calculate_feedback(guess)
        
        # Update state
        self._state.step_count += 1
        self._state.current_attempts += 1
        self._state.guesses_history.append(guess)
        self._state.feedback_history.append((black_pegs, white_pegs))
        
        # Check if code is solved
        is_solved = (black_pegs == self.code_length)
        self._state.is_solved = is_solved
        
        # Determine if game is done
        attempts_exhausted = self._state.current_attempts >= self.max_attempts
        done = is_solved or attempts_exhausted
        
        # Calculate reward
        reward = self._calculate_reward(black_pegs, white_pegs, is_solved, attempts_exhausted)
        
        # Determine game phase
        if is_solved:
            game_phase = "won"
        elif attempts_exhausted:
            game_phase = "lost"
        else:
            game_phase = "playing"
        
        # Create observation
        obs = MastermindObservation(
            black_pegs=black_pegs,
            white_pegs=white_pegs,
            attempts_remaining=self.max_attempts - self._state.current_attempts,
            all_previous_guesses=self._state.guesses_history.copy(),
            all_previous_feedback=self._state.feedback_history.copy(),
            game_phase=game_phase,
            done=done,
            reward=reward,
        )
        
        return obs
    
    @property
    def state(self) -> MastermindState:
        """Get current environment state."""
        return self._state
    
    def _calculate_feedback(self, guess: List[int]) -> Tuple[int, int]:
        """
        Calculate black and white pegs for a guess.
        
        Black peg: correct color in correct position
        White peg: correct color in wrong position
        
        Args:
            guess: The guessed code
            
        Returns:
            Tuple of (black_pegs, white_pegs)
        """
        secret = self._state.secret_code
        
        # Count black pegs (exact matches)
        black_pegs = sum(1 for i in range(self.code_length) if guess[i] == secret[i])
        
        # For white pegs, we need to count color matches in wrong positions
        # Create frequency maps excluding exact matches
        secret_colors = {}
        guess_colors = {}
        
        for i in range(self.code_length):
            if guess[i] != secret[i]:  # Not an exact match
                # Count in secret
                secret_colors[secret[i]] = secret_colors.get(secret[i], 0) + 1
                # Count in guess
                guess_colors[guess[i]] = guess_colors.get(guess[i], 0) + 1
        
        # White pegs are the minimum overlap for each color
        white_pegs = 0
        for color, count in guess_colors.items():
            if color in secret_colors:
                white_pegs += min(count, secret_colors[color])
        
        return black_pegs, white_pegs
    
    def _calculate_reward(
        self, 
        black_pegs: int, 
        white_pegs: int, 
        is_solved: bool, 
        attempts_exhausted: bool
    ) -> float:
        """
        Calculate reward for the current step.
        
        Reward structure:
        - Solving the code: +100 bonus
        - Each black peg: +10 points
        - Each white peg: +1 point
        - Each step: -1 penalty (to encourage efficiency)
        - Running out of attempts: -50 penalty
        
        Args:
            black_pegs: Number of black pegs
            white_pegs: Number of white pegs
            is_solved: Whether the code was solved
            attempts_exhausted: Whether max attempts were reached
            
        Returns:
            Calculated reward
        """
        reward = 0.0
        
        # Base feedback reward
        reward += black_pegs * 10.0
        reward += white_pegs * 1.0
        
        # Step penalty to encourage efficiency
        reward -= 1.0
        
        # Terminal rewards
        if is_solved:
            # Bonus for solving, scaled by remaining attempts
            remaining = self.max_attempts - self._state.current_attempts
            reward += 100.0 + (remaining * 5.0)
        elif attempts_exhausted:
            # Penalty for failing to solve
            reward -= 50.0
        
        return reward

