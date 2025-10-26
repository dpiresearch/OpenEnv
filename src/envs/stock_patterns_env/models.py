# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for Stock Patterns Environment.

This module defines the Action, Observation, and State types for stock pattern simulation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from core.env_server import Action, Observation, State


@dataclass
class StockPatternAction(Action):
    """
    Action for Stock Pattern environments.

    The agent can take two types of actions:
    1. Trade action: 0=Hold, 1=Buy, 2=Sell
    2. Pattern identification: Identify which pattern is currently forming

    Attributes:
        action_type: Type of action ("trade" or "identify")
        trade_action: Trading action (0=Hold, 1=Buy, 2=Sell) if action_type is "trade"
        pattern_guess: Pattern name if action_type is "identify"
    """
    action_type: str = "trade"  # "trade" or "identify"
    trade_action: int = 0  # 0=Hold, 1=Buy, 2=Sell
    pattern_guess: Optional[str] = None


@dataclass
class StockPatternObservation(Observation):
    """
    Observation from Stock Pattern environment.

    This represents what the agent sees at each step, including price data,
    technical indicators, and pattern information.

    Attributes:
        prices: List of historical prices (normalized)
        volumes: List of historical volumes (normalized)
        current_price: Current stock price
        position: Current position (-1=Short, 0=Neutral, 1=Long)
        cash: Available cash
        portfolio_value: Total portfolio value
        pattern_name: Name of the current pattern (only revealed at end or if guessed correctly)
        pattern_progress: How far along the pattern is (0.0 to 1.0)
        available_actions: List of available action types
    """
    prices: List[float]
    volumes: List[float]
    current_price: float
    position: int = 0  # -1=Short, 0=Neutral, 1=Long
    cash: float = 10000.0
    portfolio_value: float = 10000.0
    pattern_name: Optional[str] = None
    pattern_progress: float = 0.0
    available_actions: List[str] = field(default_factory=lambda: ["trade", "identify"])


@dataclass
class StockPatternState(State):
    """
    State for Stock Pattern environment.

    Attributes:
        pattern_name: Current pattern being simulated
        pattern_difficulty: Difficulty level (0.0=easy, 1.0=hard)
        initial_cash: Starting cash amount
        total_patterns_seen: Total number of patterns encountered
        correct_identifications: Number of correctly identified patterns
        total_return: Cumulative return percentage
    """
    pattern_name: str = "head_and_shoulders"
    pattern_difficulty: float = 0.5
    initial_cash: float = 10000.0
    total_patterns_seen: int = 0
    correct_identifications: int = 0
    total_return: float = 0.0

