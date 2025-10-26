# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Stock Patterns Environment Server Implementation.

This module simulates common technical stock chart patterns for training
pattern recognition and trading strategies.
"""

import uuid
import random
import math
from typing import List, Tuple

import numpy as np

from core.env_server import Action, Environment, Observation

from ..models import StockPatternAction, StockPatternObservation, StockPatternState


# Define the 10 most common technical patterns
PATTERNS = [
    "head_and_shoulders",
    "inverse_head_and_shoulders",
    "cup_and_handle",
    "double_top",
    "double_bottom",
    "ascending_triangle",
    "descending_triangle",
    "symmetrical_triangle",
    "flag",
    "pennant",
]


class StockPatternEnvironment(Environment):
    """
    Stock Pattern Environment for OpenEnv.

    This environment generates synthetic stock price data following common
    technical chart patterns. Agents can either trade the patterns or
    practice identifying them.

    Supported patterns:
    1. Head and Shoulders - Bearish reversal
    2. Inverse Head and Shoulders - Bullish reversal
    3. Cup and Handle - Bullish continuation
    4. Double Top - Bearish reversal
    5. Double Bottom - Bullish reversal
    6. Ascending Triangle - Bullish continuation
    7. Descending Triangle - Bearish continuation
    8. Symmetrical Triangle - Neutral continuation
    9. Flag - Continuation pattern
    10. Pennant - Continuation pattern

    Args:
        pattern_name: Name of pattern to simulate (random if None)
        difficulty: Pattern difficulty (0.0=easy/clear, 1.0=hard/noisy)
        window_size: Number of historical prices to show
        initial_cash: Starting cash amount
        mode: "trade" or "identify" - what the agent should do

    Example:
        >>> env = StockPatternEnvironment("head_and_shoulders", difficulty=0.3)
        >>> obs = env.reset()
        >>> print(obs.prices)
        >>> obs = env.step(StockPatternAction(action_type="trade", trade_action=1))
        >>> print(obs.reward)
    """

    def __init__(
        self,
        pattern_name: str = None,
        difficulty: float = 0.5,
        window_size: int = 50,
        initial_cash: float = 10000.0,
        mode: str = "trade",
    ):
        """Initialize Stock Pattern environment."""
        super().__init__()

        self.pattern_name = pattern_name
        self.difficulty = max(0.0, min(1.0, difficulty))
        self.window_size = window_size
        self.initial_cash = initial_cash
        self.mode = mode

        # Initialize state
        self._state = StockPatternState(
            pattern_difficulty=self.difficulty,
            initial_cash=initial_cash,
        )

        # Episode-specific variables
        self._prices: List[float] = []
        self._volumes: List[float] = []
        self._pattern_prices: List[float] = []
        self._current_step = 0
        self._position = 0  # -1=Short, 0=Neutral, 1=Long
        self._cash = initial_cash
        self._shares = 0.0
        self._entry_price = 0.0
        self._pattern_identified = False
        self._episode_done = False

    def reset(self) -> Observation:
        """
        Reset the environment and generate a new pattern.

        Returns:
            Initial observation with first few prices.
        """
        # Generate new episode ID
        self._state.episode_id = str(uuid.uuid4())
        self._state.step_count = 0

        # Select pattern
        if self.pattern_name is None:
            self._current_pattern = random.choice(PATTERNS)
        else:
            self._current_pattern = self.pattern_name

        self._state.pattern_name = self._current_pattern
        self._state.total_patterns_seen += 1

        # Generate pattern prices
        self._pattern_prices = self._generate_pattern(self._current_pattern)

        # Reset episode variables
        self._current_step = 0
        self._position = 0
        self._cash = self.initial_cash
        self._shares = 0.0
        self._entry_price = 0.0
        self._pattern_identified = False
        self._episode_done = False

        # Initialize price history with first few points
        initial_points = min(10, len(self._pattern_prices) // 4)
        self._prices = self._pattern_prices[:initial_points]
        self._volumes = [random.uniform(0.8, 1.2) for _ in range(initial_points)]

        return self._make_observation(reward=0.0, done=False)

    def step(self, action: Action) -> Observation:
        """
        Execute action and advance the pattern.

        Args:
            action: StockPatternAction containing trade or identification action.

        Returns:
            Observation after action execution.

        Raises:
            ValueError: If action is not a StockPatternAction.
        """
        if not isinstance(action, StockPatternAction):
            raise ValueError(f"Expected StockPatternAction, got {type(action)}")

        reward = 0.0

        # Handle identification action
        if action.action_type == "identify" and not self._pattern_identified:
            if action.pattern_guess == self._current_pattern:
                reward = 100.0  # Correct identification
                self._pattern_identified = True
                self._state.correct_identifications += 1
            else:
                reward = -10.0  # Incorrect identification

        # Handle trade action
        elif action.action_type == "trade":
            reward = self._execute_trade(action.trade_action)

        # Advance to next price point
        self._current_step += 1
        self._state.step_count += 1

        if self._current_step < len(self._pattern_prices):
            self._prices.append(self._pattern_prices[self._current_step])
            self._volumes.append(random.uniform(0.8, 1.2))
        else:
            # Pattern complete, close any open positions
            if self._position != 0:
                reward += self._close_position()
            self._episode_done = True

        # Update state
        current_value = self._calculate_portfolio_value()
        self._state.total_return = ((current_value - self.initial_cash) / self.initial_cash) * 100

        return self._make_observation(reward=reward, done=self._episode_done)

    @property
    def state(self) -> StockPatternState:
        """Get current environment state."""
        return self._state

    def _generate_pattern(self, pattern_name: str) -> List[float]:
        """
        Generate synthetic price data for the specified pattern.

        Args:
            pattern_name: Name of the pattern to generate.

        Returns:
            List of prices forming the pattern.
        """
        base_price = 100.0
        pattern_length = random.randint(40, 80)
        noise_level = self.difficulty * 5.0

        prices = []

        if pattern_name == "head_and_shoulders":
            prices = self._generate_head_and_shoulders(base_price, pattern_length, noise_level)
        elif pattern_name == "inverse_head_and_shoulders":
            prices = self._generate_inverse_head_and_shoulders(base_price, pattern_length, noise_level)
        elif pattern_name == "cup_and_handle":
            prices = self._generate_cup_and_handle(base_price, pattern_length, noise_level)
        elif pattern_name == "double_top":
            prices = self._generate_double_top(base_price, pattern_length, noise_level)
        elif pattern_name == "double_bottom":
            prices = self._generate_double_bottom(base_price, pattern_length, noise_level)
        elif pattern_name == "ascending_triangle":
            prices = self._generate_ascending_triangle(base_price, pattern_length, noise_level)
        elif pattern_name == "descending_triangle":
            prices = self._generate_descending_triangle(base_price, pattern_length, noise_level)
        elif pattern_name == "symmetrical_triangle":
            prices = self._generate_symmetrical_triangle(base_price, pattern_length, noise_level)
        elif pattern_name == "flag":
            prices = self._generate_flag(base_price, pattern_length, noise_level)
        elif pattern_name == "pennant":
            prices = self._generate_pennant(base_price, pattern_length, noise_level)
        else:
            # Default to simple uptrend
            prices = [base_price + i * 0.5 + random.gauss(0, noise_level) 
                     for i in range(pattern_length)]

        return prices

    def _generate_head_and_shoulders(self, base: float, length: int, noise: float) -> List[float]:
        """Generate Head and Shoulders pattern (bearish reversal)."""
        prices = []
        segment = length // 7
        
        # Uptrend to first shoulder
        for i in range(segment):
            prices.append(base + i * 0.3 + random.gauss(0, noise))
        
        # First shoulder peak
        peak1 = prices[-1] + 5
        for i in range(segment):
            t = i / segment
            prices.append(peak1 - t * 5 + random.gauss(0, noise))
        
        # Up to head
        for i in range(segment):
            t = i / segment
            prices.append(prices[-1] + t * 10 + random.gauss(0, noise))
        
        # Head peak and down
        head = prices[-1]
        for i in range(segment):
            t = i / segment
            prices.append(head - t * 10 + random.gauss(0, noise))
        
        # Second shoulder
        for i in range(segment):
            t = i / segment
            prices.append(prices[-1] + t * 5 + random.gauss(0, noise))
        
        # Second shoulder down
        for i in range(segment):
            t = i / segment
            prices.append(prices[-1] - t * 5 + random.gauss(0, noise))
        
        # Break neckline (bearish)
        for i in range(segment):
            prices.append(prices[-1] - 0.5 + random.gauss(0, noise))
        
        return prices

    def _generate_inverse_head_and_shoulders(self, base: float, length: int, noise: float) -> List[float]:
        """Generate Inverse Head and Shoulders pattern (bullish reversal)."""
        # Invert the head and shoulders pattern
        h_and_s = self._generate_head_and_shoulders(base, length, noise)
        avg = sum(h_and_s) / len(h_and_s)
        return [2 * avg - price for price in h_and_s]

    def _generate_cup_and_handle(self, base: float, length: int, noise: float) -> List[float]:
        """Generate Cup and Handle pattern (bullish continuation)."""
        prices = []
        cup_length = int(length * 0.7)
        handle_length = length - cup_length
        
        # Cup - U-shaped bottom
        for i in range(cup_length):
            t = i / cup_length
            # Parabolic shape
            price = base - 15 * (4 * t * (1 - t)) + random.gauss(0, noise)
            prices.append(price)
        
        # Handle - slight downward drift
        handle_start = prices[-1]
        for i in range(handle_length):
            t = i / handle_length
            price = handle_start - 3 * t + random.gauss(0, noise)
            prices.append(price)
        
        # Breakout
        for i in range(10):
            prices.append(prices[-1] + 1.0 + random.gauss(0, noise))
        
        return prices

    def _generate_double_top(self, base: float, length: int, noise: float) -> List[float]:
        """Generate Double Top pattern (bearish reversal)."""
        prices = []
        segment = length // 5
        
        # Up to first peak
        for i in range(segment):
            prices.append(base + i * 0.5 + random.gauss(0, noise))
        
        # First peak
        peak = prices[-1] + 5
        prices.append(peak + random.gauss(0, noise))
        
        # Down to trough
        for i in range(segment):
            t = i / segment
            prices.append(peak - t * 8 + random.gauss(0, noise))
        
        # Up to second peak (similar height)
        for i in range(segment):
            t = i / segment
            prices.append(prices[-1] + t * 8 + random.gauss(0, noise))
        
        # Second peak
        prices.append(peak + random.gauss(0, noise * 2))
        
        # Breakdown
        for i in range(segment * 2):
            prices.append(prices[-1] - 0.5 + random.gauss(0, noise))
        
        return prices

    def _generate_double_bottom(self, base: float, length: int, noise: float) -> List[float]:
        """Generate Double Bottom pattern (bullish reversal)."""
        # Invert double top
        d_top = self._generate_double_top(base, length, noise)
        avg = sum(d_top) / len(d_top)
        return [2 * avg - price for price in d_top]

    def _generate_ascending_triangle(self, base: float, length: int, noise: float) -> List[float]:
        """Generate Ascending Triangle pattern (bullish continuation)."""
        prices = []
        resistance = base + 15
        
        for i in range(length):
            t = i / length
            # Rising lows, flat highs
            low = base + t * 10
            high = resistance
            
            # Oscillate between rising low and flat high
            cycle = math.sin(t * math.pi * 4)
            price = low + (high - low) * (cycle + 1) / 2 + random.gauss(0, noise)
            prices.append(price)
        
        # Breakout
        for i in range(10):
            prices.append(resistance + i * 0.5 + random.gauss(0, noise))
        
        return prices

    def _generate_descending_triangle(self, base: float, length: int, noise: float) -> List[float]:
        """Generate Descending Triangle pattern (bearish continuation)."""
        prices = []
        support = base - 15
        
        for i in range(length):
            t = i / length
            # Flat lows, falling highs
            low = support
            high = base - t * 10
            
            # Oscillate between flat low and falling high
            cycle = math.sin(t * math.pi * 4)
            price = low + (high - low) * (cycle + 1) / 2 + random.gauss(0, noise)
            prices.append(price)
        
        # Breakdown
        for i in range(10):
            prices.append(support - i * 0.5 + random.gauss(0, noise))
        
        return prices

    def _generate_symmetrical_triangle(self, base: float, length: int, noise: float) -> List[float]:
        """Generate Symmetrical Triangle pattern (continuation)."""
        prices = []
        
        for i in range(length):
            t = i / length
            # Converging highs and lows
            amplitude = 10 * (1 - t)
            cycle = math.sin(t * math.pi * 5)
            price = base + amplitude * cycle + random.gauss(0, noise)
            prices.append(price)
        
        # Breakout (50/50 up or down)
        direction = random.choice([1, -1])
        for i in range(10):
            prices.append(prices[-1] + direction * 0.5 + random.gauss(0, noise))
        
        return prices

    def _generate_flag(self, base: float, length: int, noise: float) -> List[float]:
        """Generate Flag pattern (continuation)."""
        prices = []
        pole_length = length // 3
        flag_length = length - pole_length
        
        # Pole - strong move
        direction = random.choice([1, -1])
        for i in range(pole_length):
            prices.append(base + direction * i * 0.8 + random.gauss(0, noise))
        
        # Flag - consolidation (slight counter-trend)
        flag_start = prices[-1]
        for i in range(flag_length):
            t = i / flag_length
            price = flag_start - direction * t * 3 + random.gauss(0, noise)
            prices.append(price)
        
        # Breakout (continuation)
        for i in range(10):
            prices.append(prices[-1] + direction * 0.8 + random.gauss(0, noise))
        
        return prices

    def _generate_pennant(self, base: float, length: int, noise: float) -> List[float]:
        """Generate Pennant pattern (continuation)."""
        prices = []
        pole_length = length // 3
        pennant_length = length - pole_length
        
        # Pole - strong move
        direction = random.choice([1, -1])
        for i in range(pole_length):
            prices.append(base + direction * i * 0.8 + random.gauss(0, noise))
        
        # Pennant - converging oscillation
        pennant_start = prices[-1]
        for i in range(pennant_length):
            t = i / pennant_length
            amplitude = 5 * (1 - t)
            cycle = math.sin(t * math.pi * 4)
            price = pennant_start + amplitude * cycle + random.gauss(0, noise)
            prices.append(price)
        
        # Breakout (continuation)
        for i in range(10):
            prices.append(prices[-1] + direction * 0.8 + random.gauss(0, noise))
        
        return prices

    def _execute_trade(self, trade_action: int) -> float:
        """
        Execute a trade action and return the reward.

        Args:
            trade_action: 0=Hold, 1=Buy, 2=Sell

        Returns:
            Reward for this trade.
        """
        current_price = self._prices[-1]
        reward = 0.0

        if trade_action == 1:  # Buy
            if self._position == 0 and self._cash >= current_price:
                # Open long position
                self._shares = self._cash / current_price
                self._cash = 0.0
                self._position = 1
                self._entry_price = current_price
        
        elif trade_action == 2:  # Sell
            if self._position == 1:
                # Close long position
                self._cash = self._shares * current_price
                profit = (current_price - self._entry_price) / self._entry_price
                reward = profit * 100  # Reward based on percentage profit
                self._shares = 0.0
                self._position = 0
                self._entry_price = 0.0

        return reward

    def _close_position(self) -> float:
        """Close any open position and return profit/loss."""
        if self._position == 0:
            return 0.0
        
        current_price = self._prices[-1]
        profit = (current_price - self._entry_price) / self._entry_price
        self._cash = self._shares * current_price
        self._shares = 0.0
        self._position = 0
        
        return profit * 100

    def _calculate_portfolio_value(self) -> float:
        """Calculate current portfolio value."""
        if self._position == 0:
            return self._cash
        else:
            current_price = self._prices[-1]
            return self._shares * current_price

    def _make_observation(self, reward: float, done: bool) -> StockPatternObservation:
        """
        Create observation from current state.

        Args:
            reward: Reward for the last action.
            done: Whether episode is complete.

        Returns:
            StockPatternObservation.
        """
        # Get window of recent prices
        window_start = max(0, len(self._prices) - self.window_size)
        windowed_prices = self._prices[window_start:]
        windowed_volumes = self._volumes[window_start:]

        # Normalize prices for observation (relative to first price in window)
        if windowed_prices:
            base_price = windowed_prices[0]
            normalized_prices = [(p - base_price) / base_price for p in windowed_prices]
        else:
            normalized_prices = []

        # Only reveal pattern name if identified or episode done
        pattern_name = None
        if self._pattern_identified or done:
            pattern_name = self._current_pattern

        # Calculate pattern progress
        progress = min(1.0, len(self._prices) / len(self._pattern_prices))

        portfolio_value = self._calculate_portfolio_value()

        obs = StockPatternObservation(
            prices=normalized_prices,
            volumes=windowed_volumes,
            current_price=self._prices[-1] if self._prices else 0.0,
            position=self._position,
            cash=self._cash,
            portfolio_value=portfolio_value,
            pattern_name=pattern_name,
            pattern_progress=progress,
            done=done,
            reward=reward,
        )

        return obs

