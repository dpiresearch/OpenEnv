# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Stock Patterns Environment Integration.

This module provides an environment for simulating and learning common
technical stock chart patterns. It's useful for training agents to:
1. Recognize technical patterns (head and shoulders, cup and handle, etc.)
2. Trade based on pattern signals
3. Practice technical analysis

Supported patterns:
- Head and Shoulders (bearish reversal)
- Inverse Head and Shoulders (bullish reversal)
- Cup and Handle (bullish continuation)
- Double Top (bearish reversal)
- Double Bottom (bullish reversal)
- Ascending Triangle (bullish continuation)
- Descending Triangle (bearish continuation)
- Symmetrical Triangle (neutral continuation)
- Flag (continuation)
- Pennant (continuation)
"""

from .client import StockPatternEnv
from .models import StockPatternAction, StockPatternObservation, StockPatternState

__all__ = [
    "StockPatternEnv",
    "StockPatternAction",
    "StockPatternObservation",
    "StockPatternState",
]

