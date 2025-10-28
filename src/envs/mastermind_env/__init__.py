# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Mastermind Environment Integration.

This module provides an implementation of the classic Mastermind code-breaking game
as an OpenEnv environment. Perfect for testing and training code-breaking strategies.

Game Rules:
- A secret code of colored pegs is generated
- The agent tries to guess the code within a limited number of attempts
- After each guess, feedback is provided:
  * Black pegs: correct colors in correct positions
  * White pegs: correct colors in wrong positions
- The game ends when the code is cracked or attempts are exhausted

Default Configuration:
- Code length: 4 pegs
- Number of colors: 6
- Maximum attempts: 10
- Duplicate colors: Allowed
"""

from .client import MastermindEnv
from .models import MastermindAction, MastermindObservation, MastermindState

__all__ = ["MastermindEnv", "MastermindAction", "MastermindObservation", "MastermindState"]

