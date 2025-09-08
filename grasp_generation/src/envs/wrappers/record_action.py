from gymnasium import Wrapper, Env

import torch

from typing import Any, Optional, Tuple, Sequence, Dict, List, Union, Callable


class RecordAction(Wrapper):
    def __init__(self, env, record_action_func: Callable):
        super().__init__(env)
        self._record_action_func = record_action_func

    def step(self, action):
        self._record_action_func(action)
        obs, reward, terminated, truncated, info = self.env.step(action)
        return obs, reward, terminated, truncated, info
