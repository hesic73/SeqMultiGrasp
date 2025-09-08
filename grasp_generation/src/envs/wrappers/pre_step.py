from gymnasium import Wrapper, Env


from typing import Any, Optional, Tuple, Sequence, Dict, List, Union, Callable


class PreStepWrapper(Wrapper):
    def __init__(self, env, func: Callable[[Env,], None]):
        super().__init__(env)
        self._func = func

    def step(self, action):
        self._func(self.env)
        obs, reward, terminated, truncated, info = self.env.step(action)
        return obs, reward, terminated, truncated, info


# for video recording when num_envs=1 only
class _PreStepOrAnyDoneWrapper(Wrapper):
    def __init__(self, env, func: Callable[[Env,], None], done_indices: Optional[List[int]] = None):
        super().__init__(env)
        self._func = func
        self._done_indices = done_indices

    def step(self, action):
        self._func(self.env.unwrapped)
        obs, reward, terminated, truncated, info = self.env.step(action)
        done = terminated | truncated
        if self._done_indices is None:
            any_done = done.any()
        else:
            any_done = done[self._done_indices].any()
        if any_done:
            self._func(self.env.unwrapped)
        return obs, reward, terminated, truncated, info
