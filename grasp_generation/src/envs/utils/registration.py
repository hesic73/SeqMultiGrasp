import json
from copy import deepcopy
from functools import partial

import gymnasium as gym
from gymnasium.envs.registration import WrapperSpec

from mani_skill import logger


from mani_skill.utils.registration import register, REGISTERED_ENVS, make, make_vec

from typing import List


def register_env(
    uid: str,
    override=False,
    additional_wrappers: List[WrapperSpec] = [],
    **kwargs,
):
    """A decorator to register ManiSkill environments.

    Args:
        uid (str): unique id of the environment.
        override (bool): whether to override the environment if it is already registered.

    Notes:
        - `max_episode_steps` is processed differently from other keyword arguments in gym.
          `gym.make` wraps the env with `gym.wrappers.TimeLimit` to limit the maximum number of steps.
        - `gym.EnvSpec` uses kwargs instead of **kwargs!
    """
    try:
        json.dumps(kwargs)
    except TypeError:
        raise RuntimeError(
            f"You cannot register_env with non json dumpable kwargs, e.g. classes or types. If you really need to do this, it is recommended to create a mapping of string to the unjsonable data and to pass the string in the kwarg and during env creation find the data you need"
        )

    def _register_env(cls):

        max_episode_steps: int = cls.max_episode_steps
        assert max_episode_steps is not None, "max_episode_steps must be set"
        assert isinstance(max_episode_steps,
                          int), "max_episode_steps must be an integer"

        if uid in REGISTERED_ENVS:
            if override:
                from gymnasium.envs.registration import registry

                logger.warn(f"Override registered env {uid}")
                REGISTERED_ENVS.pop(uid)
                registry.pop(uid)
            else:
                logger.warn(
                    f"Env {uid} is already registered. Skip registration.")
                return cls

        # Register for ManiSkill
        register(
            uid,
            cls,
            max_episode_steps=max_episode_steps,
            default_kwargs=deepcopy(kwargs),
        )

        # Register for gym
        gym.register(
            uid,
            entry_point=partial(make, env_id=uid),
            vector_entry_point=partial(make_vec, env_id=uid),
            max_episode_steps=max_episode_steps,
            disable_env_checker=True,  # Temporary solution as we allow empty observation spaces
            kwargs=deepcopy(kwargs),
            additional_wrappers=[
                WrapperSpec(
                    "MSTimeLimit",
                    entry_point="mani_skill.utils.registration:TimeLimitWrapper",
                    kwargs=dict(max_episode_steps=max_episode_steps),
                )
            ]+additional_wrappers,
        )

        return cls

    return _register_env
