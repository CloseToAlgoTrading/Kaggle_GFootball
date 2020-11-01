"""Policy implementation that generates random actions."""
from __future__ import absolute_import
from __future__ import division
# Using Type Annotations.
from __future__ import print_function

from typing import Optional, Sequence

import numpy as np
import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import
from tf_agents.distributions import masked
from tf_agents.policies import py_policy
from tf_agents.specs import array_spec
from tf_agents.trajectories import policy_step
from tf_agents.trajectories import time_step as ts
from tf_agents.typing import types
from tf_agents.utils import nest_utils
from tf_agents.environments import utils

from human_agent import human_agent_wrapper
from human_agent import simple_human_policy

class MyFootbalHumanPyPolicy(py_policy.PyPolicy):
    """Returns random samples of the given action_spec."""
    
    
    def __convert_tf_obs_to_numpy(self, obs):
        
        rp = {}
        new_obs = {}
        for name in obs.keys():
            rp[name] = obs[name].numpy()[0] if len(obs[name].numpy()[0]) > 1 else obs[name].numpy()[0][0]

        new_obs['players_raw'] = [rp]
        return new_obs

    
    def __init__(self,
               time_step_spec: ts.TimeStep,
               action_spec: types.NestedArraySpec,
               info_spec: types.NestedArraySpec = (),
               seed: Optional[types.Seed] = None,
               outer_dims: Optional[Sequence[int]] = None,
               observation_and_action_constraint_splitter: Optional[
                   types.Splitter] = None):
        self._seed = seed
        self._outer_dims = outer_dims
        
        self._rng = np.random.RandomState(seed)
        if time_step_spec is None:
            time_step_spec = ts.time_step_spec()

        super(MyFootbalHumanPyPolicy, self).__init__(
            time_step_spec=time_step_spec,
            action_spec=action_spec,
            info_spec=info_spec,
            observation_and_action_constraint_splitter=(
                observation_and_action_constraint_splitter))

    def _action(self, time_step, policy_state):
        
        outer_dims = self._outer_dims
        
        if outer_dims is None:
            if self.time_step_spec.observation:
                outer_dims = nest_utils.get_outer_array_shape(
                    time_step.observation, self.time_step_spec.observation)
            else:
                outer_dims = ()

        random_action = np.array([simple_human_policy(human_agent_wrapper(self.__convert_tf_obs_to_numpy(time_step.observation))).value])

        info = array_spec.sample_spec_nest(
            self._info_spec, self._rng, outer_dims=outer_dims)

        return policy_step.PolicyStep(random_action, policy_state, info)

