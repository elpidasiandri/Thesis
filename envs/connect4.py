from typing import Optional

import gym
import numpy as np
from gym.spaces import Box, Discrete
from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector
from pettingzoo.utils.env import ObsType, ActionType
from scipy.signal import convolve2d


# TODO: Complete this
#  Multi-agent environment, PettingZoo:
#  https://www.pettingzoo.ml/environment_creation#example-custom-environment
#  https://www.pettingzoo.ml/wrappers#included-multi-agent-only-functions


class Connect4Env(AECEnv):
    _win = 0
    _loss = 1
    _invalid = 2

    def __init__(self,
                 width: int = 7,
                 height: int = 6,
                 connect: int = 4):
        super(Connect4Env, self).__init__()

        self.width = width
        self.height = height
        self.connect = connect

        self._observation_space = Box(low=0, high=2, shape=[1, height, width])
        self._action_space = Discrete(width)

        self.initial_state = np.zeros(self._observation_space.shape)

        self._build_winning_convolutions()

        self.possible_agents = ["player_" + str(p) for p in range(2)]

    def _build_winning_convolutions(self):
        self.convolutions = []
        self.convolutions.append(np.array([[1] * self.connect]))
        self.convolutions.append(np.array([[1]] * self.connect))
        self.convolutions.append(np.identity(self.connect))
        self.convolutions.append(np.diag(np.ones(self.connect)))

    def observation_space(self, agent: str) -> gym.Space:
        return self._observation_space

    def action_space(self, agent: str) -> gym.Space:
        return self._action_space

    def observe(self, agent: str) -> ObsType:
        # TODO: Fix this, it should be different, the states change as players play iteratively
        state = self.state.copy()
        if self.possible_agents.index(agent) == 1:
            p_1_mask, p_2_mask = state == 1, state == 2
            state[p_1_mask] = 2
            state[p_2_mask] = 1
        return state

    def _check_win(self):
        for conv in self.convolutions:
            filled = self.state == (1 + self.switch)
            if np.any(convolve2d(filled, conv, mode="valid") / self.connect == 1):
                return True
        return False

    def step(self, action: ActionType) -> None:
        agent = self.agent_selection
        if self.dones[agent]:
            return self._was_done_step(action)

        self._cumulative_rewards[agent] = 0
        self.state[agent] = 0
        if self._agent_selector.is_last():
            # reward based on state logic
            pass
            # dones logic
            pass

            # update the current state
            for a in self.agents:
                self.observations[a] = self.state[a]


        # TODO: this is really really wrong. There are two different rollouts.
        # if self.done:
        #     return self.state, -20, True, {}
        #
        # player = self.switch + 1
        # row = np.argmax(self.state[0, :, action] == 0)
        #
        # # case: column is full
        # if self.state[0, row, action] != 0:
        #     # TODO: fix state depending on player
        #     return self.state, -1, False, {}
        #
        # self.state[0, row, action] = player
        #
        # # case: column is valid
        # self.done = self._check_win()
        # rew = 20 if self.done else 1
        #
        # self.switch = not self.switch
        # # TODO: fix state depending on player
        # return self.state, rew, False, {}

    def reset(self, **kwargs):
        self.agents = self.possible_agents[:]
        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = self.rewards.copy()
        self.dones = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        self.state = {agent: self.initial_state.copy() for agent in self.agents}
        self.observations = self.state.copy()

        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.next()

    def render(self, mode="human"):
        pass

    def seed(self, seed: Optional[int] = None) -> None:
        pass
