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
    def __init__(self,
                 width: int = 7,
                 height: int = 6,
                 connect: int = 4):
        super(Connect4Env, self).__init__()

        self.width = width
        self.height = height
        self.connect = connect

        # 1 channel, height rows, width columns
        self._observation_space = Box(low=0, high=2, shape=[1, height, width])
        # width actions
        self._action_space = Discrete(width)

        self.initial_state = np.stack([np.full(self._observation_space.shape, np.NaN),
                                       np.zeros(self._observation_space.shape)])

        self._build_winning_convolutions()

        self.possible_agents = ["player_" + str(p) for p in range(2)]
        self.agent_name_mapping = dict(zip(self.possible_agents, list(range(2))))

    def _build_winning_convolutions(self):
        self.convolutions = []
        self.convolutions.append(np.array([[1] * self.connect]))
        self.convolutions.append(np.array([[1]] * self.connect))
        self.convolutions.append(np.identity(self.connect))
        self.convolutions.append(np.flip(np.identity(self.connect), 0))

    # return the same object
    def observation_space(self, agent: str) -> gym.Space:
        return self._observation_space

    # return the same object
    def action_space(self, agent: str) -> gym.Space:
        return self._action_space

    def observe(self, agent: str) -> ObsType:
        return self.observations[agent]

    def reset(self, **kwargs):
        self.agents = self.possible_agents[:]
        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = self.rewards.copy()
        self.dones = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        self.states = {agent: self.initial_state.copy() for agent in self.agents}
        self.observations = self.states.copy()

        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.next()

    def _check_win(self, state: np.ndarray):
        # current player is 1, opponent is 2
        filled = state == 1
        for conv in self.convolutions:
            if np.any(convolve2d(filled, conv, mode="valid") / self.connect == 1):
                return True
        return False

    def _compute_rewards_dones(self):
        """
        Updates rewards and dones when the round is over
        """
        rewards = []
        dones = [False, False]
        for a in self.states:
            # invalid move - no change
            if (self.states[a][0] == self.states[a][1]).all():
                rewards.append(-4)
            # win (lose)
            elif self._check_win(self.states[a][1, 0]):
                rewards = [10, -10]
                dones = [True, True]
                if self.agent_name_mapping[a]:
                    rewards.reverse()
                break
            # unfinished
            else:
                rewards.append(1)

        self.rewards[self.agents[0]], self.rewards[self.agents[1]] = rewards
        self.dones[self.agents[0]], self.dones[self.agents[1]] = dones

    def _update_obss(self):
        """Updates observations"""
        self.observations = {agent: self.states[agent][0] for agent in self.agents}

    def _update_state(self, agent, column):
        """Builds the state of an agent using an action"""
        # use the previous state for invalid move comparison
        old_state = self.states[self.agents[self.agent_name_mapping[agent] - 1]][1].copy()
        # swap 1s and 2s
        mask_1, mask_2 = old_state == 1, old_state == 2
        old_state[mask_1], old_state[mask_2] = 2, 1

        new_state = old_state.copy()
        if not (new_state[:, column]).all():  # if not invalid
            last_row = np.where(new_state[0, :, column] > 0)[0]
            try:
                last_row = last_row[0]
            except:
                last_row = self.height
            new_state[0, last_row - 1, column] = 1

        self.states[agent] = np.stack([old_state, new_state])

    def step(self, action: ActionType) -> None:
        agent = self.agent_selection
        if self.dones[agent]:
            return self._was_done_step(action)

        self._cumulative_rewards[agent] = 0  # don't accumulate rewards
        self._update_state(agent, action)

        if self._agent_selector.is_last():  # when the round is over
            self._compute_rewards_dones()
            self._update_obss()

        self.agent_selection = self._agent_selector.next()
        self._accumulate_rewards()  # compute rewards

    def render(self, mode="human"):
        # Empty for now
        pass

    def seed(self, seed: Optional[int] = None) -> None:
        # No seed for now
        pass

    def close(self):
        # Nothing to close
        pass
