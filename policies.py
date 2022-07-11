import torch as th
from stable_baselines3.common.policies import ActorCriticPolicy

from models import SudokuACNetwork, ActorCriticCNNNetwork


class SudokuACPolicy(ActorCriticPolicy):

    def __init__(self, *args, **kwargs):
        super(SudokuACPolicy, self).__init__(*args, **kwargs)

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = SudokuACNetwork()

    # Bypass observation preprocessing and features extractor
    def extract_features(self, obs: th.Tensor) -> th.Tensor:
        return obs.float()  # Handle Double type tensor error


class ActorCriticCNNPolicy(ActorCriticPolicy):

    def __init__(self, *args, **kwargs):
        super(ActorCriticCNNPolicy, self).__init__(*args, **kwargs)

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = ActorCriticCNNNetwork()

    # Bypass observation preprocessing and features extractor
    def extract_features(self, obs: th.Tensor) -> th.Tensor:
        return obs.float()  # Handle Double type tensor error
