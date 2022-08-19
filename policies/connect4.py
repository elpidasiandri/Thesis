import torch as th
from stable_baselines3.common.policies import ActorCriticPolicy

from models import Connect4CnnMlpACNetwork


class Connect4CnnMlpACPolicy(ActorCriticPolicy):

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = Connect4CnnMlpACNetwork()

    # Bypass observation preprocessing and features extractor
    def extract_features(self, obs: th.Tensor) -> th.Tensor:
        return obs.float()  # Handle Double type tensor error
