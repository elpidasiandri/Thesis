from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import VecMonitor

from algorithms import DeviceAlternatingDQN
from envs import VecConnect4Env
from policies import Connect4CnnMlpDQNPolicy

model_name = "Connect4CnnMlp"

if __name__ == '__main__':
    num_instances = 8
    gamma_half_life = 10  # measured in number of moves
    # gamma = np.exp(np.log(0.5) / gamma_half_life)
    gamma = 0.95

    env = VecMonitor(VecConnect4Env(num_instances))

    model = DeviceAlternatingDQN(
        Connect4CnnMlpDQNPolicy,
        env,
        batch_size=2048,
        gamma=gamma,
        exploration_fraction=0.4,
        tensorboard_log="../bin",
        verbose=1,
        dva=False
    )

    callbacks = [CheckpointCallback(20480, "../models_bin/" + model_name, model_name)]

    model.learn(total_timesteps=50_000_000,
                callback=callbacks,
                tb_log_name=model_name)
    model.save("models_bin/" + model_name + "_final")

    env.close()
