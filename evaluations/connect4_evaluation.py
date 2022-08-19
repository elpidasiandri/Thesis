from envs import Connect4Env
from stable_baselines3 import DQN
from time import sleep

# TODO: make this a callable script, select:
#  - PPO or DQN
#  - model folder to load from
#  - number of games to play
#  - human or agent first

# TODO: change env to AEC, make it simpler

if __name__ == '__main__':
    n_games = 1

    env = Connect4Env()

    custom_objects = {'n_envs': 1}
    model = DQN.load("../experiments/models_bin/Connect4CnnMlp_final.zip", device="cpu", custom_objects=custom_objects)

    print("Agent goes first as P1, human goes second as P2")

    for _ in range(n_games):
        env.reset()
        print("The game begins!")
        sleep(2)
        env.render()

        while True:
            sleep(2)
            agent_action = model.predict(env.aec_env.states["player_0"][1], deterministic=True)[0].item()
            print("\nAgent plays: " + str(agent_action + 1))
            env.aec_env._update_state("player_0", agent_action)
            env.aec_env.render(agent="player_0")

            if env.aec_env._check_win(env.aec_env.states["player_0"][1, 0]):
                print("Agent wins!")
                break

            while True:
                human_action = input("\nHuman plays: ")
                try:
                    human_action = int(human_action) - 1
                except:
                    print("Action should be an integer between 1 and " + env.aec_env.width)
                    continue
                if human_action > (env.aec_env.width - 1) or human_action < 0:
                    print("Action should be between 1 and " + env.aec_env.width)
                    continue
                break
            env.aec_env._update_state("player_1", human_action)
            env.aec_env.render(agent="player_1")

            if env.aec_env._check_win(env.aec_env.states["player_1"][1, 0]):
                print("Human wins!")
                break
