import numpy as np
import torch
from tqdm import trange

import new_environment
from TD3 import TD3
import matplotlib.pyplot as plt
import ReplayBuffer
import main as env
from scale_mid import environment_mid
from new_environment import DRL_Environment
from environment import Environment
from scale_min import environment_min2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(seed, Max_episode, steps):
    environment = environment_mid
    env_with_Dead = True
    state_dim = environment.services * environment.nodes + len(environment.start_service) + 1
    action_dim = 3
    max_action = 1.0
    expl_noise = 0.25
    print('  state_dim:', state_dim, '  action_dim:', action_dim, '  max_a:', max_action)

    random_seed = seed

    # Max_episode = 1000

    if random_seed:
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)

    kwargs = {
        "env_with_Dead": env_with_Dead,
        "state_dim": state_dim,
        "action_dim": action_dim,
        "max_action": max_action,
        "gamma": 0.99,
        "net_width": 200,
        "a_lr": 1e-6,
        "c_lr": 1e-6,
        "Q_batchsize": 600,
        "critic_tau": 0.00005,
        "actor_tau": 0.0000005,
    }
    model = TD3(**kwargs)
    replay_buffer = ReplayBuffer.ReplayBuffer(state_dim, action_dim, max_size=int(1e6))
    result_y = []
    # state_vector = []
    # line = []

    state = torch.tensor(
        [0.0000, 0.0000, 1.0000, 0.0000, 1.0000, 0.0000, 1.0000, 1.0000, 0.0000,
         0.0000, 1.0000, 0.0000, 1.0000, 1.0000, 1.0000, 1.0000, 0.0000, 0.0000,
         0.0000, 1.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 1.0000,
         1.0000, 1.0000, 0.0000, 1.0000, 0.0000, 1.0000, 0.0000, 1.0000, 0.0000,
         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 1.0000, 0.0000, 0.0000, 1.0000,
         0.0000, 1.0000, 0.0000, 0.0000, 0.0000, 1.0000, 0.0000, 1.0000, 1.0000,
         0.0000, 0.0000, 0.0000, 1.0000, 0.0000, 0.0000, 1.0000, 1.0000, 0.0000,
         0.0000, 1.0000, 0.0000, 0.0000, 1.0000, 1.0000, 1.0000, 1.0000, 0.0000,
         0.0000, 0.0000, 0.0000, 1.0000, 0.0000, 1.0000, 1.0000, 1.0000, 0.0000,
         0.0000, 1.0000, 1.0000, 0.0000, 1.0000, 1.0000, 0.0000, 0.0000, 1.0000,
         1.0000, 0.0000, 0.0000, 0.0000, 1.0000, 0.0000, 0.0000, 0.0000, 0.0000,
         1.0000, 0.9000, 0.8200, 0.4800, 0.6700])

    # environment.update_state(state)
    # while True:
    #     if not environment.instance_constrains() or not environment.node_capacity_constrains() or \
    #             environment.cost_constrains() > 0:
    #         state = environment.reset()
    #         environment.update_state(state)
    #     else:
    #         break

    environment.update_state(state)
    ori_reward = environment.get_reward()

    for episode in trange(Max_episode):
        s = state.clone()
        # s = environment.constrains_reset()
        environment.update_state(s)
        done = False
        ep_r = 0
        # max_reward = -99999
        expl_noise *= 0.999
        r = 0
        '''Interact & train'''
        for step in range(steps):
            # print(f"episode: {episode}, step: {step}")
            a = (model.select_action(s) + np.random.normal(0, max_action * expl_noise, size=action_dim)
                 ).clip(-max_action, max_action)
            pre_s = s.clone()
            s_prime, r, done = environment.step_solo(s, a)
            # max_reward = max(max_reward, r)

            replay_buffer.add(s, a, r, s_prime, done)
            if done:
                # s_prime = pre_s

                # environment.update_state(pre_s)
                # pre_r = ori_reward - environment.get_reward()
                # pre_r = new_environment.clamp(pre_r, -500, 500)
                result_y.append(ep_r)
                break

                # print("done")
                # print(s_prime)

            if replay_buffer.size > 1000:
                model.train(replay_buffer)

            s = s_prime
            ep_r += r

        if not done:
            result_y.append(ep_r)
        # new_reward = environment.heuristic_algorithm_fitness_function(s.detach().cpu().numpy())
        # if new_reward < reward:
        #     reward = new_reward
        #     state = s

    # print("y:\n", result_y[-1], "\nstate\n", state_vector[-1])
        if episode % 100 == 0:
            plt.plot(result_y)
            # plt.savefig(f"image/not_reset_modify_reward_with_dead_{Max_episode}_{steps}.svg")
            plt.savefig(f"mid/a0000005c00001epo_{episode}.svg")
            # plt.show()
            torch.save(model.actor, f"mid/a0000005c00001epo_{episode}.pt")
    # print("state:\n", state)
    # print("reward: ", reward)


if __name__ == '__main__':
    main(1, 10000, 300)
