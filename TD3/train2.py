import numpy as np
import torch
from tqdm import trange

import new_environment
from TD3 import TD3
import matplotlib.pyplot as plt
import ReplayBuffer
import main as env
from new_environment import DRL_Environment
from environment import Environment
from scale_min import environment_min2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(seed, Max_episode, steps):
    env_with_Dead = True
    state_dim = env.rows * env.cols + len(env.start_service) + 1
    # action_dim = 3 + len(env.start_service) + 1
    action_dim = 3 + 2 + 1  # 3：在x,y坐标上是否放置服务，2：在x索引的服务上增减网关因子，1：增减重要因子
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
        "a_lr": 1e-5,
        "c_lr": 1e-5,
        "Q_batchsize": 600,
        "critic_tau": 0.0005,
        "actor_tau": 0.000005,
    }
    model = TD3(**kwargs)
    replay_buffer = ReplayBuffer.ReplayBuffer(state_dim, action_dim, max_size=int(1e6))
    result_y = []
    # state_vector = []
    # line = []
    environment = environment_min2
    state = torch.tensor([0.0000, 0.0000, 1.0000, 0.0000, 0.0000, 1.0000, 1.0000, 0.0000, 1.0000,
                          0.0000, 0.0000, 0.0000, 1.0000, 0.0000, 0.0000, 0.0000, 0.0000, 1.0000,
                          1.0000, 1.0000, 0.0000, 0.0000, 0.0000, 0.0000, 1.0000, 0.8000, 0.8300,
                          0.7000])
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
        max_reward = -99999
        expl_noise *= 0.999
        r = 0
        '''Interact & train'''
        for step in range(steps):
            # print(f"episode: {episode}, step: {step}")
            a = (model.select_action(s) + np.random.normal(0, max_action * expl_noise, size=action_dim)
                 ).clip(-max_action, max_action)
            pre_s = s.clone()
            s_prime, r, done = environment.step(s, a)
            max_reward = max(max_reward, r)

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
    plt.plot(result_y)
    # plt.savefig(f"image/not_reset_modify_reward_with_dead_{Max_episode}_{steps}.svg")
    plt.savefig(f"2024-04-26/a000005c0005.svg")
    # plt.show()
    torch.save(model.actor, f"2024-04-26/a000005c0005.pt")
    # print("state:\n", state)
    # print("reward: ", reward)


if __name__ == '__main__':
    main(1, 3000, 200)
