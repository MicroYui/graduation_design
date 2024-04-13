import numpy as np
import torch
import gym
from tqdm import trange

from TD3 import TD3
import matplotlib.pyplot as plt
import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter
import main as env

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(seed):
    env_with_Dead = False
    state_dim = env.rows * env.cols + len(env.start_service) + 1
    action_dim = 3 + len(env.start_service) + 1
    max_action = 1.0
    expl_noise = 0.25
    print('  state_dim:', state_dim, '  action_dim:', action_dim, '  max_a:', max_action)

    random_seed = seed

    Max_episode = 2000

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
        "a_lr": 1e-4,
        "c_lr": 1e-4,
        "Q_batchsize": 256,
    }
    model = TD3(**kwargs)
    replay_buffer = ReplayBuffer.ReplayBuffer(state_dim, action_dim, max_size=int(1e6))

    for episode in trange(Max_episode):
        s = env.reset()
        done = False
        ep_r = 0
        steps = 0
        expl_noise *= 0.999
        r = 0
        '''Interact & train'''
        for step in range(10000):
            a = (model.select_action(s) + np.random.normal(0, max_action * expl_noise, size=action_dim)
                 ).clip(-max_action, max_action)
            s_prime, r = env.step(s, a)

            replay_buffer.add(s, a, r, s_prime, False)

            if replay_buffer.size > 1000:
                model.train(replay_buffer)

            s = s_prime
            ep_r += r
        print('Episode:', episode, ' Reward:', r)


if __name__ == '__main__':
    main(seed=1)
