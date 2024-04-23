import numpy as np
import torch
from tqdm import trange

from TD3 import TD3
import matplotlib.pyplot as plt
import ReplayBuffer
import main as env
from new_environment import DRL_Environment
from environment import Environment

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

app_fee = 1000
cpu_fee = 1
ram_fee = 0.1
disk_fee = 0.01
max_fee = 8000
app_1_request = 50
app_2_request = 30
rows = 5
cols = 5
max_time = 999
start_service = [0, 3]
delta = np.random.rand(len(start_service))
lambda_out = [app_1_request, app_2_request]
lambda_in = delta * lambda_out
access_node = [0, 3]
second = 1000
service_resource_occupancy = np.array([
    [2.0, 512, 50],
    [1.0, 256, 40],
    [1.0, 380, 120],
    [0.5, 128, 20],
    [1.5, 420, 70],
])
node_resource_capacity = np.array([
    [16, 2048, 2048],
    [10, 2048, 2048],
    [7, 2048, 2048],
    [4, 1024, 2048],
    [12, 2048, 2048],
])
instance = np.random.randint(2, size=(rows, cols))
service_dependency = np.array([
    [0, 1, 0, 0, 0],
    [0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1],
    [0, 1, 0, 0, 0]
])
net_delay = np.array([
    [1, 10, 25, 20, 50],
    [10, 1, 15, 20, 50],
    [25, 15, 1, 5, 35],
    [20, 20, 5, 1, 30],
    [50, 50, 35, 30, 1]
])
compute_time = np.array([
    [12, 10, 8, 6, 4],
    [28, 28 / 6 * 5, 28 / 3 * 2, 28 / 2, 28 / 3],
    [17, 17 / 6 * 5, 17 / 3 * 2, 17 / 2, 17 / 3],
    [9, 9 / 6 * 5, 9 / 3 * 2, 9 / 2, 9 / 3],
    [78, 78 / 6 * 5, 78 / 3 * 2, 78 / 2, 78 / 3]
])
compute_ability = second / np.array(compute_time)
request_arrive = np.zeros((rows, cols))


def main(seed, Max_episode, steps):
    env_with_Dead = False
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
        "a_lr": 1e-4,
        "c_lr": 1e-4,
        "Q_batchsize": 256,
    }
    model = TD3(**kwargs)
    replay_buffer = ReplayBuffer.ReplayBuffer(state_dim, action_dim, max_size=int(1e6))
    result_y = []
    # state_vector = []
    # line = []
    environment = DRL_Environment(app_fee, cpu_fee, ram_fee, disk_fee, max_fee, rows, cols, max_time, lambda_out,
                                  start_service, access_node, service_resource_occupancy, node_resource_capacity,
                                  instance, service_dependency, net_delay, compute_time)
    state = torch.tensor([0.0000, 0.0000, 1.0000, 0.0000, 0.0000, 1.0000, 1.0000, 0.0000, 1.0000,
                          0.0000, 0.0000, 0.0000, 1.0000, 0.0000, 0.0000, 0.0000, 0.0000, 1.0000,
                          1.0000, 1.0000, 0.0000, 0.0000, 0.0000, 0.0000, 1.0000, 0.9900, 0.8300,
                          0.7000])
    # environment.update_state(state)
    # while True:
    #     if not environment.instance_constrains() or not environment.node_capacity_constrains() or \
    #             environment.cost_constrains() > 0:
    #         state = environment.reset()
    #         environment.update_state(state)
    #     else:
    #         break

    reward = 999999

    for episode in trange(Max_episode):
        s = state.clone()
        environment.update_state(s)
        done = False
        ep_r = 0
        expl_noise *= 0.99
        r = 1000
        '''Interact & train'''
        for step in range(steps):
            a = (model.select_action(s) + np.random.normal(0, max_action * expl_noise, size=action_dim)
                 ).clip(-max_action, max_action)
            pre_s = s.clone()
            s_prime, r, done = environment.step(s, a)

            replay_buffer.add(s, a, r, s_prime, done)
            if done:
                s_prime = pre_s
                environment.update_state(s_prime)
                # print("done")
                # result_y.append(pre_r)
                # break

            if replay_buffer.size > 1000:
                model.train(replay_buffer)

            s = s_prime
            ep_r += r

        result_y.append(ep_r)
        # new_reward = environment.heuristic_algorithm_fitness_function(s.detach().cpu().numpy())
        # if new_reward < reward:
        #     reward = new_reward
        #     state = s

    # print("y:\n", result_y[-1], "\nstate\n", state_vector[-1])
    plt.plot(result_y)
    plt.savefig(f"image/not_reset_modify_reward_with_dead_{Max_episode}_{steps}.svg")
    # plt.show()
    torch.save(model.actor, f"model/not_reset_modify_reward_with_dead_{Max_episode}_{steps}.pt")
    # print("state:\n", state)
    # print("reward: ", reward)


if __name__ == '__main__':
    main(1, 500, 150)
