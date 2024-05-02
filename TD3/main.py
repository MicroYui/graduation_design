import numpy as np
import torch

from environment import Environment
from new_environment import DRL_Environment
from scale_mid import environment_mid
# from scale_min import environment_min

if __name__ == '__main__':

    # actor = torch.load('mid/a0000005c00001epo_9000.pt', map_location=torch.device('cpu'))
    # print(actor)
    # environment = DRL_Environment(app_fee, cpu_fee, ram_fee, disk_fee, max_fee, rows, cols, max_time, lambda_out,
    #                               start_service, access_node, service_resource_occupancy, node_resource_capacity,
    #                               instance, service_dependency, net_delay, compute_time)
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
    environment = environment_mid
    # state = torch.tensor([0.0000, 0.0000, 1.0000, 0.0000, 0.0000, 1.0000, 1.0000, 0.0000, 1.0000,
    #                       0.0000, 0.0000, 0.0000, 1.0000, 0.0000, 0.0000, 0.0000, 0.0000, 1.0000,
    #                       1.0000, 1.0000, 0.0000, 0.0000, 0.0000, 0.0000, 1.0000, 0.8000, 0.8300,
    #                       0.7000])
    # state = torch.tensor(
    #     [0.0000, 1.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
    #      0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 1.0000, 0.0000, 0.0000, 0.0000, 1.0000,
    #      0.0000, 0.0000, 1.0000, 0.0000, 0.0000, 1.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
    #      1.0000, 0.0000, 0.0000, 1.0000, 0.0000, 0.0000, 1.0000, 0.0000, 0.0000, 1.0000, 1.0000, 0.0000,
    #      0.0000, 1.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 1.0000,
    #      0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 1.0000, 0.0000, 0.0000, 1.0000, 0.0000, 0.0000,
    #      0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 1.0000, 0.0000, 0.0000, 0.0000, 0.0000, 1.0000,
    #      0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
    #      1.0000, 0.0000, 0.0000, 0.0000, 0.8800, 0.9100, 0.0600, 0.1000])
    # state = state.detach().cpu().numpy()
    # state = np.array([1, 0, 0, 0, 0, 1,
    #                   1, 0, 1, 0, 0, 0,
    #                   1, 0, 0, 0, 0, 0,
    #                   1, 0, 0, 0, 0, 1,
    #                   0, 0.99, 0.74, 0.99])
    environment.update_state(state)
    ori_reward = environment.get_reward()
    print(environment.get_reward())

    # print(environment.heuristic_algorithm_fitness_function(state))
    # print(environment.request_arrive)

    # print(environment.request_arrive)
    # print(state)

    # s = state
    # environment.update_state(s)
    # for i in range(100):
    #     action = actor(s)
    #     s, _, dead = environment.step_solo(s, action)
    #     print(state)
    #     environment.update_state(s)
    #     print("reward: ", environment.get_reward())
    #     if dead:
    #         print("dead")
    #         # break
    #     print("---------------------")

    # min_state = reset()
    # min_reward = 999999
    # for _ in range(100):
    #     ori_state = reset()
    #     ori_state = ori_state.float()
    #     ori_reward = 0
    #     for _ in range(100):
    #         action = actor(ori_state)
    #         # print(action)
    #         # print(state)
    #         ori_state, ori_reward = step(ori_state, action)
    #     if min_reward > ori_reward:
    #         min_reward = ori_reward
    #         min_state = ori_state
    #
    # print("state:\n", min_state, "\nreward:", min_reward)
