import copy
import numpy as np
import math

import torch

# from PPO.scale_min import environment_min
# from PPO.scale_7_7 import environment_7services_7nodes
# from PPO.scale_mid import environment_mid
# from PPO.scale_12_12 import environment_12services_12nodes
# from PPO.scale_max import environment_max
# from PPO.new_environment import DRL_Environment

from scale_min import environment_min
from scale_7_7 import environment_7services_7nodes
from scale_mid import environment_mid
from scale_12_12 import environment_12services_12nodes
from scale_max import environment_max
from new_environment import DRL_Environment

scale_5_5 = copy.deepcopy(environment_min)
scale_5_5.instance = np.zeros((scale_5_5.services, scale_5_5.nodes))
scale_5_5.delta = np.ones(len(scale_5_5.start_service))
scale_7_7 = copy.deepcopy(environment_7services_7nodes)
scale_7_7.instance = np.zeros((scale_7_7.services, scale_7_7.nodes))
scale_7_7.delta = np.ones(len(scale_7_7.start_service))
scale_10_10 = copy.deepcopy(environment_mid)
scale_10_10.instance = np.zeros((scale_10_10.services, scale_10_10.nodes))
scale_10_10.delta = np.ones(len(scale_10_10.start_service))
scale_12_12 = copy.deepcopy(environment_12services_12nodes)
scale_12_12.instance = np.zeros((scale_12_12.services, scale_12_12.nodes))
scale_12_12.delta = np.ones(len(scale_12_12.start_service))
scale_15_15 = copy.deepcopy(environment_max)
scale_15_15.instance = np.zeros((scale_15_15.services, scale_15_15.nodes))
scale_15_15.delta = np.ones(len(scale_15_15.start_service))


# 极小型指标转化为极大型指标的函数
def minTomax(maxx, x):
    x = list(x)  # 将输入的指标数据转换为列表
    ans = [[(maxx - e)] for e in x]  # 计算最大值与每个指标值的差，并将其放入新列表中
    return np.array(ans)  # 将列表转换为numpy数组并返回


# n是参评数目，m是指标数目
def topsis(environment: DRL_Environment, A):
    n = environment.nodes
    m = 5
    # 统一指标类型，将所有指标转化为极大型指标
    X = np.zeros(shape=(n, 1))
    for i in range(m):
        if i == 0:
            maxA = max(A[:, i])
            v = minTomax(maxA, A[:, i])
            X = v.reshape(-1, 1)  # 如果是第一个指标，直接替换X数组
        else:
            v = np.array(A[:, i])
            X = np.hstack([X, v.reshape(-1, 1)])  # 如果不是第一个指标，则将新指标列拼接到X数组上
    # print("统一指标后矩阵为：\n{}".format(X))

    # 对统一指标后的矩阵X进行标准化处理
    X = X.astype('float')
    for j in range(m):
        X[:, j] = X[:, j] / np.sqrt(sum(X[:, j] ** 2))  # 对每一列数据进行归一化处理，即除以该列的欧几里得范数
    # print("标准化矩阵为：\n{}".format(X))  # 打印标准化后的矩阵X

    # 最大值最小值距离的计算
    x_max = np.max(X, axis=0)  # 计算标准化矩阵每列的最大值
    x_min = np.min(X, axis=0)  # 计算标准化矩阵每列的最小值
    d_z = np.sqrt(np.sum(np.square((X - np.tile(x_max, (n, 1)))), axis=1))  # 计算每个参评对象与最优情况的距离d+
    d_f = np.sqrt(np.sum(np.square((X - np.tile(x_min, (n, 1)))), axis=1))  # 计算每个参评对象与最劣情况的距离d-
    # print('每个指标的最大值:', x_max)
    # print('每个指标的最小值:', x_min)
    # print('d+向量:', d_z)
    # print('d-向量:', d_f)

    # 计算每个参评对象的得分排名
    s = d_f / (d_z + d_f)  # 根据d+和d-计算得分s，其中s接近于1则表示较优，接近于0则表示较劣
    Score = 100 * s / sum(s)  # 将得分s转换为百分制，便于比较
    # for i in range(len(Score)):
    #     print(f"第{i + 1}个标准化后百分制得分为：{Score[i]}")  # 打印每个参评对象的得分
    return Score


def get_value_matrix(environment: DRL_Environment):
    cpu_capacity = environment.node_resource_capacity[:, 0]
    ram_capacity = environment.node_resource_capacity[:, 1]
    disk_capacity = environment.node_resource_capacity[:, 2]
    net_ability = get_net_importance(environment)
    compute_ability = get_compute_ability(environment)
    # 矩阵由网络重要性、节点处理能力、cpu容量、内存容量组成
    return np.column_stack((net_ability, compute_ability, cpu_capacity, ram_capacity, disk_capacity))


def get_compute_ability(environment: DRL_Environment):
    return environment.compute_ability[0]


def get_net_importance(environment: DRL_Environment):
    importance_list = []
    for node in range(environment.nodes):
        delay = 0
        for node_id in range(environment.nodes):
            delay += environment.net_delay[node][node_id]
        importance_list.append(delay / (environment.nodes - 1))
    return importance_list


def get_service_importance(environment: DRL_Environment):
    importance_list = []
    for service in range(environment.services):
        upstream_service_list = get_upstream_service_list(environment, service)
        importance = environment.compute_time[service][0]
        for upstream_service, pop in upstream_service_list:
            importance += environment.compute_time[upstream_service][0] * math.exp(-pop ** 2)
        importance_list.append(importance)
    return importance_list


def get_upstream_service_list(environment, service):
    service_dependency = environment.service_dependency
    upstream_service_list = []
    current_services = [service]
    front_services = []
    # 记录所有上游微服务和与当前微服务的跳数
    pop = 1
    while True:
        for i in range(environment.services):
            for current_service in current_services:
                if service_dependency[i][current_service] == 1:
                    upstream_service_list.append((i, pop))
                    front_services.append(i)
        if not front_services:
            break
        current_services = front_services
        front_services = []
        pop += 1
    return upstream_service_list


def greedy_deploy(environment: DRL_Environment):
    A = get_value_matrix(environment)
    # print(A)
    service_priority = get_service_importance(environment)
    node_priority = topsis(environment, A)

    # 先按照优先级把所有微服务都部署一遍
    # 按优先级依次试能不能部署
    sorted_service_idx = np.argsort(service_priority)
    sorted_reverse_service_idx = sorted_service_idx[::-1]

    for service in sorted_reverse_service_idx:

        flag = False

        sorted_node_idx = np.argsort(node_priority)
        sorted_reverse_node_idx = sorted_node_idx[::-1]

        for node in sorted_reverse_node_idx:
            if flag:
                break

            # 看是不是未部署状态
            if environment.instance[service][node] == 0:
                # 看容量够不够
                if environment.service_resource_occupancy[service][0] <= A[node][2] and \
                        environment.service_resource_occupancy[service][1] <= A[node][3] and \
                        environment.service_resource_occupancy[service][2] <= A[node][4]:
                    environment.instance[service][node] = 1
                    # 更新A中资源容量
                    A[node][2] -= environment.service_resource_occupancy[service][0]
                    A[node][3] -= environment.service_resource_occupancy[service][1]
                    A[node][4] -= environment.service_resource_occupancy[service][2]

                    node_priority = topsis(environment, A)
                    flag = True
    # print(A)

    while True:
        # 按优先级依次试能不能部署
        sorted_service_idx = np.argsort(service_priority)
        sorted_reverse_service_idx = sorted_service_idx[::-1]

        # 增加flag看是否有服务部署成功
        flag = False

        for service in sorted_reverse_service_idx:
            # 如果已经部署成功，就不再部署
            if flag:
                break

            sorted_node_idx = np.argsort(node_priority)
            sorted_reverse_node_idx = sorted_node_idx[::-1]

            for node in sorted_reverse_node_idx:
                if flag:
                    break

                # 看是不是未部署状态
                if environment.instance[service][node] == 0:
                    # 看容量够不够
                    if environment.service_resource_occupancy[service][0] <= A[node][2] and \
                            environment.service_resource_occupancy[service][1] <= A[node][3] and \
                            environment.service_resource_occupancy[service][2] <= A[node][4]:
                        environment.instance[service][node] = 1

                        # 更新A中资源容量
                        A[node][2] -= environment.service_resource_occupancy[service][0]
                        A[node][3] -= environment.service_resource_occupancy[service][1]
                        A[node][4] -= environment.service_resource_occupancy[service][2]

                        # print(f"部署了服务{service}到节点{node}")
                        # print(A)

                        # 部署的微服务的优先级减半
                        service_priority[service] /= 2
                        node_priority = topsis(environment, A)
                        flag = True
            # print(A)
        # 如果没有服务部署成功，就退出
        if not flag:
            break


def min_route_importance(environment: DRL_Environment):
    greedy_deploy(environment)
    # print(environment.instance)
    route_importance = 0.01
    min_state = None
    delay = 100000
    while route_importance < 1:
        instance_reset = torch.tensor(environment.instance.flatten())
        delta_reset = torch.ones(len(environment.start_service))
        request_rate_reset = torch.tensor(np.array([route_importance]))
        state = torch.cat((instance_reset, delta_reset, request_rate_reset))
        environment.update_state(state)
        if environment.check_constrains():
            total_delay = environment.get_reward()
            if total_delay < delay:
                min_state = state
                delay = total_delay
        route_importance += 0.01
    return min_state


def min_request_in(environment: DRL_Environment):
    state = min_route_importance(environment)
    environment.update_state(state)
    while True:
        max_time_app = environment.get_max_time_app()
        # print(max_time_app)
        # print(environment.remain_cost())
        # print(environment.request_out)
        # print(0.01 * environment.request_out[max_time_app])
        if environment.remain_cost() >= 0.01 * environment.request_out[max_time_app] * environment.app_fee:
            state[environment.services * environment.nodes + max_time_app] -= 0.01
            environment.update_state(state)
        else:
            break
    return state


def get_state_list():
    state_list = [min_request_in(scale_5_5), min_request_in(scale_7_7), min_request_in(scale_10_10),
                  min_request_in(scale_12_12), min_request_in(scale_15_15)]
    return state_list


if __name__ == '__main__':
    # print(scale_5_5.instance)
    # greedy_deploy(scale_5_5)
    # print(scale_5_5.service_resource_occupancy)
    # print(min_route_importance(scale_5_5))
    # print(scale_5_5.instance)
    greedy_state = min_request_in(scale_5_5)
    scale_5_5.update_state(greedy_state)
    print(scale_5_5.check_constrains())
    print(f"scale_5_5, reward: {scale_5_5.get_reward()}")

    greedy_state = min_request_in(scale_7_7)
    scale_7_7.update_state(greedy_state)
    print(scale_7_7.check_constrains())
    print(f"scale_7_7, reward: {scale_7_7.get_reward()}")

    greedy_state = min_request_in(scale_10_10)
    scale_10_10.update_state(greedy_state)
    print(scale_10_10.check_constrains())
    print(f"scale_10_10, reward: {scale_10_10.get_reward()}")

    greedy_state = min_request_in(scale_12_12)
    scale_12_12.update_state(greedy_state)
    print(scale_12_12.check_constrains())
    print(f"scale_12_12, reward: {scale_12_12.get_reward()}")

    greedy_state = min_request_in(scale_15_15)
    scale_15_15.update_state(greedy_state)
    print(scale_15_15.check_constrains())
    print(f"scale_15_15, reward: {scale_15_15.get_reward()}")

