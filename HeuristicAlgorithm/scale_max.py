import numpy as np
import torch

import other_environment
from new_environment import DRL_Environment

app_fee = 1000
cpu_fee = 1
ram_fee = 0.001
disk_fee = 0.001
max_fee = 8000
app_1_request = 50
app_2_request = 30
app_3_request = 30
app_4_request = 30
services = 15
nodes = 15
max_time = 999
start_service = [0, 6, 9, 11]
lambda_out = [app_1_request, app_2_request, app_3_request, app_4_request]
access_node = [0, 3, 8, 13]
service_resource_occupancy = np.array([
    [2.0, 512, 50],
    [1.0, 256, 40],
    [1.0, 380, 120],
    [0.5, 128, 20],
    [1.5, 420, 70],
    [2.0, 512, 50],
    [1.0, 256, 40],
    [1.0, 380, 120],
    [0.5, 128, 20],
    [1.5, 420, 70],
    [2.0, 512, 50],
    [1.0, 256, 40],
    [1.0, 380, 120],
    [0.5, 128, 20],
    [1.0, 256, 40],
])
node_resource_capacity = np.array([
    [10, 2048, 2048],
    [7, 2048, 2048],
    [4, 1024, 2048],
    [20, 8192, 2048],
    [12, 2048, 2048],
    [16, 2048, 2048],
    [10, 2048, 2048],
    [7, 2048, 2048],
    [4, 1024, 2048],
    [12, 2048, 2048],
    [16, 2048, 2048],
    [10, 2048, 2048],
    [7, 2048, 2048],
    [4, 1024, 2048],
    [12, 2048, 2048],
])
instance = np.random.randint(2, size=(services, nodes))
# 服务依赖关系
# 0 → 1 → 2 → 3 → 4 → 5
#         ↑       ↑
# 6 → 7 → 8       ↑
#     ↑           ↑
# 9 → 10          ↑
#                 ↑
# 11 → 12 → 13 → 14
service_dependency = np.zeros((services, services))
service_dependency[0][1] = 1
service_dependency[1][2] = 1
service_dependency[2][3] = 1
service_dependency[3][4] = 1
service_dependency[4][5] = 1
service_dependency[6][7] = 1
service_dependency[7][8] = 1
service_dependency[8][2] = 1
service_dependency[9][10] = 1
service_dependency[10][7] = 1
service_dependency[11][12] = 1
service_dependency[12][13] = 1
service_dependency[13][14] = 1
service_dependency[14][4] = 1


# 计算所有节点之间的最短路径延迟
def calculate_shortest_path_delay(delay_matrix):
    # 初始化最短路径延迟矩阵为延迟矩阵的副本
    distance_matrix = np.copy(delay_matrix)

    # 使用 Floyd-Warshall 算法计算最短路径延迟
    for k in range(nodes):
        for i in range(nodes):
            for j in range(nodes):
                if distance_matrix[i][j] > distance_matrix[i][k] + distance_matrix[k][j]:
                    distance_matrix[i][j] = distance_matrix[i][k] + distance_matrix[k][j]

    return distance_matrix


# 网络无向图连接矩阵
net_delay = np.zeros((nodes, nodes))
net_delay[0][1] = 5
net_delay[0][6] = 10
net_delay[1][2] = 7
net_delay[2][3] = 8
net_delay[2][10] = 10
net_delay[3][5] = 20
net_delay[3][6] = 6
net_delay[3][7] = 10
net_delay[4][5] = 7
net_delay[4][10] = 9
net_delay[5][8] = 4
net_delay[6][7] = 11
net_delay[6][14] = 13
net_delay[7][12] = 9
net_delay[7][14] = 10
net_delay[8][13] = 10
net_delay[9][11] = 8
net_delay[9][13] = 9
net_delay[10][11] = 15
net_delay[12][13] = 6
net_delay[12][14] = 15
net_delay = net_delay + net_delay.T
net_delay = np.where(net_delay == 0, max_time, net_delay)
# np.fill_diagonal(net_delay, 0)
net_delay = calculate_shortest_path_delay(net_delay)
np.fill_diagonal(net_delay, 1)


def add_column_divided(matrix, divider):
    new_column = (matrix[:, 0] * divider).reshape((-1, 1))
    matrix = np.hstack((matrix, new_column))
    return matrix


compute_time = np.array([[20], [15], [28], [6],
                         [30], [17], [13], [10],
                         [30], [17], [13],
                         [10], [10], [100], [30]])
for _ in range(7):
    compute_time = add_column_divided(compute_time, 1 / 3)
    compute_time = add_column_divided(compute_time, 1 / 2)

environment_max = DRL_Environment(app_fee, cpu_fee, ram_fee, disk_fee, max_fee, services, nodes, max_time, lambda_out,
                                  start_service, access_node, service_resource_occupancy, node_resource_capacity,
                                  instance, service_dependency, net_delay, compute_time)

without_request_environment_max = (
    other_environment.without_request_environment(app_fee, cpu_fee, ram_fee, disk_fee,
                                                  max_fee, services, nodes, max_time, lambda_out,
                                                  start_service, access_node, service_resource_occupancy,
                                                  node_resource_capacity,
                                                  instance, service_dependency, net_delay, compute_time))

without_route_environment_max = (
    other_environment.without_route_environment(app_fee, cpu_fee, ram_fee, disk_fee,
                                                max_fee, services, nodes, max_time, lambda_out,
                                                start_service, access_node, service_resource_occupancy,
                                                node_resource_capacity,
                                                instance, service_dependency, net_delay, compute_time))

only_instance_environment_max = (
    other_environment.only_instance_environment(app_fee, cpu_fee, ram_fee, disk_fee,
                                                max_fee, services, nodes, max_time, lambda_out,
                                                start_service, access_node, service_resource_occupancy,
                                                node_resource_capacity,
                                                instance, service_dependency, net_delay, compute_time))

if __name__ == '__main__':
    state = torch.tensor(
            [
                0.0000, 0.0000, 0.0000, 1.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
                0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
                1.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
                0.0000, 0.0000, 0.0000, 1.0000, 1.0000, 0.0000, 1.0000, 0.0000, 1.0000,
                0.0000, 0.0000, 0.0000, 0.0000, 1.0000, 0.0000, 0.0000, 0.0000, 0.0000,
                0.0000, 0.0000, 0.0000, 1.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
                0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
                1.0000, 0.0000, 1.0000, 0.0000, 0.0000, 0.0000, 0.0000, 1.0000, 0.0000,
                0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 1.0000, 0.0000, 0.0000,
                0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
                0.0000, 0.0000, 0.0000, 1.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
                0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
                1.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
                0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 1.0000, 0.0000, 0.0000,
                0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
                0.0000, 0.0000, 0.0000, 1.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
                0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
                1.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
                0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 1.0000, 0.0000, 0.0000,
                0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
                0.0000, 0.0000, 0.0000, 1.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
                0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 1.0000, 0.0000, 0.0000,
                0.0000, 0.0000, 1.0000, 1.0000, 0.0000, 1.0000, 1.0000, 1.0000, 0.0000,
                0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 1.0000,
                0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
                1.0000, 1.0000, 1.0000, 1.0000, 0.4500
            ]
        )
    environment_max.update_state(state)
    print(environment_max.check_constrains())
    print(environment_max.get_reward())
