import numpy as np
import torch

from new_environment import DRL_Environment

app_fee = 1000
cpu_fee = 1
ram_fee = 0.001
disk_fee = 0.001
max_fee = 8000
app_1_request = 50
app_2_request = 30
services = 7
nodes = 7
max_time = 999
start_service = [0, 5]
lambda_out = [app_1_request, app_2_request]
access_node = [0, 3]
service_resource_occupancy = np.array([
    [2.0, 512, 50],
    [1.0, 256, 40],
    [1.0, 380, 120],
    [0.5, 128, 20],
    [1.5, 420, 70],
    [2.0, 512, 50],
    [1.0, 256, 40],
])
node_resource_capacity = np.array([
    [20, 4096, 2048],
    [10, 2048, 2048],
    [7, 2048, 2048],
    [4, 1024, 2048],
    [12, 2048, 2048],
    [16, 2048, 2048],
    [10, 2048, 2048],
])
instance = np.random.randint(2, size=(services, nodes))
# 服务依赖关系
# 0 → 1 → 2 → 3 → 4
#         ↑
# 5 → 6 → ↑
service_dependency = np.zeros((services, services))
service_dependency[0][1] = 1
service_dependency[1][2] = 1
service_dependency[2][3] = 1
service_dependency[3][4] = 1
service_dependency[5][6] = 1
service_dependency[6][2] = 1


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
net_delay[0][1] = 20
net_delay[0][3] = 5
net_delay[0][5] = 10
net_delay[1][2] = 7
net_delay[1][3] = 15
net_delay[2][4] = 5
net_delay[3][4] = 10
net_delay[3][5] = 15
net_delay[3][6] = 7
net_delay[4][6] = 6
net_delay = net_delay + net_delay.T
net_delay = np.where(net_delay == 0, max_time, net_delay)
# np.fill_diagonal(net_delay, 0)
net_delay = calculate_shortest_path_delay(net_delay)
np.fill_diagonal(net_delay, 1)


def add_column_divided(matrix, divider):
    new_column = (matrix[:, 0] * divider).reshape((-1, 1))
    matrix = np.hstack((matrix, new_column))
    return matrix


compute_time = np.array([[20], [15], [28], [6], [30], [17], [13]])
for _ in range(2):
    compute_time = add_column_divided(compute_time, 1 / 3 * 2)
    compute_time = add_column_divided(compute_time, 1 / 2)
    compute_time = add_column_divided(compute_time, 1 / 3)

environment_7services_7nodes = DRL_Environment(app_fee, cpu_fee, ram_fee, disk_fee, max_fee, services, nodes, max_time,
                                               lambda_out, start_service, access_node, service_resource_occupancy,
                                               node_resource_capacity, instance,
                                               service_dependency, net_delay, compute_time)

if __name__ == '__main__':
    state = torch.tensor(
        [1.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 1.0000, 0.0000, 0.0000,
         1.0000, 1.0000, 0.0000, 0.0000, 0.0000, 1.0000, 0.0000, 1.0000, 0.0000,
         1.0000, 1.0000, 0.0000, 1.0000, 0.0000, 1.0000, 0.0000, 0.0000, 1.0000,
         1.0000, 0.0000, 0.0000, 1.0000, 0.0000, 0.0000, 1.0000, 0.0000, 0.0000,
         0.0000, 0.0000, 1.0000, 1.0000, 0.0000, 0.0000, 0.0000, 0.0000, 1.0000,
         0.0000, 1.0000, 1.0000, 1.0000, 0.9000, 1.0000, 0.3600]
    )
    environment_7services_7nodes.update_state(state)
    print(len(state))
    print(environment_7services_7nodes.get_reward())
