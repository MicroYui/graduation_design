import numpy as np

# 行数rows为微服务数
rows = 5
# 列数cols为节点数
cols = 5
# 不可达时间
max_time = 99999

# 实例部署情况，行表示微服务，列表示节点
instance = np.random.randint(2, size=(rows, cols))

# 服务依赖关系，矩阵为行依赖列
# 0 → 1 → 2
#     ↑
# 3 → 4
service_dependency = np.array([
    [0, 1, 0, 0, 0],
    [0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1],
    [0, 1, 0, 0, 0]
])

# 节点网络延迟矩阵(对称矩阵)
# 0 --10-- 1, 0--20--3
# 1 --15--2
# 2 --5-- 3
# 3 --30-- 4
net_delay = np.array([
    [0, 10, 25, 20, 50],
    [10, 0, 15, 20, 50],
    [25, 15, 0, 5, 35],
    [20, 20, 5, 0, 30],
    [50, 50, 35, 30, 0]
])

# 节点处理微服务时间矩阵
# 行表示不同的微服务，列表示不同节点
# 处理能力 4 > 3 > 2 > 1 > 0
compute_time = np.array([
    [12, 10, 8, 6, 4],
    [28, 28 / 6 * 5, 28 / 3 * 2, 28 / 2, 28 / 3],
    [17, 17 / 6 * 5, 17 / 3 * 2, 17 / 2, 17 / 3],
    [9, 9 / 6 * 5, 9 / 3 * 2, 9 / 2, 9 / 3],
    [78, 78 / 6 * 5, 78 / 3 * 2, 78 / 2, 78 / 3]
])


# 获取当前节点到所有目标微服务所在节点的传输时间(假设不可到达的传输时间为99999ms)
#                                   与处理能力(假设不可到达的处理时间为99999ms)
def get_compute_and_transmit_time(node: int, service: int):
    compute_vector = []
    transmit_vector = []
    row = instance[service]
    # 遍历所有节点
    for remote_node in range(len(row)):
        # 在此节点部署了实例
        if row[remote_node] == 1:
            # 查找service在node上的执行时间
            execute_time = compute_time[service, remote_node]
            transmit_time = net_delay[node, remote_node]
            compute_vector.append(execute_time)
            transmit_vector.append(transmit_time)
        # 没部署实例
        else:
            compute_vector.append(max_time)
            transmit_vector.append(max_time)
    return [compute_vector, transmit_vector]


# 请求路由函数
# 先简化为所有节点同样概率发送
def route(compute_vector: list, transmit_vector: list):
    route_vector = []
    count = 0
    # 计算所有可达的节点数
    for compute in compute_vector:
        if compute != max_time:
            count += 1
    # 给所有节点附上概率
    for compute in compute_vector:
        if compute == max_time:
            route_vector.append(0)
        else:
            route_vector.append(1 / count)
    return route_vector


# 获取一个服务部署的所有节点的列表
def get_nodes_of_a_service(service: int):
    node_vector = []
    for node in range(len(instance[service])):
        if instance[service, node] == 1:
            node_vector.append(node)
    return node_vector


# 当前节点当前服务被请求的总数
def get_request_number_by_node_and_service(node, service) -> int:
    # 找到所有上游微服务
    services = []
    count = 0
    for row in service_dependency:
        if row[service] == 1:
            services.append(count)
        count += 1

    request_number = 0

    # 遍历所有上游微服务，获取所有的请求数
    for upstream_service in services:
        upstream_nodes = get_nodes_of_a_service(upstream_service)
        # 遍历本服务部署的所有节点
        for upstream_node in upstream_nodes:
            # 获取此服务此节点的服务速率
            compute_ability = 1 / compute_time[upstream_service, upstream_node]
            # 根据路由函数得到向目标节点转发的速率
            compute_and_transmit_time = get_compute_and_transmit_time(upstream_node, service)
            route_vector = route(compute_and_transmit_time[0], compute_and_transmit_time[1])
            request_number += compute_ability * route_vector[service]

    return request_number


if __name__ == '__main__':
    print(instance)
    print(get_nodes_of_a_service(2))
