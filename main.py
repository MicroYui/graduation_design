import numpy as np

# 应用请求次数
app_1_request = 30
app_2_request = 10
# 行数rows为微服务数
rows = 5
# 列数cols为节点数
cols = 5
# 不可达时间
max_time = 99999
# 限流向量
delta = np.random.rand(2)
# 外部流量向量
lambda_out = [app_1_request, app_2_request]
# 实际进入流量
lambda_in = delta * lambda_out
# 应用对应的微服务
start_service = [0, 3]
# 接入点所在的节点
access_node = [0, 3]

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

# 请求到达率矩阵，即每个微服务在每个节点上的服务到达率
# 行表示微服务，列表示节点
request_arrive = np.zeros((rows, cols))


# 根据当前转发函数、流量调控因子和服务依赖情况更新请求达到矩阵
def update_request_arrive_array_matrix():
    # 将矩阵清空
    arrive = np.zeros((rows, cols))

    # 先处理开始的微服务(无上游微服务)
    for index in range(len(start_service)):
        # 初始化每个app对应的矩阵
        arrive_app = np.zeros((rows, cols))
        # 获取接入点所在的节点
        access = access_node[index]
        # 获取app对应的第一个微服务
        first_service = start_service[index]
        # 获取当前节点与目标微服务所在节点的传输时间与处理能力
        compute_and_transmit_time = get_compute_and_transmit_time(access, first_service)
        # 获取转发概率向量
        route_vector = route(compute_and_transmit_time[0], compute_and_transmit_time[1])
        request_dispatch = np.array(route_vector) * lambda_out[index]
        # 将请求到达率写入矩阵
        arrive_app[first_service, :] += request_dispatch

        # 处理此app链路的所有微服务带来的请求
        current_service = first_service
        # 如果有服务依赖就继续往下找
        while np.array_equal(service_dependency[current_service, :], np.zeros(rows)) is False:
            service_vector = service_dependency[current_service, :]
            for service_index in range(len(service_vector)):
                # 找到下游微服务
                if service_vector[service_index] == 1:
                    next_service = service_index
                    current_service_node_vector = get_nodes_of_a_service(current_service)
                    # 遍历当前服务所有节点
                    for current_service_node in current_service_node_vector:
                        compute_and_transmit_time = get_compute_and_transmit_time(current_service_node, next_service)
                        route_vector = route(compute_and_transmit_time[0], compute_and_transmit_time[1])
                        request_dispatch = np.array(route_vector) * arrive_app[current_service, current_service_node]
                        arrive_app[next_service, :] += request_dispatch
                    current_service = next_service
        arrive += arrive_app
    return arrive


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
def get_nodes_of_a_service(service: int) -> list:
    node_vector = []
    for node in range(len(instance[service])):
        if instance[service, node] == 1:
            node_vector.append(node)
    return node_vector


if __name__ == '__main__':
    print(instance)
    # print(get_nodes_of_a_service(2))
    print(delta)
    print(lambda_in)
    print(request_arrive)
    request_arrive = update_request_arrive_array_matrix()
    print(request_arrive)
