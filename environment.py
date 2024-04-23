import numpy as np
import torch


# 计算约束代价
def compute_constrains(value):
    if value < 0:
        return 0
    else:
        return 0.05 * value


class Environment(object):
    def __init__(self, app_fee, cpu_fee, ram_fee, disk_fee, max_fee, rows, cols,
                 max_time, lambda_out, start_service, access_node, service_resource_occupancy,
                 node_resource_capacity, instance, service_dependency, net_delay, compute_time):
        # 应用耗费成本
        self.app_fee = app_fee
        # cpu | ram | disk 单价
        self.cpu_fee = cpu_fee
        self.ram_fee = ram_fee
        self.disk_fee = disk_fee
        # 可接受的最大成本
        self.max_fee = max_fee
        # 行数rows为微服务数
        self.rows = rows
        # 列数cols为节点数
        self.cols = cols
        # 不可达时间
        self.max_time = max_time
        # 应用对应的微服务
        self.start_service = start_service
        # 限流向量
        self.delta = np.random.rand(len(start_service))
        # 外部流量向量
        self.lambda_out = lambda_out
        # 实际进入流量
        self.lambda_in = self.delta * lambda_out
        # 接入点所在的节点
        self.access_node = access_node
        # 微服务占用资源情况，行表示微服务，列表示资源 cpu | ram | disk
        self.service_resource_occupancy = service_resource_occupancy
        # 节点资源容量情况，行表示节点，列表示资源 cpu | ram | disk
        self.node_resource_capacity = node_resource_capacity
        # 实例部署情况，行表示微服务，列表示节点
        self.instance = instance
        # 服务依赖关系，矩阵为行依赖列
        self.service_dependency = service_dependency
        # 节点网络延迟矩阵(对称矩阵)
        self.net_delay = net_delay
        # 节点处理微服务时间矩阵
        self.compute_time = compute_time
        self.second = 1000
        # 节点计算能力矩阵
        self.compute_ability = self.second / np.array(self.compute_time)
        # 请求到达率矩阵，即每个微服务在每个节点上的服务到达率
        self.request_arrive = np.zeros((self.rows, self.cols))
        # 请求重要因子
        self.request_rate = 0.5
        self.network_rate = 1 - self.request_rate

    # 计算一条链路的总花费时间
    def get_app_total_time(self, first_service: int):
        total_time = 0
        current_service = first_service
        while np.array_equal(self.service_dependency[current_service, :], np.zeros(self.rows)) is False:
            next_service = np.where(self.service_dependency[current_service, :] == 1)[0][0]
            total_time += (
                    self.get_service_queue_time(current_service) + self.get_service_transmit_time(current_service))
            # print("service:", current_service, "queue_time:", self.get_service_queue_time(current_service),
            #       "transmit_time:", self.get_service_transmit_time(current_service))
            print(f"service{current_service}: ", total_time)
            current_service = next_service
        total_time += (self.get_service_queue_time(current_service) + self.get_service_transmit_time(current_service))
        # print("service:", current_service, "queue_time:", self.get_service_queue_time(current_service),
        #       "transmit_time:", self.get_service_transmit_time(current_service))
        return total_time

    # 计算所有链路中最大时间
    def get_max_total_time(self):
        max_total_time = 0
        # if not cost_constrains() or not instance_constrains() or not node_capacity_constrains():
        #     return max_time
        for app in self.start_service:
            total_time = self.get_app_total_time(app)
            print("total_time: ", total_time)
            max_total_time = max(max_total_time, total_time)
        return max_total_time

    def update_request_arrive_array_matrix(self):
        # 将矩阵清空
        arrive = np.zeros((self.rows, self.cols))

        # 先处理开始的微服务(无上游微服务)
        for index in range(len(self.start_service)):
            # 初始化每个app对应的矩阵
            arrive_app = np.zeros((self.rows, self.cols))
            # 获取接入点所在的节点
            access = self.access_node[index]
            # 获取app对应的第一个微服务
            first_service = self.start_service[index]
            # 获取当前节点与目标微服务所在节点的传输时间与处理能力
            compute_and_transmit_time = self.get_compute_and_transmit_time(access, first_service)
            # 获取转发概率向量
            route_vector = self.route(compute_and_transmit_time[0], compute_and_transmit_time[1])
            request_dispatch = np.array(route_vector) * self.lambda_in[index]
            # 将请求到达率写入矩阵
            arrive_app[first_service, :] += request_dispatch

            # 处理此app链路的所有微服务带来的请求
            current_service = first_service
            # 如果有服务依赖就继续往下找
            while np.array_equal(self.service_dependency[current_service, :], np.zeros(self.rows)) is False:
                service_vector = self.service_dependency[current_service, :]
                for service_index in range(len(service_vector)):
                    # 找到下游微服务
                    if service_vector[service_index] == 1:
                        next_service = service_index
                        current_service_node_vector = self.get_nodes_of_a_service(current_service)
                        # 遍历当前服务所有节点
                        for current_service_node in current_service_node_vector:
                            compute_and_transmit_time = self.get_compute_and_transmit_time(current_service_node,
                                                                                           next_service)
                            route_vector = self.route(compute_and_transmit_time[0], compute_and_transmit_time[1])
                            request_dispatch = np.array(route_vector) * arrive_app[
                                current_service, current_service_node]
                            arrive_app[next_service, :] += request_dispatch
                        current_service = next_service
            arrive += arrive_app

        return arrive

    # 计算一个微服务的加权平均排队时间
    def get_service_queue_time(self, service: int):
        request_number = self.request_arrive[service, :].sum()
        mean_queue_time = 0
        for node in range(len(self.request_arrive[service, :])):
            request = self.request_arrive[service, node]
            if request != 0.0:
                # 如果服务到达率 > 服务完成率，则排队时间无限长
                if self.compute_ability[service, node] > self.request_arrive[service, node]:
                    mean_queue_time += request / request_number / (
                            self.compute_ability[service, node] - self.request_arrive[service, node])
                else:
                    mean_queue_time = self.max_time
        return mean_queue_time * 1000

    # 计算一个服务是否不满足排队论约束
    def service_queue_constrains(self, service: int) -> bool:
        for node in range(len(self.request_arrive[service, :])):
            request = self.request_arrive[service, node]
            if request != 0.0:
                # 如果服务到达率 > 服务完成率，则排队时间无限长
                if self.compute_ability[service, node] <= self.request_arrive[service, node]:
                    return False
        return True

    # 计算是否有不满足排队论约束
    def queue_constrains(self) -> bool:
        for app in self.start_service:
            # print("app: ", app)
            current_service = app
            while np.array_equal(self.service_dependency[current_service, :], np.zeros(self.rows)) is False:
                next_service = np.where(self.service_dependency[current_service, :] == 1)[0][0]
                # print("service:", current_service, "queue_time:", self.get_service_queue_time(current_service))
                if self.get_service_queue_time(current_service) >= self.max_time * self.second:
                    return False
                current_service = next_service
            # print("service:", current_service, "queue_time:", self.get_service_queue_time(current_service))
            if self.get_service_queue_time(current_service) >= self.max_time * self.second:
                return False
        return True

    # 计算一个微服务的加权传输时间
    def get_service_transmit_time(self, service: int):
        mean_transmit_time = 0
        request_number = self.request_arrive[service, :].sum()
        if request_number == 0:
            return 0

        # 如果是起始微服务，无上游微服务
        for index in range(len(self.start_service)):
            if service == self.start_service[index]:
                # 拿到接入点节点
                access = self.access_node[index]
                for node in range(len(self.request_arrive[service, :])):
                    request = self.request_arrive[service, node]
                    if request != 0.0:
                        mean_transmit_time += request / request_number * self.net_delay[access, node]
                return mean_transmit_time

        # 如果不是起始微服务，需要计算所有上游微服务
        upstream_services = self.get_upstream_services_of_a_service(service)
        for upstream_service in upstream_services:
            # 遍历上游微服务的所有节点
            for upstream_node in self.get_nodes_of_a_service(upstream_service):
                compute_and_transmit_time = self.get_compute_and_transmit_time(upstream_node, service)
                route_vector = self.route(compute_and_transmit_time[0], compute_and_transmit_time[1])
                # 遍历当前微服务的所有节点
                for node in self.get_nodes_of_a_service(service):
                    mean_transmit_time += self.request_arrive[upstream_service, upstream_node] * \
                                          route_vector[node] * self.net_delay[upstream_node, node] / request_number
        return mean_transmit_time

    # 获取当前节点到所有目标微服务所在节点的传输时间(假设不可到达的传输时间为99999ms)
    #                                   与处理能力(假设不可到达的处理时间为99999ms)
    def get_compute_and_transmit_time(self, node: int, service: int):
        compute_vector = []
        transmit_vector = []
        row = self.instance[service]
        # 遍历所有节点
        for remote_node in range(len(row)):
            # 在此节点部署了实例
            if row[remote_node] == 1:
                # 查找service在node上的执行时间
                execute_time = self.compute_time[service, remote_node]
                transmit_time = self.net_delay[node, remote_node]
                compute_vector.append(execute_time)
                transmit_vector.append(transmit_time)
            # 没部署实例
            else:
                compute_vector.append(self.max_time)
                transmit_vector.append(self.max_time)
        return [compute_vector, transmit_vector]

    # 更新重要因子
    def update_important_rate(self, new_rate):
        self.request_rate = new_rate
        self.network_rate = 1 - new_rate

    # 请求路由函数
    def route(self, compute_vector: list, transmit_vector: list):
        route_vector = []
        importance_rate_vector = []
        total = 0
        # 计算所有可达的节点数
        for index in range(len(compute_vector)):
            compute = compute_vector[index]
            transmit = transmit_vector[index]
            if compute != self.max_time:
                importance = self.request_rate * compute + self.network_rate * transmit
                importance_rate_vector.append(importance)
                total += importance
            else:
                importance_rate_vector.append(0)
        # 给所有节点附上概率
        for index in range(len(compute_vector)):
            compute = compute_vector[index]
            if compute == self.max_time:
                route_vector.append(0)
            else:
                if total == 0:
                    route_vector.append(float(1))
                else:
                    route_vector.append(importance_rate_vector[index] / total)
        return route_vector

    # 获取一个服务部署的所有节点的列表
    def get_nodes_of_a_service(self, service: int) -> list:
        node_vector = []
        for node in range(len(self.instance[service])):
            if self.instance[service, node] == 1:
                node_vector.append(node)
        return node_vector

    # 获取一个微服务上游微服务的列表
    def get_upstream_services_of_a_service(self, service: int) -> list:
        upstream_services = []
        for upstream_service in range(len(self.service_dependency)):
            if self.service_dependency[upstream_service, service] == 1:
                upstream_services.append(upstream_service)
        return upstream_services

    # 检测实例部署约束，即每个微服务至少部署一个实例约束
    def instance_constrains(self) -> bool:
        for service in range(len(self.instance)):
            instance_num = self.instance[service, :].sum()
            if instance_num == 0:
                return False
        return True

    # 实例部署资源约束，即每个节点部署的总资源不能超过本身容量
    def node_capacity_constrains(self) -> bool:
        # 遍历每个节点
        for node in range(self.cols):
            # 计算所有微服务的占用
            cpu, ram, disk = 0, 0, 0
            for service in range(self.rows):
                cpu += self.service_resource_occupancy[service, 0] * self.instance[service, node]
                ram += self.service_resource_occupancy[service, 1] * self.instance[service, node]
                disk += self.service_resource_occupancy[service, 2] * self.instance[service, node]
            if cpu > self.node_resource_capacity[node, 0] or ram > self.node_resource_capacity[node, 1] \
                    or disk > self.node_resource_capacity[node, 2]:
                return False
        return True

    # 更新进入的流量
    def update_lambda_in(self):
        self.lambda_in = self.delta * self.lambda_out

    # 成本约束，即所有治理手段的成本呢不能超过预定标准
    def cost_constrains(self) -> bool:
        # 计算实例部署成本
        instance_cost = 0
        for node in range(self.cols):
            cpu, ram, disk = 0, 0, 0
            for service in range(self.rows):
                cpu += self.service_resource_occupancy[service, 0] * self.instance[service, node]
                ram += self.service_resource_occupancy[service, 1] * self.instance[service, node]
                disk += self.service_resource_occupancy[service, 2] * self.instance[service, node]
            instance_cost += cpu * self.cpu_fee + ram * self.ram_fee + disk * self.disk_fee

        # 计算流量调控成本
        request_decline = sum(self.lambda_out) - sum(self.lambda_in)
        request_cost = request_decline * self.app_fee

        total_cost = instance_cost + request_cost
        # print("instance_cost: ", instance_cost, "request_cost: ", request_cost, "total_cost: ", total_cost)
        # return compute_constrains(total_cost - self.max_fee)
        return total_cost <= self.max_fee

    # reward函数
    def get_reward(self, state):
        instance_vector = state[0: self.rows * self.cols]
        region = instance_vector.detach().numpy().reshape(self.rows, self.cols)
        instance_punishment = 0.0
        # for row in region:
        #     row_sum = row.sum()
        #     if row_sum < 2.5:
        #         instance_punishment += 10000 * (2.5 - row_sum)
        #     else:
        #         instance_punishment += 0.0
        instance_vector = (instance_vector * 2).to(torch.int).detach().numpy().reshape(self.rows, self.cols)
        for row in instance_vector:
            row_sum = row.sum()
            if row_sum < 1:
                instance_punishment += 100 * (1 - row_sum)
            else:
                instance_punishment += 0.0
        self.instance = instance_vector
        request_vector = state[self.rows * self.cols: self.rows * self.cols + len(self.start_service)].detach().numpy()
        self.delta = request_vector
        self.update_lambda_in()
        route_vector = state[-1].flatten().detach().numpy()
        self.request_rate = route_vector[0]
        self.update_important_rate(self.request_rate)
        # print("instance: ", instance)
        # print("delta: ", delta)
        # print("request_rate: ", request_rate)
        self.request_arrive = self.update_request_arrive_array_matrix()
        # print("max_total_time: ", get_max_total_time())
        reward = self.get_max_total_time() + instance_punishment + self.cost_constrains()
        # print("constraint: ", instance_constrains(), cost_constrains(), node_capacity_constrains())

        return reward

    def heuristic_algorithm_fitness_function(self, state):
        instance_vector = state[0: self.rows * self.cols].reshape(self.rows, self.cols).astype(int)
        # instance_punishment = 0.0
        # for row in instance_vector:
        #     row_sum = row.sum()
        #     if row_sum < 1:
        #         instance_punishment += 100 * (1 - row_sum)
        #     else:
        #         instance_punishment += 0.0
        self.instance = instance_vector
        request_vector = state[self.rows * self.cols: self.rows * self.cols + len(self.start_service)]
        self.delta = request_vector
        self.update_lambda_in()
        route_vector = state[-1]
        self.request_rate = route_vector
        self.update_important_rate(self.request_rate)
        self.request_arrive = self.update_request_arrive_array_matrix()
        # node_capacity_punishment = 0.0
        # if not self.node_capacity_constrains():
        #     node_capacity_punishment = 500
        # reward = self.get_max_total_time() + instance_punishment + self.cost_constrains() + node_capacity_punishment
        reward = self.get_max_total_time()
        # if not self.cost_constrains() or not self.instance_constrains() \
        #         or not self.node_capacity_constrains() or not self.queue_constrains():
        #     print("不满足约束")

        return reward

    def reset(self):
        instance_reset = torch.tensor(np.random.randint(2, size=(self.rows, self.cols))).flatten()
        # 版本1：使用精度非常高的随机数
        # delta_reset = torch.tensor(np.random.rand(len(delta)))
        # request_rate_reset = torch.tensor(np.random.rand(1))
        # 版本2：使用指定精度的随机数
        delta_reset = torch.tensor(np.around(np.random.rand(len(self.delta)), decimals=2))
        request_rate_reset = torch.tensor(np.around(np.random.rand(1), decimals=2))

        return torch.cat((instance_reset, delta_reset, request_rate_reset))

    def step(self, state, action):
        dead = False
        reward = 0
        pre_state = state.detach().cpu().numpy()
        pre_state_fitness = self.heuristic_algorithm_fitness_function(pre_state)
        # pre_constrains = not self.cost_constrains() or not self.instance_constrains() or not \
        #     self.node_capacity_constrains() or not self.queue_constrains()
        # action = action.detach().cpu().numpy()
        x = int((action[0] / 2 + 0.5) * self.rows) % self.rows
        y = int((action[1] / 2 + 0.5) * self.cols) % self.cols
        if action[2] > 0:
            pre_state[x * self.cols + y] = 1
        else:
            pre_state[x * self.cols + y] = 0
        # 版本1：action直接生成对应的数字
        # state[rows * cols:] = action[3:]/2 + 0.5
        # 版本2：action生成对应位置是否增减，精度为0.01
        x = int((action[3] / 2 + 0.5) * len(self.delta)) % len(self.delta)
        if action[4] > 0:
            pre_state[self.rows * self.cols + x] += 0.01
        else:
            pre_state[self.rows * self.cols + x] -= 0.01
        pre_state[self.rows * self.cols + x] = np.clip(pre_state[self.rows * self.cols + x], 0, 1)
        if action[5] > 0:
            pre_state[-1] += 0.01
        else:
            pre_state[-1] -= 0.01
        pre_state[-1] = np.clip(pre_state[-1], 0, 1)
        state_fitness = self.heuristic_algorithm_fitness_function(pre_state)
        # constrains = not self.cost_constrains() or not self.instance_constrains() or not \
        #     self.node_capacity_constrains() or not self.queue_constrains()
        # if constrains:
        #     reward = -1
        # elif pre_constrains:
        #     reward = 0
        if not self.cost_constrains() or not self.instance_constrains() or not \
                self.node_capacity_constrains() or not self.queue_constrains():
            # print("不满足约束")
            reward = 0
            dead = True
        else:
            reward = pre_state_fitness - state_fitness
        # print("pre_state_fitness: ", pre_state_fitness, "state_fitness: ", state_fitness, "reward: ", reward)
        # reward = self.heuristic_algorithm_fitness_function(state)
        pre_state = torch.tensor(pre_state)
        return pre_state, reward, dead

    # 解析action
    def show_action(self, action):
        show_action = [int((action[0] / 2 + 0.5) * self.rows) % self.rows,
                       int((action[1] / 2 + 0.5) * self.cols) % self.cols]
        if action[2] > 0:
            show_action.append(1)
        else:
            show_action.append(0)
        show_action.append(int((action[3] / 2 + 0.5) * len(self.delta)) % len(self.delta))
        show_action.append(action[4])
        show_action.append(action[5])
        print(show_action)

    def update_state(self, state):
        pre_state = state.detach().cpu().numpy()
        instance_vector = pre_state[0: self.rows * self.cols].reshape(self.rows, self.cols).astype(int)
        self.instance = instance_vector
        request_vector = pre_state[self.rows * self.cols: self.rows * self.cols + len(self.start_service)]
        self.delta = request_vector
        self.update_lambda_in()
        route_vector = pre_state[-1]
        self.request_rate = route_vector
        self.update_important_rate(self.request_rate)
        self.request_arrive = self.update_request_arrive_array_matrix()
