import numpy as np
import torch


def clamp(n, minn, maxn):
    return max(min(maxn, n), minn)


class DRL_Environment(object):
    def __init__(self, app_fee, cpu_fee, ram_fee, disk_fee, max_fee, services, nodes,
                 max_time, request_out, start_service, access_node, service_resource_occupancy,
                 node_resource_capacity, instance, service_dependency, net_delay, compute_time):
        self.app_fee = app_fee
        self.cpu_fee = cpu_fee
        self.ram_fee = ram_fee
        self.disk_fee = disk_fee
        self.max_fee = max_fee
        self.services = services
        self.nodes = nodes
        self.max_time = max_time
        self.start_service = start_service
        self.delta = np.random.rand(len(self.start_service))
        self.request_out = request_out
        self.request_in = self.delta * self.request_out
        self.access_node = access_node
        self.service_resource_occupancy = service_resource_occupancy
        self.node_resource_capacity = node_resource_capacity
        self.instance = instance
        self.service_dependency = service_dependency
        self.net_delay = net_delay
        self.compute_time = compute_time
        self.second = 1000
        self.compute_ability = self.second / np.array(self.compute_time)
        self.request_arrive = np.zeros((self.services, self.nodes))
        self.request_rate = 0.5
        self.network_rate = 1 - self.request_rate
        self.constrains = True

    def cost_constrains(self) -> bool:
        instance_cost = 0
        for node in range(self.nodes):
            cpu, ram, disk = 0, 0, 0
            for service in range(self.services):
                cpu += self.service_resource_occupancy[service, 0] * self.instance[service, node]
                ram += self.service_resource_occupancy[service, 1] * self.instance[service, node]
                disk += self.service_resource_occupancy[service, 2] * self.instance[service, node]
            instance_cost += cpu * self.cpu_fee + ram * self.ram_fee + disk * self.disk_fee
        request_decline = sum(self.request_out) - sum(self.request_in)
        request_cost = request_decline * self.app_fee
        total_cost = instance_cost + request_cost
        return total_cost <= self.max_fee

    def node_capacity_constrains(self) -> bool:
        for node in range(self.nodes):
            cpu, ram, disk = 0, 0, 0
            for service in range(self.services):
                cpu += self.service_resource_occupancy[service, 0] * self.instance[service, node]
                ram += self.service_resource_occupancy[service, 1] * self.instance[service, node]
                disk += self.service_resource_occupancy[service, 2] * self.instance[service, node]
            if cpu > self.node_resource_capacity[node, 0] or ram > self.node_resource_capacity[node, 1] \
                    or disk > self.node_resource_capacity[node, 2]:
                return False
        return True

    def instance_constrains(self) -> bool:
        for service in range(self.services):
            instance_num = self.instance[service, :].sum()
            if instance_num == 0:
                return False
        return True

    def get_upstream_services(self, service: int) -> list:
        upstream_services = []
        for upstream_service in range(self.services):
            if self.service_dependency[upstream_service, service] == 1:
                upstream_services.append(upstream_service)
        return upstream_services

    def get_nodes_of_a_service(self, service: int) -> list:
        node_vector = []
        for node in range(self.nodes):
            if self.instance[service, node] == 1:
                node_vector.append(node)
        return node_vector

    def update_important_rate(self, new_rate):
        self.request_rate = new_rate
        self.network_rate = 1 - new_rate

    def get_compute_and_transmit_time(self, node: int, service: int) -> list:
        compute_vector = []
        transmit_vector = []
        row = self.instance[service]
        for remote_node in range(self.nodes):
            if row[remote_node] == 1:
                execute_time = self.compute_time[service, remote_node]
                transmit_time = self.net_delay[node, remote_node]
                compute_vector.append(execute_time)
                transmit_vector.append(transmit_time)
            else:
                compute_vector.append(self.max_time)
                transmit_vector.append(self.max_time)
        return [compute_vector, transmit_vector]

    def route(self, compute_time_vector: list, transmit_time_vector: list):
        route_vector = []
        importance_rate_vector = []
        total = 0

        for node in range(self.nodes):
            compute_time = compute_time_vector[node]
            transmit_time = transmit_time_vector[node]
            if compute_time != self.max_time:
                importance = (self.request_rate / compute_time) + (self.network_rate / transmit_time)
                importance_rate_vector.append(importance)
                total += importance
            else:
                importance_rate_vector.append(0)

        for node in range(self.nodes):
            compute_time = compute_time_vector[node]
            if compute_time == self.max_time:
                route_vector.append(0)
            else:
                route_vector.append(importance_rate_vector[node] / total)
        return route_vector

    def check_constrains(self) -> bool:
        if not self.cost_constrains() or not self.instance_constrains() or not \
                self.node_capacity_constrains():
            # print("基础约束不满足")
            self.constrains = False
        if self.request_rate == 0 or self.network_rate == 0:
            self.constrains = False
        return self.constrains

    def update_request_arrive_array_matrix(self):
        arrive = np.zeros((self.services, self.nodes))

        for index in range(len(self.start_service)):
            arrive_app = np.zeros((self.services, self.nodes))
            access = self.access_node[index]
            first_service = self.start_service[index]
            compute_and_transmit_time = self.get_compute_and_transmit_time(access, first_service)
            route_vector = self.route(compute_and_transmit_time[0], compute_and_transmit_time[1])
            request_dispatch = np.array(route_vector) * self.request_in[index]
            arrive_app[first_service, :] += request_dispatch

            current_service = first_service
            while np.array_equal(self.service_dependency[current_service, :], np.zeros(self.services)) is False:
                service_vector = self.service_dependency[current_service, :]
                for service_index in range(len(service_vector)):
                    if service_vector[service_index] == 1:
                        next_service = service_index
                        current_service_node_vector = self.get_nodes_of_a_service(current_service)
                        for current_service_node in current_service_node_vector:
                            compute_and_transmit_time = self.get_compute_and_transmit_time(current_service_node,
                                                                                           next_service)
                            route_vector = self.route(compute_and_transmit_time[0], compute_and_transmit_time[1])
                            request_dispatch = np.array(route_vector) * \
                                               arrive_app[current_service, current_service_node]
                            arrive_app[next_service, :] += request_dispatch
                        current_service = next_service
            arrive += arrive_app

        self.request_arrive = arrive

    def get_service_transmit_time(self, service: int):
        mean_transmit_time = 0
        request_number = self.request_arrive[service, :].sum()
        if request_number == 0:
            # print("有服务到达率为0")
            self.constrains = False
            return 0

        for index in range(len(self.start_service)):
            if service == self.start_service[index]:
                access = self.access_node[index]
                for node in range(self.nodes):
                    request = self.request_arrive[service, node]
                    if request != 0.0:
                        mean_transmit_time += request / request_number * self.net_delay[access, node]
                return mean_transmit_time

        upstream_services = self.get_upstream_services(service)
        for upstream_service in upstream_services:
            for upstream_node in self.get_nodes_of_a_service(upstream_service):
                compute_and_transmit_time = self.get_compute_and_transmit_time(upstream_node, service)
                route_vector = self.route(compute_and_transmit_time[0], compute_and_transmit_time[1])
                for node in self.get_nodes_of_a_service(service):
                    mean_transmit_time += self.request_arrive[upstream_service, upstream_node] * \
                                          route_vector[node] * self.net_delay[upstream_node, node] / request_number
        return mean_transmit_time

    def get_service_queue_time(self, service: int):
        request_number = self.request_arrive[service, :].sum()
        mean_queue_time = 0
        for node in range(self.nodes):
            request = self.request_arrive[service, node]
            if request != 0.0:
                if self.compute_ability[service, node] - \
                        self.request_arrive[service, node] > 4:
                    # print("compute_ability:", self.compute_ability[service, node],
                    #       "request_arrive:", self.request_arrive[service, node])
                    mean_queue_time += request / request_number / (
                            self.compute_ability[service, node] -
                            self.request_arrive[service, node])
                else:
                    # print(f"service:{service}排队论约束不满足")
                    # print(f"node:{node}, compute_ability:{self.compute_ability[service, node]}, "
                    #       f"request_arrive:{self.request_arrive[service, node]}")
                    self.constrains = False
                    return 0
        # print("mean_queue_time", mean_queue_time)
        return mean_queue_time * self.second

    def get_app_total_time(self, first_service: int):
        total_time = 0
        current_service = first_service
        while np.array_equal(self.service_dependency[current_service, :],
                             np.zeros(self.services)) is False:
            next_service = np.where(self.service_dependency[current_service, :]
                                    == 1)[0][0]
            total_time += (self.get_service_queue_time(current_service) +
                           self.get_service_transmit_time(current_service))
            # print(f"service{current_service}: ", total_time)
            current_service = next_service
        total_time += (self.get_service_queue_time(current_service) +
                       self.get_service_transmit_time(current_service))
        # print(f"service{current_service}: ", total_time)
        return total_time

    def get_max_total_time(self):
        max_total_time = 0
        for app in self.start_service:
            total_time = self.get_app_total_time(app)
            # print("total_time: ", total_time)
            max_total_time = max(max_total_time, total_time)
        # if max_total_time > 1000:
        #     self.constrains = False
        return max_total_time

    def update_state(self, state):
        self.constrains = True
        pre_state = state.detach().cpu().numpy()
        instance_vector = pre_state[0: self.services * self.nodes].reshape(self.services, self.nodes).astype(int)
        self.instance = instance_vector
        request_vector = pre_state[self.services * self.nodes: self.services * self.nodes + len(self.start_service)]
        self.delta = request_vector
        self.request_in = self.delta * self.request_out
        self.request_rate = pre_state[-1]
        self.update_important_rate(self.request_rate)
        self.update_request_arrive_array_matrix()
        self.get_max_total_time()
        self.check_constrains()

    def reset(self):
        instance_reset = torch.tensor(np.random.randint(2, size=(self.services, self.nodes))).flatten()
        delta_reset = torch.tensor(np.around(np.random.rand(len(self.delta)), decimals=2))
        request_rate_reset = torch.tensor(np.around(np.random.rand(1), decimals=2))
        return torch.cat((instance_reset, delta_reset, request_rate_reset))

    def constrains_reset(self):
        while True:
            instance_reset = torch.tensor(np.random.randint(2, size=(self.services, self.nodes))).flatten()
            delta_reset = torch.tensor(np.around(np.random.rand(len(self.delta)), decimals=2))
            request_rate_reset = torch.tensor(np.around(np.random.rand(1), decimals=2))
            state = torch.cat((instance_reset, delta_reset, request_rate_reset))

            self.update_state(state)
            if self.constrains:
                return state

    def get_reward(self):
        return self.get_max_total_time()

    def step(self, state, action):
        dead = False
        self.update_state(state)
        pre_state_fitness = self.get_reward()
        x = int((action[0] / 2 + 0.5) * self.services) % self.services
        y = int((action[1] / 2 + 0.5) * self.nodes) % self.nodes
        if action[2] > 0:
            state[x * self.nodes + y] = 1
        else:
            state[x * self.nodes + y] = 0
        x = int((action[3] / 2 + 0.5) * len(self.delta)) % len(self.delta)
        if action[4] > 0:
            state[self.services * self.nodes + x] += 0.01
        else:
            state[self.services * self.nodes + x] -= 0.01
        state[self.services * self.nodes + x] = np.clip(state[self.services * self.nodes + x], 0, 1)
        if action[5] > 0:
            state[-1] += 0.01
        else:
            state[-1] -= 0.01
        state[-1] = np.clip(state[-1], 0, 1)
        self.update_state(state)
        state_fitness = self.get_reward()
        reward = pre_state_fitness - state_fitness
        # reward = ori_reward - state_fitness
        reward = clamp(reward, -300, 300)
        # reward = state_fitness
        # reward = clamp(reward, -1000, 1000)
        # print("pre_state_fitness: ", pre_state_fitness, "state_fitness: ", state_fitness, "reward: ", reward)
        # print("state: \n", state)
        if not self.check_constrains():
            reward = -300
            dead = True
        return state, reward, dead

    def heuristic_algorithm_fitness_function(self, state):
        current_state = torch.tensor(state)
        self.update_state(current_state)
        state_fitness = self.get_reward()
        if not self.constrains:
            return self.max_time + self.cost_punishment() + self.capacity_punishment() + self.instance_punishment()
        return state_fitness

    def step_solo(self, state, action):
        dead = False
        self.update_state(state)
        pre_state_fitness = self.get_reward()
        selection = int((action[0] / 2 + 0.5) * 3) % 3
        # 选择进行实例部署操作
        if selection == 0:
            x = int((action[1] / 2 + 0.5) * self.services) % self.services
            y = int((action[2] / 2 + 0.5) * self.nodes) % self.nodes
            if state[x * self.nodes + y] == 1:
                state[x * self.nodes + y] = 0
            else:
                state[x * self.nodes + y] = 1
        # 选择进行流量调控操作
        elif selection == 1:
            x = int((action[1] / 2 + 0.5) * len(self.delta)) % len(self.delta)
            if action[2] > 0:
                state[self.services * self.nodes + x] += 0.01
            else:
                state[self.services * self.nodes + x] -= 0.01
            state[self.services * self.nodes + x] = np.clip(state[self.services * self.nodes + x], 0, 1)
        # 选择进行路由调控操作
        else:
            if action[2] > 0:
                state[-1] += 0.01
            else:
                state[-1] -= 0.01
            state[-1] = np.clip(state[-1], 0, 1)
        self.update_state(state)
        state_fitness = self.get_reward()
        reward = pre_state_fitness - state_fitness
        reward = clamp(reward, -300, 300)
        if not self.check_constrains():
            reward = -300
            dead = True
        return state, reward, dead

    # 用于进行缺少路由的实验
    def simple_route(self, compute_time_vector: list, transmit_time_vector: list):
        route_vector = []
        importance_rate_vector = []
        total = 0

        for node in range(self.nodes):
            compute_time = compute_time_vector[node]
            if compute_time != self.max_time:
                total += 1

        for node in range(self.nodes):
            compute_time = compute_time_vector[node]
            if compute_time == self.max_time:
                route_vector.append(0)
            else:
                route_vector.append(1 / total)
        return route_vector

    def my_heuristic_algorithm_fitness_function(self, state):
        # 将state实例部署部分都变成0和1，后面部分都局限在0和1
        current_state = torch.tensor(state)
        current_state[:self.services * self.nodes] = torch.round(current_state[:self.services * self.nodes])
        current_state[self.services * self.nodes:] = torch.clamp(current_state[self.services * self.nodes:], 0, 1)
        self.update_state(current_state)
        state_fitness = self.get_reward()
        if not self.constrains:
            return self.max_time
        return state_fitness

    def cost_punishment(self):
        instance_cost = 0
        for node in range(self.nodes):
            cpu, ram, disk = 0, 0, 0
            for service in range(self.services):
                cpu += self.service_resource_occupancy[service, 0] * self.instance[service, node]
                ram += self.service_resource_occupancy[service, 1] * self.instance[service, node]
                disk += self.service_resource_occupancy[service, 2] * self.instance[service, node]
            instance_cost += cpu * self.cpu_fee + ram * self.ram_fee + disk * self.disk_fee
        request_decline = sum(self.request_out) - sum(self.request_in)
        request_cost = request_decline * self.app_fee
        total_cost = instance_cost + request_cost
        if total_cost <= self.max_fee:
            return 0
        else:
            return total_cost - self.max_fee

    def instance_punishment(self):
        number = 0
        for service in range(self.services):
            instance_num = self.instance[service, :].sum()
            if instance_num == 0:
                number += 1
        return number * self.max_time

    def capacity_punishment(self):
        total = 0
        for node in range(self.nodes):
            cpu, ram, disk = 0, 0, 0
            for service in range(self.services):
                cpu += self.service_resource_occupancy[service, 0] * self.instance[service, node]
                ram += self.service_resource_occupancy[service, 1] * self.instance[service, node]
                disk += self.service_resource_occupancy[service, 2] * self.instance[service, node]
            if cpu > self.node_resource_capacity[node, 0]:
                total += (cpu - self.node_resource_capacity[node, 0]) * self.cpu_fee * 100
            if ram > self.node_resource_capacity[node, 1]:
                total += (ram - self.node_resource_capacity[node, 1]) * self.ram_fee * 100
            if disk > self.node_resource_capacity[node, 2]:
                total += (disk - self.node_resource_capacity[node, 2]) * self.disk_fee * 100
        return total
