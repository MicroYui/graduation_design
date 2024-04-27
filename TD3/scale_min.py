import numpy as np
from new_environment import DRL_Environment

app_fee = 200
cpu_fee = 1
ram_fee = 0.1
disk_fee = 0.01
max_fee = 8000
app_1_request = 50
app_2_request = 30
rows = 5
cols = 5
max_time = 999999
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

environment_min = DRL_Environment(app_fee, cpu_fee, ram_fee, disk_fee, max_fee, rows, cols, max_time, lambda_out,
                                  start_service, access_node, service_resource_occupancy, node_resource_capacity,
                                  instance, service_dependency, net_delay, compute_time)

environment_min2 = DRL_Environment(app_fee, cpu_fee, ram_fee, disk_fee, max_fee, rows, cols, max_time, lambda_out,
                                   start_service, access_node, service_resource_occupancy, node_resource_capacity,
                                   instance, service_dependency, net_delay, compute_time)

environment_min3 = DRL_Environment(app_fee, cpu_fee, ram_fee, disk_fee, max_fee, rows, cols, max_time, lambda_out,
                                   start_service, access_node, service_resource_occupancy, node_resource_capacity,
                                   instance, service_dependency, net_delay, compute_time)
