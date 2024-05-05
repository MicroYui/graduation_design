import numpy as np

from PPO.scale_min import environment_min as scale_5_5
from new_environment import DRL_Environment


def get_service_importance(environment: DRL_Environment):
    compute_time = environment.compute_time
    for service in range(environment.services):
        upstream_service_list = get_upstream_service_list(environment, service)
        importance = 0


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


# 极小型指标转化为极大型指标的函数
def minTomax(maxx, x):
    x = list(x)  # 将输入的指标数据转换为列表
    ans = [[(maxx - e)] for e in x]  # 计算最大值与每个指标值的差，并将其放入新列表中
    return np.array(ans)  # 将列表转换为numpy数组并返回


if __name__ == '__main__':
    print(get_upstream_service_list(scale_5_5, 2))
