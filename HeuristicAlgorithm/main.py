import numpy as np
import pandas as pd
import torch
from ORI_GA import GA as O_GA
from myGA import GA
from myPSO import PSO
from sko.operators.crossover import crossover_2point_bit
import matplotlib.pyplot as plt
from PPO.scale_min import environment_min as environment1
from PPO.scale_7_7 import environment_7services_7nodes as environment2
from PPO.scale_mid import environment_mid as environment3
from PPO.scale_12_12 import environment_12services_12nodes as environment4
from PPO.scale_max import environment_max as environment5


def my_pso(environment, state):
    ub = [1] * (environment.services * environment.nodes + len(environment.start_service) + 1)
    lb = [0] * (environment.services * environment.nodes + len(environment.start_service) + 1)
    precision = [1] * (environment.services * environment.nodes) + [1e-2] * (len(environment.start_service) + 1)
    n_particles = int(50 * len(lb))
    max_iter = int(100)

    # define GA
    pso = PSO(func=environment.heuristic_algorithm_fitness_function,
              n_dim=len(lb),
              pop=n_particles,
              max_iter=max_iter,
              lb=lb,
              ub=ub,
              w=0.8, c1=0.5, c2=0.5,
              services=environment.services,
              nodes=environment.nodes,
              state=state
              )

    x_opt, y_opt = pso.run()
    print('best_x is \n', x_opt, '\nbest_y is\n', y_opt)
    print("best result:", environment.heuristic_algorithm_fitness_function(x_opt))
    Y_history = pd.DataFrame(pso.gbest_y_hist)

    best_x = pd.DataFrame({'x_opt': x_opt})
    best_x.to_csv(f'run_data/PSO_{environment.services}_{environment.nodes}_bestX.csv', index=False, sep=',')
    Y_history.to_csv(f'run_data/PSO_{environment.services}_{environment.nodes}.csv', index=False, sep=',')


def my_ga(environment, state):
    ub = [1] * (environment.services * environment.nodes + len(environment.start_service) + 1)
    lb = [0] * (environment.services * environment.nodes + len(environment.start_service) + 1)
    precision = [1] * (environment.services * environment.nodes) + [1e-2] * (len(environment.start_service) + 1)
    n_particles = int(50 * len(lb))
    max_iter = int(100)

    # define GA
    ga = GA(func=environment.heuristic_algorithm_fitness_function,
            n_dim=len(lb),
            size_pop=n_particles,
            max_iter=max_iter,
            precision=precision,
            lb=lb,
            ub=ub,
            services=environment.services,
            nodes=environment.nodes,
            first_chrom=state,
            )
    ga.register(operator_name='crossover', operator=crossover_2point_bit)

    # iter
    x_opt, y_opt = ga.run(max_iter=max_iter)

    # show result
    Y_history = pd.DataFrame(ga.all_history_Y)
    # fig, ax = plt.subplots(2, 1)
    # ax[0].plot(Y_history.index, Y_history.values, '.', color='red')
    Y_history.min(axis=1).cummin().plot(kind='line')

    name = "GA"
    if state is None:
        name = "ORI_GA"

    best_x = pd.DataFrame({'x_opt': x_opt})
    best_x.to_csv(f'run_data/{name}_{environment.services}_{environment.nodes}_bestX.csv', index=False, sep=',')
    Y_history.to_csv(f'run_data/{name}_{environment.services}_{environment.nodes}.csv', index=False, sep=',')

    print("best_x:\n", ga.best_x, "\nbest_y:", ga.best_y)
    print(torch.tensor(ga.best_x))
    # plt.xlabel('iter')
    # plt.ylabel('fitness')
    # plt.title(f'GA_{environment.services}services_{environment.nodes}nodes')
    # plt.savefig(f"GA_{environment.services}services_{environment.nodes}nodes")
    # plt.show()


def ori_ga(environment):
    ub = [1] * (environment.services * environment.nodes + len(environment.start_service) + 1)
    lb = [0] * (environment.services * environment.nodes + len(environment.start_service) + 1)
    precision = [1] * (environment.services * environment.nodes) + [1e-2] * (len(environment.start_service) + 1)
    n_particles = int(50 * len(lb))
    max_iter = int(100)

    # define GA
    ga = O_GA(func=environment.heuristic_algorithm_fitness_function,
              n_dim=len(lb),
              size_pop=n_particles,
              max_iter=max_iter,
              precision=precision,
              lb=lb,
              ub=ub,
              services=environment.services,
              nodes=environment.nodes,
              )
    ga.register(operator_name='crossover', operator=crossover_2point_bit)

    # iter
    x_opt, y_opt = ga.run(max_iter=max_iter)

    # show result
    Y_history = pd.DataFrame(ga.all_history_Y)
    # fig, ax = plt.subplots(2, 1)
    # ax[0].plot(Y_history.index, Y_history.values, '.', color='red')
    Y_history.min(axis=1).cummin().plot(kind='line')

    best_x = pd.DataFrame({'x_opt': x_opt})
    best_x.to_csv(f'run_data/ORI_GA_{environment.services}_{environment.nodes}_bestX.csv', index=False, sep=',')
    Y_history.to_csv(f'run_data/ORI_GA_{environment.services}_{environment.nodes}.csv', index=False, sep=',')

    print("best_x:\n", ga.best_x, "\nbest_y:", ga.best_y)
    print(torch.tensor(ga.best_x))


state1 = torch.tensor([0.0000, 0.0000, 1.0000, 1.0000, 0.0000, 1.0000, 1.0000, 0.0000, 1.0000,
                       0.0000, 1.0000, 1.0000, 1.0000, 0.0000, 0.0000, 0.0000, 0.0000, 1.0000,
                       1.0000, 1.0000, 0.0000, 1.0000, 0.0000, 0.0000, 1.0000, 0.9600, 0.9500,
                       0.7000])

state2 = torch.tensor(
    [1.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 1.0000, 0.0000, 0.0000,
     1.0000, 1.0000, 0.0000, 0.0000, 0.0000, 1.0000, 0.0000, 1.0000, 0.0000,
     1.0000, 1.0000, 0.0000, 1.0000, 0.0000, 1.0000, 0.0000, 0.0000, 1.0000,
     1.0000, 0.0000, 0.0000, 1.0000, 0.0000, 0.0000, 1.0000, 0.0000, 0.0000,
     0.0000, 0.0000, 1.0000, 1.0000, 0.0000, 0.0000, 0.0000, 0.0000, 1.0000,
     0.0000, 1.0000, 1.0000, 1.0000, 0.9000, 1.0000, 0.3600]
)

state3 = torch.tensor(
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
     1.0000, 0.9500, 0.9500, 0.9500, 0.6700]
)

state4 = torch.tensor(
    [0.0000, 0.0000, 0.0000, 1.0000, 1.0000, 0.0000, 1.0000, 0.0000, 1.0000, 1.0000, 1.0000, 1.0000,
     1.0000, 0.0000, 0.0000, 1.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 1.0000, 1.0000, 0.0000,
     0.0000, 0.0000, 0.0000, 1.0000, 0.0000, 1.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
     1.0000, 0.0000, 0.0000, 1.0000, 1.0000, 1.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 1.0000,
     0.0000, 1.0000, 0.0000, 0.0000, 0.0000, 1.0000, 0.0000, 1.0000, 0.0000, 0.0000, 0.0000, 0.0000,
     1.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 1.0000, 0.0000, 1.0000, 1.0000, 0.0000,
     0.0000, 1.0000, 1.0000, 0.0000, 0.0000, 0.0000, 1.0000, 0.0000, 0.0000, 0.0000, 0.0000, 1.0000,
     1.0000, 1.0000, 0.0000, 0.0000, 1.0000, 1.0000, 1.0000, 1.0000, 0.0000, 0.0000, 0.0000, 1.0000,
     1.0000, 0.0000, 0.0000, 1.0000, 1.0000, 0.0000, 1.0000, 0.0000, 1.0000, 1.0000, 0.0000, 0.0000,
     0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 1.0000, 0.0000, 1.0000, 0.0000, 0.0000,
     0.0000, 1.0000, 0.0000, 1.0000, 1.0000, 1.0000, 1.0000, 0.0000, 0.0000, 0.0000, 1.0000, 0.0000,
     0.0000, 1.0000, 1.0000, 1.0000, 1.0000, 0.0000, 0.0000, 1.0000, 0.0000, 0.0000, 0.0000, 0.0000,
     1.0000, 1.0000, 1.0000, 0.6200]
)

state5 = torch.tensor(
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

if __name__ == '__main__':
    environment_list = [environment1, environment2, environment3, environment4, environment5]
    state_list = [state1, state2, state3, state4, state5]
    for index in range(len(environment_list)):
        environment = environment_list[index]
        # state = state_list[index]
        ori_ga(environment)
        # my_pso(environment, state)
        # my_ga(environment, None)
