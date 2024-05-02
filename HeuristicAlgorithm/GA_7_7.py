import numpy as np
import pandas as pd
import torch
from sko.GA import GA
# from myGA import GA
from sko.operators.crossover import crossover_2point_bit
import matplotlib.pyplot as plt

import main
from TD3.scale_7_7 import environment_7services_7nodes as environment

if __name__ == '__main__':
    # init
    ub = [1] * (environment.services * environment.nodes + len(environment.start_service) + 1)
    lb = [0] * (environment.services * environment.nodes + len(environment.start_service) + 1)
    precision = [1] * (environment.services * environment.nodes) + [1e-2] * (len(environment.start_service) + 1)
    n_particles = int(100 * len(lb))
    max_iter = int(20 * len(lb))

    # define GA
    ga = GA(func=environment.heuristic_algorithm_fitness_function,
            n_dim=len(lb),
            size_pop=n_particles,
            max_iter=max_iter,
            precision=precision,
            lb=lb,
            ub=ub,
            )
    ga.register(operator_name='crossover', operator=crossover_2point_bit)

    # iter
    x_opt, y_opt = ga.run(max_iter=max_iter)

    # show result
    Y_history = pd.DataFrame(ga.all_history_Y)
    # fig, ax = plt.subplots(2, 1)
    # ax[0].plot(Y_history.index, Y_history.values, '.', color='red')
    Y_history.min(axis=1).cummin().plot(kind='line')
    print("best_x:\n", ga.best_x, "\nbest_y:", ga.best_y)
    print(torch.tensor(ga.best_x))
    plt.xlabel('iter')
    plt.ylabel('fitness')
    plt.title('GA_7services_7nodes')
    plt.savefig("GA_7services_7nodes.svg")
    plt.show()

