import numpy as np
import pandas as pd
from sko.GA import GA
from sko.operators.crossover import crossover_2point
import matplotlib.pyplot as plt

import main
from main import heuristic_algorithm_fitness_function

if __name__ == '__main__':
    # init
    ub = [1] * (main.rows * main.cols + len(main.start_service) + 1)
    lb = [0] * (main.rows * main.cols + len(main.start_service) + 1)
    precision = [1] * (main.rows * main.cols) + [1e-7] * (len(main.start_service) + 1)
    n_particles = int(1000 * np.log(len(lb)))
    n_particles = n_particles + n_particles % 2
    # print("n_particles", n_particles)
    max_iter = int(20 * np.log(len(lb)))
    # print("max_iter", max_iter)

    # define GA
    ga = GA(func=heuristic_algorithm_fitness_function,
            n_dim=len(lb),
            size_pop=n_particles,
            max_iter=max_iter,
            precision=precision,
            lb=lb,
            ub=ub,
            )
    ga.register(operator_name='crossover', operator=crossover_2point)

    # iter
    x_opt, y_opt = ga.run(max_iter=max_iter)

    # show result
    Y_history = pd.DataFrame(ga.all_history_Y)
    # fig, ax = plt.subplots(2, 1)
    # ax[0].plot(Y_history.index, Y_history.values, '.', color='red')
    Y_history.min(axis=1).cummin().plot(kind='line')
    print("best_x:\n", ga.best_x, "\nbest_y:", ga.best_y)
    plt.savefig("GA.svg")
    plt.show()

