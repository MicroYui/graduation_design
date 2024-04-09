import numpy as np
import pandas as pd
from numpy import shape
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
            ub=ub
            )
    ga.register(operator_name='crossover', operator=crossover_2point)

    # iter
    x_opt, y_opt = ga.run(max_iter=max_iter)

    # show result
    Y_history = pd.DataFrame(ga.all_history_Y)
    Y_history_min = Y_history.min(axis=1)
    # print(ga.best_x)
    # print(ga.best_y)
    # print(heuristic_algorithm_fitness_function(ga.best_x))
    print("best x:\n", ga.best_x)
    print("best y:", heuristic_algorithm_fitness_function(ga.best_x))

    # df = pd.DataFrame({
    #     'A': [1, 2, 3, 4, 5],
    #     'B': [5, 4, 3, 2, 1],
    #     'C': [2, 3, 4, 5, 6]
    # })
    #
    # print(df)

    # 绘制散点图
    rows = range(len(Y_history_min))
    plt.scatter(rows, Y_history_min)
    plt.title('GA')
    plt.show()

