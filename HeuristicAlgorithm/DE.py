import numpy as np
from matplotlib import pyplot as plt
from sko.DE import DE

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
    de = DE(func=heuristic_algorithm_fitness_function,
            n_dim=len(lb),
            size_pop=n_particles,
            max_iter=max_iter,
            lb=lb,
            ub=ub,
            )

    best_x, best_y = de.run()
    print('best_x:', best_x, '\n', 'best_y:', best_y)


