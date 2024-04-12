import numpy as np
import pandas as pd
from sko.operators.crossover import crossover_2point
import matplotlib.pyplot as plt
from sko.PSO import PSO

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
    pso = PSO(func=heuristic_algorithm_fitness_function,
              n_dim=len(lb),
              pop=n_particles,
              max_iter=max_iter,
              lb=lb,
              ub=ub,
              w=0.8, c1=0.5, c2=0.5,
              )
    # pso.register(operator_name='crossover', operator=crossover_2point)

    # iter
    pso.run()
    print('best_x is \n', pso.gbest_x, '\nbest_y is\n', pso.gbest_y)
    print("best result:", heuristic_algorithm_fitness_function(pso.gbest_x))
    main.test_heuristic_algorithm_fitness_function(pso.gbest_x)
    plt.plot(pso.gbest_y_hist)
    plt.show()