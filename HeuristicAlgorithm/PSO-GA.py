import numpy as np
import pandas as pd
from sko.operators.crossover import crossover_2point
import matplotlib.pyplot as plt
from PSGA import PSO

# from TD3.scale_min import environment_min as environment
from TD3.scale_mid import environment_mid3 as environment

if __name__ == '__main__':
    # init
    ub = [1] * (environment.services * environment.nodes + len(environment.start_service) + 1)
    lb = [0] * (environment.services * environment.nodes + len(environment.start_service) + 1)
    precision = [1] * (environment.services * environment.nodes) + [1e-7] * (len(environment.start_service) + 1)
    n_particles = int(1000 * np.log(len(lb)))
    n_particles = n_particles + n_particles % 2
    max_iter = int(20 * np.log(len(lb)))

    # define GA
    pso = PSO(func=environment.heuristic_algorithm_fitness_function,
              n_dim=len(lb),
              pop=n_particles,
              max_iter=max_iter,
              lb=lb,
              ub=ub,
              w=0.8, c1=0.5, c2=0.5,
              precision=int(n_particles/5),
              )

    pso.run()
    print('best_x is \n', pso.gbest_x, '\nbest_y is\n', pso.gbest_y)
    print("best result:", environment.heuristic_algorithm_fitness_function(pso.gbest_x))
    plt.plot(pso.gbest_y_hist)
    plt.savefig("PSO_GA_mid.svg")
    plt.show()
