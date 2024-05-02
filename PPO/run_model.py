import copy

import torch
from scale_min import two_environment_min as env

if __name__ == '__main__':
    state = torch.tensor([0.0000, 0.0000, 1.0000, 0.0000, 0.0000, 1.0000, 1.0000, 0.0000, 1.0000,
                          0.0000, 0.0000, 0.0000, 1.0000, 0.0000, 0.0000, 0.0000, 0.0000, 1.0000,
                          1.0000, 1.0000, 0.0000, 0.0000, 0.0000, 0.0000, 1.0000, 0.5000])
    environment = copy.deepcopy(env)
    # state = torch.tensor([0.0000, 1.0000, 1.0000, 0.0000, 0.0000, 1.0000, 1.0000, 0.0000, 1.0000,
    #                       0.0000, 0.0000, 1.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 1.0000,
    #                       1.0000, 1.0000, 0.0000, 0.0000, 1.0000, 0.0000, 0.0000, 0.5000])
    environment.update_state(state)
    print(environment.get_reward())
