import torch

from scale_min import environment_min

if __name__ == '__main__':
    environment = environment_min
    state = torch.tensor([0.0000, 0.0000, 1.0000, 0.0000, 0.0000, 1.0000, 1.0000, 0.0000, 1.0000,
                          0.0000, 0.0000, 0.0000, 1.0000, 0.0000, 1.0000, 0.0000, 0.0000, 1.0000,
                          1.0000, 1.0000, 0.0000, 0.0000, 0.0000, 0.0000, 1.0000])
    print(environment.update_state(state))
    print(environment.request_arrive)
    print(environment.check_constrains())
    print(environment.get_reward())
