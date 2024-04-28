from scale_mid import environment_mid

if __name__ == '__main__':
    for i_ in range(50):
        state = environment_mid.constrains_reset()
        print(state)
        print(environment_mid.get_reward())
        print("----------------------")
