from matplotlib import pyplot as plt
import pandas as pd

if __name__ == '__main__':
    scale_list = ['5_5', '7_7', '10_10', '12_12', '15_15']
    fig, axes = plt.subplots(2, 3)
    # fig.set_size_inches(10, 5 * len(data_types))

    for idx in range(3):
        Y_history1 = pd.read_csv(f'run_data/G-GA_{scale_list[idx]}.csv')
        Y_history2 = pd.read_csv(f'run_data/PSO_{scale_list[idx]}.csv')
        Y_history3 = pd.read_csv(f'run_data/ORI_GA_{scale_list[idx]}.csv')

        # 设置线条粗细
        axes[0][idx].plot(Y_history2.index, Y_history2.min(axis=1).cummin(), label='PSO', color='steelblue', alpha=0.8, linewidth=0.5)
        axes[0][idx].plot(Y_history1.index, Y_history1.min(axis=1).cummin(), label='G_GA', color='indianred', alpha=0.8, linewidth=0.5)
        axes[0][idx].plot(Y_history3.index, Y_history3.min(axis=1).cummin(), label='ORI_GA', color='green', alpha=0.8, linewidth=0.5)

        axes[0][idx].set_title(f'scale-{scale_list[idx]}')

        axes[0][idx].legend(loc='upper right')

        axes[0][idx].set_xlabel('Iteration')
        axes[0][idx].set_ylabel('Fitness')

    for idx in range(3, 4):
        Y_history1 = pd.read_csv(f'run_data/G-GA_{scale_list[idx]}.csv')
        Y_history2 = pd.read_csv(f'run_data/PSO_{scale_list[idx]}.csv')
        Y_history3 = pd.read_csv(f'run_data/ORI_GA_{scale_list[idx]}.csv')

        axes[1][idx-3].plot(Y_history2.index, Y_history2.min(axis=1).cummin(), label='PSO', color='steelblue', alpha=0.8, linewidth=0.5)
        axes[1][idx-3].plot(Y_history1.index, Y_history1.min(axis=1).cummin(), label='G-GA', color='indianred', alpha=0.8, linewidth=0.5)
        axes[1][idx-3].plot(Y_history3.index, Y_history3.min(axis=1).cummin(), label='ORI_GA', color='green', alpha=0.8, linewidth=0.5)
        axes[1][idx-3].set_title(f'scale-{scale_list[idx]}')

        axes[1][idx-3].legend(loc='upper right')

        axes[1][idx-3].set_xlabel('Iteration')
        axes[1][idx-3].set_ylabel('Fitness')

    axes[1, 2].axis('off')
    plt.tight_layout()
    plt.savefig(f"heuristic_algorithm_comparison.svg")
    plt.show()

