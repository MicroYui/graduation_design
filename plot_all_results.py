import numpy as np
from matplotlib import pyplot as plt
import pandas as pd


def plot_all_algorithm_result():
    scale_list = ['5_5', '7_7', '10_10', '12_12', '15_15']
    method_list = ['GA', 'PSO']
    G_GA_list = []
    PSO_list = []
    PPO_list = [173.58909087216088, 114.67042717332686, 265.7320079876676, 161.54614411910057, 182.65317310290948]
    for method in method_list:
        for scale in scale_list:
            Y_history = pd.read_csv(f'HeuristicAlgorithm/run_data/{method}_{scale}.csv')
            if method == 'GA':
                G_GA_list.append(Y_history.min().min())
            else:
                PSO_list.append(Y_history.min().min())
    # 设置每个柱子的位置
    x = np.arange(len(scale_list))

    # 设置每个柱子的宽度
    width = 0.2

    # print(G_GA_list)
    # print(PSO_list)

    # 创建柱状图
    plt.bar(x - width, G_GA_list, width, label='G-GA')
    plt.bar(x, PSO_list, width, label='PSO')
    plt.bar(x + width, PPO_list, width, label='PPO')

    # 添加标题和标签
    plt.title('Comparison Of All Algorithm')
    plt.xlabel('scale')
    plt.ylabel('fitness')
    plt.xticks(x, scale_list)
    plt.legend()

    # 显示图形
    plt.show()


if __name__ == '__main__':
    plot_all_algorithm_result()
