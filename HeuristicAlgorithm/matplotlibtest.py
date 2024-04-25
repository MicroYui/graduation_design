from matplotlib import pyplot as plt

list1 = [1, 2, 3, 4, 5]
list2 = [2, 3, 5, 7, 11]

if __name__ == '__main__':
    plt.plot(list1, label='list1')
    plt.plot(list2, label='list2')

    plt.legend()  # 显示图例
    plt.show()
