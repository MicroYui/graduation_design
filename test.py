import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

# 初始化图表
fig, ax = plt.subplots()
line, = ax.plot([], [])

# 更新函数，用于更新数据和图表
def update(frame):
    # 这里假设data是你不断更新的数组
    data.append(np.random.rand())  # 这里用随机值代替实际的数据更新
    line.set_data(range(len(data)), data)  # 更新图表数据
    ax.relim()  # 重新计算轴的界限
    ax.autoscale_view()  # 自动调整轴的范围
    return line,

# 初始化数组
data = []

if __name__ == '__main__':
    ani = animation.FuncAnimation(fig, update, frames=100, interval=200)  # frames表示总帧数，interval表示更新间隔
    plt.show(block=True)

