import numpy as np

# 示例列表
lst = [10, 10, 6, 20]

# 获取元素从小到大排序的索引
sorted_idx = np.argsort(lst)

# 获取元素从大到小排序的索引
sorted_reverse_idx = sorted_idx[::-1]

print(sorted_reverse_idx)