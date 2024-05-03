def knapsack_3d(weights, values, capacities):
    n = len(weights)
    dp = [[[0 for _ in range(capacities[2] + 1)] for _ in range(capacities[1] + 1)] for _ in range(capacities[0] + 1)]

    for i in range(1, n + 1):
        for j in range(capacities[0] + 1):
            for k in range(capacities[1] + 1):
                for l in range(capacities[2] + 1):
                    if weights[i - 1] <= j and weights[i - 1] <= k and weights[i - 1] <= l:
                        dp[j][k][l] = max(dp[j][k][l],
                                          dp[j - weights[i - 1]][k - weights[i - 1]][l - weights[i - 1]] + values[
                                              i - 1])
                    else:
                        dp[j][k][l] = dp[j][k][l]

    return dp[capacities[0]][capacities[1]][capacities[2]]


# 示例
weights = [1, 2, 3]  # 物品重量
values = [6, 10, 12]  # 物品价值
capacities = [5, 5, 5]  # 背包容量
max_value = knapsack_3d(weights, values, capacities)
print("背包能装的最大价值为:", max_value)
