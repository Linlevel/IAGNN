import matplotlib.pyplot as plt
import numpy as np

# 模拟数据
data = np.array([
    [0.512, 0.544, 0.525, 0.541, 0.585, 0.652],
    [0.531, 0.551, 0.561, 0.642, 0.621, 0.657],
    [0.585, 0.666, 0.651, 0.621, 0.712, 0.721],
    [0.656, 0.755, 0.794, 0.824, 0.82, 0.85],
    [0.667, 0.756, 0.796, 0.816, 0.825, 0.858],
    [0.725, 0.805, 0.815, 0.855, 0.845, 0.836]
])

fig, ax = plt.subplots()
cax = ax.matshow(data, cmap='summer_r')

# 添加颜色条
cbar = fig.colorbar(cax)

# # 设置颜色条标签
# cbar.set_label('')
# cbar.set_ticks([0.83, 0.84, 0.85, 0.86, 0.87, 0.88])

# 设置刻度和标签
ax.set_xticks(np.arange(len(data[0])))
ax.set_yticks(np.arange(len(data)))
ax.set_xticklabels([f'{i/10:.1f}' for i in range(len(data[0]))])
ax.set_yticklabels([f'{i/10:.1f}' for i in range(len(data))])

# 设置标题和轴标签
ax.set_title('Kinship')
ax.set_xlabel('Hidden dropout')
ax.set_ylabel('Feature map dropout')

plt.show()
