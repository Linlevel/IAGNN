import matplotlib.pyplot as plt
import numpy as np

# 模拟数据
beta = np.arange(0.0, 10, 1)
mrr_hran = [1.71,1.7,1.65,1.68,1.54,1.57,1.56,1.63,1.53,1.6]

fig, ax = plt.subplots()

# 绘制HRAN和HRN的折线图
ax.plot(beta, mrr_hran, marker='s', color='orange', label='IAGNN')

# 标注最优点
opt_beta = 8
opt_mrr = 1.53
ax.plot(opt_beta, opt_mrr, marker='*', color='red', markersize=10)


# 绘制矩形框
rect_x = [opt_beta - 0.25, opt_beta + 0.25, opt_beta + 0.25, opt_beta - 0.25, opt_beta - 0.25]
rect_y = [opt_mrr-0.005, opt_mrr-0.005, opt_mrr+0.005, opt_mrr+0.005, opt_mrr-0.005]
ax.plot(rect_x, rect_y, color='red')

# 设置标签、标题和图例
ax.set_xlabel(r'$\beta$')
ax.set_ylabel('MR')
ax.set_title('')
ax.set_xticks(beta)
ax.set_ylim([1.50, 1.75])
ax.legend()

# 添加网格
ax.grid(True)

plt.show()
