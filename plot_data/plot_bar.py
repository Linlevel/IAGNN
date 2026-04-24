import matplotlib.pyplot as plt
import numpy as np
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 14
# 数据
filter_sizes = ['100', '200', '300', '400']
Kinship = [0.778, 0.840, 0.818, 0.826]
UMLS = [0.977, 0.971, 0.972, 0.964]

x = np.arange(len(filter_sizes))  # 标签位置
width = 0.2  # 条形宽度

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, UMLS, width, label='UMLS', color='#ff9999',edgecolor='k', linewidth=0.1)
rects2 = ax.bar(x + width/2+0.03, Kinship, width, label='Kinship', color='#66b3ff',edgecolor='k', linewidth=0.1)

# 添加标签、标题、自定义x轴刻度等
ax.set_xlabel('实体和关系嵌入维度')
ax.set_ylabel('Hits@1')
ax.set_xticks(x)
ax.set_xticklabels(filter_sizes)
ax.set_ylim([0.73, 1.00])
ax.legend(loc='lower right')
# 只显示横向网格
ax.yaxis.grid(True)
ax.xaxis.grid(False)

# 在rects1和rects2中的每个条形上方附加文本标签，显示其高度。
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.3f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, -20),  # 垂直偏移量
                    textcoords="offset points",
                    ha='center', va='center', rotation=90, color='black', fontsize=12)

autolabel(rects1)
autolabel(rects2)

fig.tight_layout()

plt.show()
