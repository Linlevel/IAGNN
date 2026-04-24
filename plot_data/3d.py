import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d.proj3d import proj_transform


class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        FancyArrowPatch.draw(self, renderer)


# 数据
batch_sizes = [64, 128, 256]
label_smoothings = [0.0, 0.1, 0.2, 0.3]
hits_at_10 = np.array([
    [0.791, 0.815, 0.834, 0.818],  # 64 batch_sizes
    [0.799, 0.823, 0.840, 0.821],  # 128 batch_sizes
    [0.792, 0.811, 0.830, 0.812],  # 256 batch_sizes
])

# 颜色
colors = ['#FFA07A', '#20B2AA', '#FF6347']

# z轴范围
z_min = hits_at_10.min() - 0.01
z_max = hits_at_10.max() + 0.01

# 创建图形
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# X, Y 位置
_x = np.arange(len(batch_sizes))
_y = np.arange(len(label_smoothings))
_xx, _yy = np.meshgrid(_x, _y)
x, y = _xx.ravel(), _yy.ravel()

# 柱状体高度
top = hits_at_10.T.ravel()  # 转置以正确匹配 x 和 y 的索引顺序

# 柱状体位置
bottom = np.zeros_like(top)
width = depth = 0.7

# 绘制柱状图
for i in range(len(batch_sizes)):
    for j in range(len(label_smoothings)):
        x_pos = i
        y_pos = j
        z_pos = z_min
        dx = width
        dy = depth
        dz = hits_at_10[i][j] - z_min
        color = colors[i % len(colors)]
        ax.bar3d(x_pos, y_pos, z_pos, dx, dy, dz, color=color, edgecolor='k', linewidth=0.1, shade=False, zorder=1)

# 设置刻度和标签
ax.set_xticks(_x)
ax.set_xticklabels(batch_sizes)
ax.set_yticks(_y)
ax.set_yticklabels(label_smoothings)
ax.set_xlabel('Batch size')
ax.set_ylabel('Label smoothing')
ax.set_zlabel('Hits@1')
ax.set_title('Kinship')
ax.set_zlim(z_min, z_max)

# 调整视角
ax.view_init(elev=30, azim=140)

# 找到最大值
max_value = hits_at_10.max()
max_index = np.unravel_index(np.argmax(hits_at_10, axis=None), hits_at_10.shape)

# 获取最大值的x和y位置
max_x = max_index[0]
max_y = max_index[1]

# 确保箭头在最高柱状体的正上方
arrow_x = max_x
arrow_y = max_y
arrow_z_start = hits_at_10[max_x, max_y]
arrow_z_end = arrow_z_start + 0.02

# 添加箭头和文本标签（确保在最后添加，并设置更高的zorder）
arrow = Arrow3D([arrow_x + width / 2, arrow_x + width / 2], [arrow_y + depth / 2, arrow_y + depth / 2], [arrow_z_start, arrow_z_end],
                mutation_scale=20, lw=1.5, arrowstyle="<|-", color="r", zorder=10)
ax.add_artist(arrow)
ax.text(arrow_x + width / 2, arrow_y + depth / 2, arrow_z_end, 'Max', color='red', fontsize=10, ha='center', zorder=10)

plt.show()
