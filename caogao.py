import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# 生成两个随机函数
x = np.linspace(0, 10, 100)
y1 = np.sin(x)
y2 = np.cos(x)

# 创建画布和子图
fig, ax = plt.subplots()

# 创建两个曲线对象
line1, = ax.plot([], [], label='sin')
line2, = ax.plot([], [], label='cos')

# 添加图例
ax.legend()
ax.set_xlim(min(x), max(x))
ax.set_ylim(min(y1), max(y2))
# 定义更新函数，每次更新两个曲线的数据
def update(frame):
    line1.set_data(x[:frame], y1[:frame])
    line2.set_data(x[:frame], y2[:frame])
    return line1, line2,

# 创建动画对象
ani = FuncAnimation(fig, update, frames=len(x), blit=True)

# 显示动画
plt.show()
