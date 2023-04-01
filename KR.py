import random

import numpy
import numpy as np
import matplotlib.pyplot as plt
import math
from matplotlib.animation import FuncAnimation
# random.seed(10)
# np.random.seed(20)

# 生成一个随机平滑函数的算法：在频谱上随机取F_size=20个常数，
# 然后利用这些单频对应的三角函数直接相加再截断就可以得到一个有限的平滑的函数
F_size=20
freqs = np.random.uniform(low=0, high=0.1, size=F_size)
amps = np.random.uniform(low=0, high=80, size=F_size)
phi=np.random.uniform(low=0, high=3.14*2, size=F_size)
# print("F:\n",freqs,"\nA\n:",amps,"\nP:\n",phi)
tmp=0

#生成时间轴、设定时间切片，x为坐标轴，dt为时间片长度，ti为模拟总时间
dt=0.01
ti=100
x = np.arange(start=0, stop=ti, step=dt)

# zs存储观察值，也就是传感器得到的带噪声值
zs = np.arange(start=0, stop=ti, step=dt)
# zsc (zs_clean)为不带噪声的观察值，可做调试对比用
zsc = np.arange(start=0, stop=ti, step=dt)
# 噪声分布符合 N(0,h),实际中可以在无信源的情况下测测量得到其分布,噪声的分布情况会可作为调整参数的根据
h=5
for i in range(int(ti/dt)):
    zs[i]=0
    for j in range(F_size):
        zsc[i]+=amps[j]*math.sin(freqs[j]*x[i]+phi[j])
        # 用循环将上文产生的随机频谱叠加成连续光滑函数

# 下面这个for循环可以强行将信源的20%至50%设置为常数，调试用
# for i in range(int(0.2*ti/dt),int(0.5*ti/dt)):
#     zsc[i] =zsc[int(0.2*ti/dt)]

# 加噪声，random.normal(0, h)为产生符合N(0,h)的随机浮点数
for i in range(int(ti/dt)):
    zs[i] = zsc[i] + np.random.normal(0, h)

# pd1数组后文用于存储均值滤波器得到的值，初始化全0
pd1=np.arange(start=0, stop=ti, step=dt)
for i in range(int(ti/dt)):
    pd1[i]=0


# 均值滤波，大小为fsize，可自行更改
fsize=40
for i in range(int(ti/dt)-fsize):
    # 循环采样，存于tmplist等一下计算平均数----
    tmplist=[]
    for j in range(fsize):
        tmplist.append(zs[i+j])
    tmp=numpy.mean(tmplist)
    # 循环采样-------------，其实均值不需要这样循环，可以直接利用加权和来得到，不过我懒得写了
    # 存入pd1，等一下绘图
    pd1[i]=tmp


# 滤波的思想如下：
# 在已知信源无噪声真实值连续，即不存在大量的阶跃的情况下,
# 可以将下一次采样的值进行提前预测，即：
# （未知的下一次的值）=（已知的这次值）+（变化速度）*dt
#  同时 还可以 将 下一次的变化速度 认为是 当前认为的变化速度 + 加速度*dt
# 同理可以一直求导，对信号的每一阶导数来进行测量和估计，程度取决于设备的计算能力和精度需求

# pd2用于存储预测值，初始化全0
pd2=np.arange(start=0, stop=ti, step=dt)
for i in range(int(ti/dt)):
    pd2[i]=0

test=np.arange(start=0, stop=ti, step=dt)


# 初始化部分计算需要的变量，v为估计的信号变化速度
# ov（old_v）用于记录上次的速度，用于做差计算加速度
v=0
ov=v

# 改进型滤波器，核大小为fsize，可自行更改，在小于均值滤波器核大小时都可以效果更换
# 本算法在均值滤波器上改进，故也要先采样fsize个值，注意此处的fsize比上文的均值滤波小很多
fsize=30
for i in range(1,int(ti/dt)-fsize):
    # old 用于记录上一次的均值
    old=tmp
    # --------------------
    tmp=0
    tmplist=[]
    for j in range(fsize):
        tmplist.append(zs[i+j])
    tmp=numpy.mean(tmplist)
    # 循环采样-------------，其实均值不需要这样循环，可以直接利用加权和来得到，不过我懒得写了

    # 这个算法的核心思想在于推测信号的高阶导数。也就是变化速度、加速度或者更高阶
    # 故前200个采样点用于拿“信息”，直接用带噪的采样值处理，利用这些信息初始化速度v和加速度a
    if i<200:
        pd2[i+1]=tmp
        ov=v
        v=(tmp-old)/dt
        a=(v-ov)/dt
    else:
        # pdt为预测第i+1次值，显然预测公式为（第i+1次值）=（第i次值）+（变化速度v）*dt
        pdt=pd2[i]+v*dt
        # kv为权重，即认为预测值的重要程度，可以根据实际来调整
        # 显然当噪声比较大时候kv应该调大。若噪声很小，以至于采样的信号很纯净，则kv应该趋近于0
        kv=0.99
        # 同时对估计的速度进行更新，下一次的估计速度 = 旧的估计速度 与 当前的测量速度 按权相加
        # 当前的测量速度 = (tmp-old)/dt
        v=kv*v+(1-kv)*(tmp-old)/dt
        # 同理更新加速度，当前加速度=(v-ov)/dt
        a = a*kv+ (1-kv)*(v-ov)/dt
        # 存储旧的速度old_v用于下一次迭代
        ov=v
        # 将预测的信号值pdt与当成采样得到的带噪值tmp按权相加
        # 存入pd2等一下绘图
        pd2[i+1]=kv*(pdt)+(1-kv)*tmp



# 开始绘制动画，初始化图像
fig, ax = plt.subplots()
# 创建两个曲线对象
line1, = ax.plot([], [], label='origin')
line2, = ax.plot([], [], label='fitter')
line3, = ax.plot([], [], label='mean')
# 添加图例
ax.legend()
ax.set_xlim(min(x), max(x))
ax.set_ylim(min(zs), max(zs))
# 定义更新函数，每次更新两个曲线的数据
def update(frame):
    line1.set_data(x[:frame], zs[:frame])
    line2.set_data(x[:frame], pd2[:frame])
    line3.set_data(x[:frame], pd1[:frame])
    return line1, line2,line3

# 创建动画对象
# interval 为绘制下一个点的时间间隔，单位毫秒，不过实际测试发现过小后没啥用
ani = FuncAnimation(fig, update, frames=len(x), blit=True,interval=4)
# 显示动画
# plt.plot(x, zs,color='m')
# plt.plot(x, zsc,color='red')
# plt.plot(x, pd1,color='k')
# plt.plot(x, pd2,color='blue')
plt.show()


