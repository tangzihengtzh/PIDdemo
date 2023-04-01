import random

import numpy as np
import matplotlib.pyplot as plt
import math

# random.seed(10)
# np.random.seed(20)
# 生成随机的频谱
F_size=20
freqs = np.random.uniform(low=0, high=0.05, size=F_size)
amps = np.random.uniform(low=0, high=5, size=F_size)
phi=np.random.uniform(low=0, high=3.14*2, size=F_size)
# print("F:\n",freqs,"\nA\n:",amps,"\nP:\n",phi)

#生成时间轴、设定时间切片
dt=0.001
ti=200
x = np.arange(start=0, stop=ti, step=dt)

# 生成跟随目标值函数（部分三角函数部分阶跃函数）
y = np.arange(start=0, stop=ti, step=dt)
for i in range(int(ti/dt)):
    y[i]=0
    for j in range(F_size):
        y[i]+=amps[j]*math.sin(freqs[j]*x[i]+phi[j])
        # y[i] = 100 + np.random.normal(0, 1)

for i in range(int(0.2*ti/dt),int(0.5*ti/dt)):
    y[i] =y[int(0.2*ti/dt)]
    # y[i]=y[i] + 3
for i in range(int(0.4*ti/dt),int(0.6*ti/dt)):
    # y[i] = y[i] -3
    y[i] =y[int(0.6*ti/dt)]
# for i in range(int(0.8*ti/dt),int(1*ti/dt)):
#     y[i] = 4

# for i in range(int(ti/dt)):
#     y[i]=1

# 数组t用于记录无人机的高度坐标
t = np.arange(start=0, stop=ti, step=dt)
for i in range(1000):
    t[i]=0

# 假设初始状态起点,可以自行更改
t[0]=y[i]
t[1]=y[i]
# t[0]=1

# 假设程序可以控制无人机发动机的推力大小 FO = output
# output = Kp * error + Ki * integral(error) + Kd * derivative(error)
# PID三参数，自行更改
kp=0.2
ki=0.01
kd=155

# 假设无人机质量 me 为 1
me=1

# 临时变量
# integral(error)
ig=0
# derivative(error)
dv=0
# error
er=0
# 假设无人机初速度为 0
VO=0
G=0
maxF=200
for i in range(1,int(ti/dt)-1):
    er = y[i]-t[i]
    # 对误差进行积分
    ig = ig + er*dt
    # 对误差作差分（微分）,er[i]-er[i-1] = (y[i]-t[i]) - (y[i-1]-t[i-1])
    dv=t[i-1]-t[i]
    # 计算发动机输出推力，假定无人机推力最大为 maxF
    FO = kp * er + ki * ig + kd * dv
    if FO>maxF:
        FO=maxF
    if FO<-maxF:
        FO=-maxF
    if (t[i]-y[i])<=-10.5:
        FO=maxF
    # 加速度 = 推力+重力+扰动 / 质量
    # AO=(FO-G*me+random.uniform(-0.5,0.5))/me
    AO = (FO - G * me ) / me
    # if(i<=4000):
    #     AO=0
    # 更新速度，新速度 = 旧速度 + 加速度*时间片
    VO=VO+AO*dt
    # 更新坐标，新坐标 = 旧坐标 + 速度*时间片
    t[i+1]=t[i]+VO*dt


# 绘制跟随目标图像
plt.plot(x, y)
# 绘制物体坐标图像
plt.plot(x, t)
# 显示图像
plt.show()

