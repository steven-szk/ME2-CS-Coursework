import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq

# ==========================================
# 1. 参数设置 (基于文档 H 问的物理与网格参数)
# ==========================================
Lx = 10.0          # x 轴长度 (m)
Ly = 10.0          # y 轴长度 (m)
Nx = 80            # x 轴网格数
Ny = 80            # y 轴网格数
h = Lx / (Nx - 1)  # 空间步长 h = dx = dy = 0.126 m
c = 343.0          # 声速 (m/s)

# 为了满足文档 H 问中算出的 Courant Number C = 0.54
# dt 必须设定为 0.0002s (文档中的 0.002s 是一个笔误，会导致 C=5.44 而爆炸)
dt = 0.0002        # 时间步长 k = dt (s)
T_total = 0.1      # 总模拟时间 (s)
Nt = int(T_total / dt) # 总时间步数

# 库朗数 (Courant Number)
C_courant = c * dt / h
print(f"Space step h: {h:.3f} m")
print(f"Time step dt: {dt:.4f} s")
print(f"Courant Number C: {C_courant:.3f} (Must be <= 1/sqrt(2) approx 0.707)")

# ==========================================
# 2. 变量声明与初始条件 (基于文档 F 问和 C 问)
# ==========================================
# 根据 F 问的要求，创建一个 3D NumPy 数组来存储所有时间点和空间点的数据
# 维度顺序为: [时间 n, 空间 y, 空间 x]
P = np.zeros((Nt, Ny, Nx), dtype=np.float64)

# 生成坐标网格
x = np.linspace(0, Lx, Nx)
y = np.linspace(0, Ly, Ny)
X, Y = np.meshgrid(x, y)

# IC 初始条件设置：两个高斯脉冲的叠加
A = 100.0     # 振幅
sigma = 0.5   # 宽度

# 根据 Figure 1 简图，两个音箱位于房间前部 (y=8左右)，分别在左边和右边
x1, y1 = 2.0, 8.0  # 左前音箱
x2, y2 = 8.0, 8.0  # 右前音箱

# 文档 C 问的初始声压公式
P[0, :, :] = A * np.exp(-((X - x1)**2 + (Y - y1)**2) / (2 * sigma**2)) + \
             A * np.exp(-((X - x2)**2 + (Y - y2)**2) / (2 * sigma**2))

# ==========================================
# 3. 数值离散求解 (基于文档 E 问的公式推导)
# ==========================================
# n = 0 时的特殊启动步 (代入初始速度为 0 的条件)
# 严格对应 E 问公式: p_{i,j}^1 = p_{i,j}^0 + (C^2 / 2) * (...)
P[1, 1:-1, 1:-1] = P[0, 1:-1, 1:-1] + (C_courant**2 / 2.0) * (
    P[0, 1:-1, 2:] + P[0, 1:-1, :-2] + P[0, 2:, 1:-1] + P[0, :-2, 1:-1] - 4 * P[0, 1:-1, 1:-1]
)

# 施加 n = 1 时的 Neumann 边界条件 (四周刚性墙壁，法向梯度为0)
P[1, 0, :] = P[1, 1, :]       # y=0
P[1, -1, :] = P[1, -2, :]     # y=Ly
P[1, :, 0] = P[1, :, 1]       # x=0
P[1, :, -1] = P[1, :, -2]     # x=Lx

# n >= 1 的主循环
for n in range(1, Nt - 1):
    # 严格对应 E 问公式: p_{i,j}^{n+1} = 2p_{i,j}^n - p_{i,j}^{n-1} + C^2 * (...)
    P[n+1, 1:-1, 1:-1] = 2 * P[n, 1:-1, 1:-1] - P[n-1, 1:-1, 1:-1] + C_courant**2 * (
        P[n, 1:-1, 2:] + P[n, 1:-1, :-2] + P[n, 2:, 1:-1] + P[n, :-2, 1:-1] - 4 * P[n, 1:-1, 1:-1]
    )
    
    # 施加 Neumann 边界条件
    P[n+1, 0, :] = P[n+1, 1, :]       # Bottom wall
    P[n+1, -1, :] = P[n+1, -2, :]     # Top wall
    P[n+1, :, 0] = P[n+1, :, 1]       # Left wall
    P[n+1, :, -1] = P[n+1, :, -2]     # Right wall

# ==========================================
# 4. 后处理与绘图 (基于文档 G 问和 K 问)
# ==========================================
# 提取特定时刻用于绘制 2D 和 3D 图像
plot_time = 0.015
plot_step = int(plot_time / dt)

# 提取虚拟麦克风数据 (位于房间中央)
mic_x_idx, mic_y_idx = Nx // 2, Ny // 2
# 因为我们使用了 3D array，可以直接通过切片获取整个时间历史！
mic_history = P[:, mic_y_idx, mic_x_idx]
time_array = np.linspace(0, T_total, Nt)

plt.style.use('bmh')
fig = plt.figure(figsize=(16, 12))

# 1. 3D 曲面图 (3D Surface Plot)
ax1 = fig.add_subplot(2, 2, 1, projection='3d')
surf = ax1.plot_surface(X, Y, P[plot_step, :, :], cmap='viridis', edgecolor='none')
ax1.set_title(f'3D Acoustic Pressure at t={plot_time}s')
ax1.set_xlabel('x (m)'); ax1.set_ylabel('y (m)'); ax1.set_zlabel('Pressure (Pa)')
ax1.set_zlim(-A, A) 

# 2. 2D 等高线干涉图 (2D Contour Plot)
ax2 = fig.add_subplot(2, 2, 2)
contour = ax2.contourf(X, Y, P[plot_step, :, :], levels=60, cmap='RdBu', vmin=-A/3, vmax=A/3)
fig.colorbar(contour, ax=ax2, label='Pressure (Pa)')
ax2.set_title('Wave Interference & Wall Reflection')
ax2.set_xlabel('x (m)'); ax2.set_ylabel('y (m)')

# 3. 麦克风时域波形图 (1D Line Plot)
ax3 = fig.add_subplot(2, 2, 3)
ax3.plot(time_array, mic_history, color='navy')
ax3.set_title('Microphone Pressure History')
ax3.set_xlabel('Time (s)'); ax3.set_ylabel('Pressure (Pa)')

# 4. FFT 频谱分析 (Task K)
N_fft = len(mic_history)
yf = fft(mic_history)
xf = fftfreq(N_fft, dt)[:N_fft//2]

ax4 = fig.add_subplot(2, 2, 4)
ax4.plot(xf[1:], 2.0/N_fft * np.abs(yf[1:N_fft//2]), color='darkred')
ax4.set_title('Fourier Analysis (Room Resonant Frequencies)')
ax4.set_xlabel('Frequency (Hz)'); ax4.set_ylabel('Amplitude')
ax4.set_xlim(0, 1500) 

plt.tight_layout()
plt.show()