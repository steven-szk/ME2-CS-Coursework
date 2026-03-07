import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.fft import fft, fftfreq

# ==========================================
# 1. 物理与网格参数设置 (Parameters)
# ==========================================
Lx, Ly = 10.0, 10.0     
Nx, Ny = 80, 80         
dx = Lx / (Nx - 1)      
dy = Ly / (Ny - 1)      
c = 343.0               

# 修复数值爆炸：减小时间步长以严格满足 CFL 条件！
dt = 0.0002             # 时间步长改小到 0.0002s
T_total = 0.1           
Nt = int(T_total / dt)  

Cx2 = (c * dt / dx)**2
Cy2 = (c * dt / dy)**2

print(f"CFL Number Squared (must be <= 1): {Cx2 + Cy2:.3f}")

# (后面代码的不用变，直接运行)


# ==========================================
# 2. 初始化与初始条件 (Task 2 & 5)
# ==========================================
x = np.linspace(0, Lx, Nx)
y = np.linspace(0, Ly, Ny)
X, Y = np.meshgrid(x, y)

# 建立三个时间层：过去(n-1), 现在(n), 未来(n+1)
p_old = np.zeros((Ny, Nx))
p_curr = np.zeros((Ny, Nx))
p_new = np.zeros((Ny, Nx))

# 初始条件：在房间中心放置一个高斯声压脉冲
A = 100.0     # 脉冲初始声压振幅 (Pa)
sigma = 0.5   # 脉冲宽度
xc, yc = Lx/2, Ly/2
p_curr = A * np.exp(-((X - xc)**2 + (Y - yc)**2) / (2 * sigma**2))

# 记录角落某一点的声压历史，留给 Task 7 做 FFT 分析
# 我们选择左下角靠里一点的位置作为“虚拟麦克风”
mic_x_idx, mic_y_idx = 10, 10 
mic_history = np.zeros(Nt)
time_array = np.linspace(0, T_total, Nt)

# ==========================================
# 3. 核心求解循环 (Task 5)
# ==========================================
# 用于记录某一个中间时刻的数据以供画图
plot_step = int(0.02 / dt) # 大约 0.015 秒时的声波状态
p_plot_2d = None

for n in range(Nt):
    # 记录麦克风位置的声压
    mic_history[n] = p_curr[mic_y_idx, mic_x_idx]
    
    # --- 内部节点更新 ---
    if n == 0:
        # 第一步特殊处理：利用课件中初始速度为0的推导消除 p_old
        p_new[1:-1, 1:-1] = p_curr[1:-1, 1:-1] + 0.5 * (
            Cx2 * (p_curr[1:-1, 2:] - 2*p_curr[1:-1, 1:-1] + p_curr[1:-1, :-2]) +
            Cy2 * (p_curr[2:, 1:-1] - 2*p_curr[1:-1, 1:-1] + p_curr[:-2, 1:-1])
        )
    else:
        # 主循环：使用矩阵切片快速计算中心差分
        p_new[1:-1, 1:-1] = (2 * p_curr[1:-1, 1:-1] - p_old[1:-1, 1:-1] + 
            Cx2 * (p_curr[1:-1, 2:] - 2*p_curr[1:-1, 1:-1] + p_curr[1:-1, :-2]) +
            Cy2 * (p_curr[2:, 1:-1] - 2*p_curr[1:-1, 1:-1] + p_curr[:-2, 1:-1]))
            
    # --- 施加 Neumann 边界条件 (刚性墙壁反弹) ---
    # 墙壁上的压力等于内部紧邻网格的压力，使得梯度(导数)为 0
    p_new[0, :] = p_new[1, :]   # 下墙壁
    p_new[-1, :] = p_new[-2, :] # 上墙壁
    p_new[:, 0] = p_new[:, 1]   # 左墙壁
    p_new[:, -1] = p_new[:, -2] # 右墙壁
    
    # 提取中间时刻数据用于等高线绘图
    if n == plot_step:
        p_plot_2d = p_new.copy()
        
    # 时间推进
    p_old = p_curr.copy()
    p_curr = p_new.copy()

# ==========================================
# 4. 绘图与后处理 (Task 6 & Task 7)
# ==========================================
plt.style.use('bmh')
fig = plt.figure(figsize=(16, 12))

# --- Plot 1: 3D Surface Plot (0.015s 时的声波 3D 曲面) ---
ax1 = fig.add_subplot(2, 2, 1, projection='3d')
# 为了 3D 图好看，我们在 Z 轴上做一点限制
surf = ax1.plot_surface(X, Y, p_plot_2d, cmap='coolwarm', edgecolor='none')
ax1.set_title(f'Task 6-1: 3D Acoustic Pressure at t={plot_step*dt:.3f}s')
ax1.set_xlabel('x (m)')
ax1.set_ylabel('y (m)')
ax1.set_zlabel('Pressure (Pa)')
ax1.set_zlim(-A/2, A/2)

# --- Plot 2: 2D Contour (波纹撞击墙壁的热力图) ---
ax2 = fig.add_subplot(2, 2, 2)
# 使用等高线填充图展示波的传播和反弹
contour = ax2.contourf(X, Y, p_plot_2d, levels=50, cmap='RdBu', vmin=-A/3, vmax=A/3)
fig.colorbar(contour, ax=ax2, label='Pressure (Pa)')
ax2.set_title('Task 6-2: 2D Wave Propagation & Reflection')
ax2.set_xlabel('x (m)')
ax2.set_ylabel('y (m)')

# --- Plot 3: 1D Line Plot (麦克风记录的声压随时间变化) ---
ax3 = fig.add_subplot(2, 2, 3)
ax3.plot(time_array, mic_history, color='navy', linewidth=1)
ax3.set_title('Task 6-3: Microphone Pressure History')
ax3.set_xlabel('Time (s)')
ax3.set_ylabel('Pressure (Pa)')

# --- Plot 4: Task 7 整合傅里叶分析 (FFT 找房间共振频率) ---
# 利用上面的麦克风历史数据进行快速傅里叶变换
N_fft = len(mic_history)
yf = fft(mic_history)
xf = fftfreq(N_fft, dt)[:N_fft//2] # 获取正频率部分

ax4 = fig.add_subplot(2, 2, 4)
# 绘制频谱幅度图
ax4.plot(xf[1:], 2.0/N_fft * np.abs(yf[1:N_fft//2]), color='darkred')
ax4.set_title('Task 7: Fourier Analysis (Room Resonant Frequencies)')
ax4.set_xlabel('Frequency (Hz)')
ax4.set_ylabel('Amplitude')
ax4.set_xlim(0, 1500) # 我们主要关注 0-1500 Hz 的声学频率

plt.tight_layout()
plt.show()