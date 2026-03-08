import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq

# ==========================================
# 1. Physical parameters
# ==========================================
Lx = 10.0      # room length in x (m)
Ly = 10.0      # room length in y (m)

Nx = 80        # grid points in x
Ny = 80        # grid points in y

dx = Lx/(Nx-1)
dy = Ly/(Ny-1)

c = 343.0      # speed of sound (m/s)

# ==========================================
# 2. Stability condition (CFL condition)
# ==========================================
C = 0.54                # chosen Courant number (<0.707 for stability)
dt = C * dx / c         # compute time step from CFL condition

T_total = 0.1
Nt = int(T_total/dt)

print("dx =",dx)
print("dt =",dt)
print("Courant number =",C)

# ==========================================
# 3. Create grid
# ==========================================
x = np.linspace(0,Lx,Nx)
y = np.linspace(0,Ly,Ny)

X,Y = np.meshgrid(x,y)

# ==========================================
# 4. Pressure array (3D: time,y,x)
# ==========================================
P = np.zeros((Nt,Ny,Nx))

# ==========================================
# 5. Initial condition (two Gaussian pulses)
# ==========================================
A = 100
sigma = 0.5

x1,y1 = 2,8
x2,y2 = 8,8

P[0,:,:] = A*np.exp(-((X-x1)**2+(Y-y1)**2)/(2*sigma**2)) + \
           A*np.exp(-((X-x2)**2+(Y-y2)**2)/(2*sigma**2))

# ==========================================
# 6. First time step (special start)
# ==========================================
P[1,1:-1,1:-1] = P[0,1:-1,1:-1] + (C**2/2)*(

    P[0,1:-1,2:] +
    P[0,1:-1,:-2] +
    P[0,2:,1:-1] +
    P[0,:-2,1:-1] -
    4*P[0,1:-1,1:-1]

)

# Neumann boundary conditions
P[1,0,:] = P[1,1,:]
P[1,-1,:] = P[1,-2,:]
P[1,:,0] = P[1,:,1]
P[1,:,-1] = P[1,:,-2]

# ==========================================
# 7. Time stepping loop
# ==========================================
for n in range(1,Nt-1):

    P[n+1,1:-1,1:-1] = (

        2*P[n,1:-1,1:-1] - P[n-1,1:-1,1:-1]
        + C**2*(

            P[n,1:-1,2:] +
            P[n,1:-1,:-2] +
            P[n,2:,1:-1] +
            P[n,:-2,1:-1] -
            4*P[n,1:-1,1:-1]

        )
    )

    # Neumann boundary
    P[n+1,0,:] = P[n+1,1,:]
    P[n+1,-1,:] = P[n+1,-2,:]
    P[n+1,:,0] = P[n+1,:,1]
    P[n+1,:,-1] = P[n+1,:,-2]

# ==========================================
# 8. Extract microphone signal
# ==========================================
mic_x = Nx//2
mic_y = Ny//2

mic_history = P[:,mic_y,mic_x]

time = np.linspace(0,T_total,Nt)

# ==========================================
# 9. Choose snapshot time
# ==========================================
plot_time = 0.0
step = int(plot_time/dt)

# ==========================================
# 10. FFT analysis
# ==========================================
N_fft = len(mic_history)

yf = fft(mic_history)
xf = fftfreq(N_fft,dt)

xf = xf[:N_fft//2]
yf = 2.0/N_fft*np.abs(yf[:N_fft//2])

# ==========================================
# 11. Plot results
# ==========================================
plt.style.use('bmh')

fig = plt.figure(figsize=(16,12))

# 3D surface
ax1 = fig.add_subplot(2,2,1,projection='3d')
surf = ax1.plot_surface(X,Y,P[step,:,:],cmap='viridis')
ax1.set_title("3D Acoustic Pressure")
ax1.set_xlabel("x (m)")
ax1.set_ylabel("y (m)")
ax1.set_zlabel("Pressure")

# Contour plot
ax2 = fig.add_subplot(2,2,2)
cont = ax2.contourf(X,Y,P[step,:,:],60,cmap='RdBu')
fig.colorbar(cont,ax=ax2)
ax2.set_title("Wave Interference Pattern")
ax2.set_xlabel("x (m)")
ax2.set_ylabel("y (m)")

# Microphone time signal
ax3 = fig.add_subplot(2,2,3)
ax3.plot(time,mic_history)
ax3.set_title("Microphone Pressure vs Time")
ax3.set_xlabel("Time (s)")
ax3.set_ylabel("Pressure")

# Frequency spectrum
ax4 = fig.add_subplot(2,2,4)
ax4.plot(xf,yf)
ax4.set_xlim(0,1500)
ax4.set_title("FFT Frequency Spectrum")
ax4.set_xlabel("Frequency (Hz)")
ax4.set_ylabel("Amplitude")

plt.tight_layout()
plt.show()



