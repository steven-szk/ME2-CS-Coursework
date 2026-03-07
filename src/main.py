import numpy as np
import matplotlib.pyplot as plt

#const
Lx, Ly = 10.0, 10.0     # size of room
Nx, Ny = 80, 80         # num of grid points
dx = Lx / (Nx - 1)      
dy = dx    
c = 343.0                 # speed of sound

dt = 0.0002 # temporal increment
tend = 0.1  # temporal span
Nt = int(tend / dt)
C= c * dt / dx  # Courant number

#create matrix:
x = np.linspace(0, Lx, Nx)
y = np.linspace(0, Ly, Ny)
X, Y = np.meshgrid(x, y) #This is so easier to cal IC
P = np.zeros((Nt,Nx,Ny))

A = 100.0     # amplitude of the pulse
sigma = 0.5   # width of the pulse

# position of the speaker
xc1, yc1 = 2.0, 8.0  # top left corner
xc2, yc2 = 8.0, 8.0  # top right corner

#========================================================================

#INITIAL CONDITION
P[0, :, :] = (A * np.exp(-((X - xc1)**2 + (Y - yc1)**2) / (2 * sigma**2)) + 
              A * np.exp(-((X - xc2)**2 + (Y - yc2)**2) / (2 * sigma**2)))
for x in range(Nx):
    for y in range(Ny):
        P[1, x, y] = P[0, x, y] + (C**2 / 2.0) * (
            P[0, min(x+1,Nx-1), y] + P[0, max(x-1,0), y] + P[0, x, min(y+1,Ny-1)] + P[0, x, max(y-1,0)] - 4 * P[0, x, y]
        )

for n in range(2, Nt): #for all time steps except the first two
    for x in range(1, Nx - 1): # for all interior points
        for y in range(1, Ny - 1):
            # Update the pressure field using the finite difference scheme
            P[n, x, y] = 2 * P[n-1, x, y] - P[n-2, x, y] + C**2 * (
                P[n-1, x, y+1] + P[n-1, x, y-1] + P[n-1, x+1, y] + P[n-1, x-1, y] - 4 * P[n-1, x, y]
            )
    
    # NUEMANN BOUNDARY CONDITION (rigid walls, zero normal gradient)
    P[n, 0, :] = P[n, 1, :]       # Bottom wall
    P[n, -1, :] = P[n, -2, :]     # Top wall
    P[n, :, 0] = P[n, :, 1]       # Left wall
    P[n, :, -1] = P[n, :, -2]     # Right wall

