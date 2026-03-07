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

#BOUNDARY CONDITION





#INITIAL CONDITION
P[0, :, :] = (A * np.exp(-((X - xc1)**2 + (Y - yc1)**2) / (2 * sigma**2)) + 
              A * np.exp(-((X - xc2)**2 + (Y - yc2)**2) / (2 * sigma**2)))


for n in range(1,Nt):
    P[n,0,0]=... #in terms of T in prev vals
    P[n,-1]=...
    for i in range(1, Nx-1):
        P[n ,i]=.... #in terms of T in previous time steps

