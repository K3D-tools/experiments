import numpy as np
import pandas as pd
import k3d
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

#from resources import cubic, fcc, distance


def cubic(sx, sy, sz, nx, ny, nz):
    X, Y, Z = np.arange(sx, nx), np.arange(sy, ny), np.arange(sz, nz)
    x, y, z = np.meshgrid(X, Y, Z)
    xyz = np.vstack([np.ravel(x), np.ravel(y), np.ravel(z)]).T.astype(np.float32)
    
    return xyz

def fcc(sx, sy, sz, nx, ny, nz):
    X, Y, Z = np.arange(sx, nx), np.arange(sy, ny), np.arange(sz, nz)
    x, y, z = np.meshgrid(X, Y, Z)

    xyz = np.vstack([np.ravel(x), np.ravel(y), np.ravel(z)]).T
    xyz1 = xyz + np.array([0.5, 0.5,0])
    xyz2 = xyz + np.array([0.5, 0, 0.5])
    xyz3 = xyz + np.array([0, 0.5, 0.5])

    XYZ = np.vstack([xyz, xyz1, xyz2, xyz3]).astype(np.float32)

    return XYZ

def distance(a, b):
    return np.sqrt(np.sum((a - b)**2, axis=1))

### FIRST DRAWING
cubic_system = cubic(0,0,0,3,3,3)
red_point = k3d.points(cubic_system[0], point_size=0.2, color=0xff0000)
system = k3d.points(cubic_system[1:], point_size=0.2, color=0)
a_vec = k3d.vectors(red_point.positions, cubic_system[3], head_size=1.5, labels=['a'])
b_vec = k3d.vectors(red_point.positions, cubic_system[9], head_size=1.5, labels=['b'])
c_vec = k3d.vectors(red_point.positions, cubic_system[1], head_size=1.5, labels=['c'])
ab_vec = k3d.vectors(red_point.positions, cubic_system[3]+2*cubic_system[9], head_size=1.5, labels=['a+2b'])

plot1 = k3d.plot()
plot1 += red_point + system + a_vec + b_vec + c_vec + ab_vec


### SECOND DRAWING
cubic_system = cubic(0,0,0,3,3,3)
mask = np.arange(cubic_system.shape[0])
chosen = [0,1,3,4,21,22,24,25]

red_points = k3d.points(cubic_system[chosen], point_size=0.2, color=0xff0000)
system = k3d.points(np.delete(cubic_system, chosen,axis=0), point_size=0.2, color=0)
ab_vec = k3d.vectors(red_point.positions, cubic_system[3]+2*cubic_system[9], head_size=1.5, labels=['a+2b'])

plot2 = k3d.plot()
plot2 += red_points + system + ab_vec


### THIRD DRAWING
cubic_system = cubic(0,0,0,5,5,5)
system = k3d.points(cubic_system, point_size=0.15, color=0)
gbpoints = k3d.points(cubic_system[[37, 92]], point_size=0.15, colors=[0x0000ff, 0x00ff00])

data = pd.DataFrame(np.concatenate([cubic_system, cubic_system-cubic_system[37], cubic_system-cubic_system[92]], axis=1), 
                    columns=['x', 'y', 'z','dx37','dy37','dz37','dx92','dy92','dz92'])

data['distance_from_37'] = np.sqrt((data[['dx37','dy37','dz37']]**2).sum(axis=1))
data['distance_from_92'] = np.sqrt((data[['dx92','dy92','dz92']]**2).sum(axis=1))
data = data.sort_values(by='distance_from_37')

nn37 = (data.where(data['distance_from_37'] == 1.0)).dropna()
nn92 = (data.where(data['distance_from_92'] == 1.0)).dropna()
tr37 = (data.where(data['distance_from_37'] == np.sqrt(3))).dropna()
tr92 = (data.where(data['distance_from_92'] == np.sqrt(3))).dropna()

lines37 = np.concatenate([np.array(nn37[['x','y','z']]), np.resize(np.array(data.loc[37][['x','y','z']]), (6,3))], axis=1)
lines92 = np.concatenate([np.array(nn92[['x','y','z']]), np.resize(np.array(data.loc[92][['x','y','z']]), (6,3))], axis=1)
linestr37 = np.concatenate([np.array(tr37[['x','y','z']]), np.resize(np.array(data.loc[37][['x','y','z']]), (8,3))], axis=1)
linestr92 = np.concatenate([np.array(tr92[['x','y','z']]), np.resize(np.array(data.loc[92][['x','y','z']]), (8,3))], axis=1)

blue_lines = k3d.line(lines37, shader='mesh', width=0.02)
green_lines = k3d.line(lines92, shader='mesh', width=0.02, color=0x00ff00)
red_lines = k3d.line(linestr37, shader='mesh', width=0.02, color=0xff0000)
magenta_lines = k3d.line(linestr92, shader='mesh', width=0.02, color=0xff00ff)

plot3 = k3d.plot()
plot3 += system + gbpoints + blue_lines + green_lines + red_lines + magenta_lines

### FOURTH DRAWING
np.random.seed(137)
gas_data = np.random.uniform(0,5, size=(5**3, 3)).astype(np.float32)

gas = k3d.points(gas_data, point_size=0.15, color=0)
bgpoints = k3d.points(gas_data[[13,17]], point_size=0.15, colors=[0x00ff00, 0x0000ff])

gas_df = pd.DataFrame(np.concatenate([gas_data, gas_data-gas_data[13], gas_data-gas_data[17]], axis=1), 
                    columns=['x', 'y', 'z','dx13','dy13','dz13','dx17','dy17','dz17'])

gas_df['distance_from_13'] = np.sqrt((gas_df[['dx13','dy13','dz13']]**2).sum(axis=1))
gas_df['distance_from_17'] = np.sqrt((gas_df[['dx17','dy17','dz17']]**2).sum(axis=1))

gas_df = gas_df.sort_values(by='distance_from_13')
green_lines = np.concatenate([np.array(gas_df[1:9][['x','y','z']]), np.resize(np.array(gas_df.loc[13][['x','y','z']]), (8,3))], axis=1)

gas_df = gas_df.sort_values(by='distance_from_17')
blue_lines = np.concatenate([np.array(gas_df[1:9][['x','y','z']]), np.resize(np.array(gas_df.loc[17][['x','y','z']]), (8,3))], axis=1)

glines = k3d.line(green_lines, shader='mesh', width=0.02, color=0x00ff00)
blines = k3d.line(blue_lines, shader='mesh', width=0.02, color=0x0000ff)

plot4 = k3d.plot()
plot4 += gas + bgpoints + glines + blines

### FIFTH DRAWING
cubic_system = cubic(-3,-3,-3,6,6,6)
crystal = fcc(-3,-3,-3,6,6,6)

CUBIC = k3d.points(cubic_system, point_size=0.1, color=0xff0000)
CRYSTAL = k3d.points(crystal, point_size=0.05, color=0)

plot5 = k3d.plot()
plot5 += CUBIC + CRYSTAL

### FIRST PLOT
fcc_system = fcc(0,0,0,3,3,3)
fcc_df = pd.DataFrame(fcc_system, columns=['x','y','z'])
crystal_df = pd.DataFrame(crystal, columns=['x','y','z'])

crystal_df['d0'] = distance(crystal_df, fcc_df.iloc[0])

crystal_df = crystal_df.sort_values(by='d0')
zr = crystal_df['d0'].value_counts()

xs = zr.sort_values().index[1:] 
ys = zr.sort_values().values[1:]

fig1 = plt.figure(figsize=(10,6))

for i,x in enumerate(xs):
    plt.vlines(x, ymin=0, ymax=ys[i])

plt.xlim(0,4)
plt.xlabel('r')
plt.ylabel('z')
plt.title('fcc rdf')
plt.grid()


### SECOND PLOT 
cubic_system = cubic(0,0,0,3,3,3)
cubic_crystal = cubic(-3,-3,-3,6,6,6)

cubic_df = pd.DataFrame(cubic_system, columns=['x','y','z'])
cubic_crystal_df = pd.DataFrame(cubic_crystal, columns=['x','y','z'])

cubic_crystal_df['d0'] = distance(cubic_crystal_df, cubic_df.iloc[0])

cubic_crystal_df = cubic_crystal_df.sort_values(by='d0')
cubic_zr = cubic_crystal_df['d0'].value_counts()

cubic_xs = cubic_zr.sort_values().index[1:] 
cubic_ys = cubic_zr.sort_values().values[1:]

fig2 = plt.figure(figsize=(10,6))

for i,x in enumerate(cubic_xs):
    plt.vlines(x, ymin=0, ymax=cubic_ys[i])

plt.xlim(0,4)
plt.xlabel('r')
plt.ylabel('z')
plt.title('cubic rdf')
plt.grid()

### THIRD PLOT
fig3 = plt.figure(figsize=(10,6))

for i,x in enumerate(xs):
    plt.vlines(x, ymin=0, ymax=ys[i])

for i,x in enumerate(cubic_xs):
    plt.vlines(x, ymin=0, ymax=cubic_ys[i], colors='r')

plt.xlim(0,4)
plt.xlabel('r')
plt.ylabel('z')
plt.grid()


legend_elements = [Line2D([0], [0], color='k', lw=4, label='Line'), Line2D([0], [0], color='r', lw=4, label='Line')]
plt.legend(loc='upper left', labels=['fcc', 'regularna, fcc'], handles=legend_elements)

### SIXTH DRAWING
np.random.seed(137)
gas_data = np.random.uniform(-8,8, size=(2**12, 3)).astype(np.float32)
green_gas_data = gas_data[distance(gas_data, np.array([0,0,0])) < 3]

gas = k3d.points(gas_data, point_size=0.1, color=0)
green_gas = k3d.points(green_gas_data, point_size=0.1, color=0x00ff00)
plot6 = k3d.plot(grid_visible=False, camera_auto_fit=False)
plot6 += gas + green_gas


gas_df = pd.DataFrame(gas_data, columns=['x','y','z'])
green_gas_df = pd.DataFrame(green_gas_data, columns=['x','y','z'])

r0, r, dr = 0, 8, 0.01
array_size = np.int((r-r0)/dr)
counts = np.zeros(array_size)
r_space = np.arange(r0+dr, r+dr, dr)

for i in range(10):
    diffs = distance(gas_df, green_gas_df.iloc[i])
    diffs = diffs.sort_values()

    for b in range(array_size):
        counts[b] += ( np.sum((b*dr <= diffs[1:]) & (diffs[1:] <= b*dr + dr)) )

        
### FOURTH PLOT
fig4 = plt.figure(figsize=(10,6))
plt.grid()
plt.title('gas rdf')
plt.xlabel('r')
plt.ylabel('z')
plt.plot(r_space, counts, 'r--')

### FIFTH PLOT
normed_counts = np.copy(counts)
normed_counts[1:] /= 4/3 * np.pi *(r_space[1:]**3 - r_space[:-1]**3)

fig5 = plt.figure(figsize=(10, 6))
plt.grid()
plt.title('normed gas rdf')
plt.xlabel('$r/V_{shell}$')
plt.ylabel('$z$')
plt.plot(r_space, normed_counts, 'r--')

