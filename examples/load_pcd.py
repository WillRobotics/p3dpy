import p3dpy as pp

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

pc = pp.io.load_pcd('data/bunny_lzf.pcd')
pc = pp.filter.remove_invalid_points(pc)
pc = pp.filter.voxel_grid_filter(pc, 0.02)
pc.compute_normals(0.1)
print(pc.data)

fig = plt.figure(figsize = (8, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(pc.points[:, 0], pc.points[:, 1], pc.points[:, 2], s = 40, c = "blue")
plt.show()
