import p3dpy as pp

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


pc = pp.io.load_pcd('data/test0.pcd')
pc = pp.filter.remove_invalid_points(pc)
pc = pp.filter.voxel_grid_filter(pc, 0.01)
print(pc.points)
plane, mask = pp.segmentation.segmentation_plane(pc)
print(plane, sum(mask))

fig = plt.figure(figsize = (8, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(pc.points[:, 0], pc.points[:, 1], pc.points[:, 2], s = 40, c = "blue")
plt.show()