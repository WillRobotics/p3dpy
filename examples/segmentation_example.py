import p3dpy as pp

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


pc = pp.io.load_pcd('data/test0.pcd')
pc = pp.filter.remove_invalid_points(pc)
pc = pp.filter.voxel_grid_filter(pc, 0.01)
print(pc.points)
plane, mask = pp.segmentation.segmentation_plane(pc, dist_thresh=0.01)
print(plane, sum(mask))

fig = plt.figure(figsize = (8, 8))
ax = fig.add_subplot(111, projection='3d')
plane_pts = pc.points[mask, :]
not_plane_pts = pc.points[~mask, :]
ax.scatter(plane_pts[:, 0], plane_pts[:, 1], plane_pts[:, 2], s=20, c="blue")
ax.scatter(not_plane_pts[:, 0], not_plane_pts[:, 1], not_plane_pts[:, 2], s=20, c="red")
plt.show()