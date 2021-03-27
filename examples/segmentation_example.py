import p3dpy as pp

import numpy as np
import transforms3d as t3d
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


pc = pp.io.load_pcd('data/test0.pcd')
pc = pp.filter.remove_invalid_points(pc)
pc = pp.filter.voxel_grid_filter(pc, 0.01)
print(pc.points)
plane, mask = pp.segmentation.segmentation_plane(pc, dist_thresh=0.01)
print(plane, sum(mask))
axis = np.array([-plane[1], plane[0], 0.0])
axis /= np.linalg.norm(axis)
angle = np.arccos(plane[2] / np.linalg.norm(plane[:3]))
trans = np.identity(4)
trans[:3, :3] = t3d.axangles.axangle2mat(axis, angle)

fig = plt.figure(figsize = (8, 8))
ax = fig.add_subplot(111, projection='3d')
pc.transform_(trans.T)
plane_pts = pc.points[mask, :]
not_plane_pts = pc.points[~mask, :]
ax.scatter(plane_pts[:, 0], plane_pts[:, 1], plane_pts[:, 2], s=20, c="blue")
ax.scatter(not_plane_pts[:, 0], not_plane_pts[:, 1], not_plane_pts[:, 2], s=20, c="red")
plt.show()