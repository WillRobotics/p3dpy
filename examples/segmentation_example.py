import p3dpy as pp

import numpy as np
from scipy.spatial import ConvexHull
from sklearn.cluster import DBSCAN
import transforms3d as t3d


pp.vizspawn()

# Segment plane
pc = pp.io.load_pcd('data/test0.pcd')
pc = pp.filter.remove_invalid_points(pc)
pc = pp.filter.voxel_grid_filter(pc, 0.01)
print(pc.points)
plane, mask = pp.segmentation.segmentation_plane(pc, dist_thresh=0.01)
print(plane, sum(mask))

# Rotate to plane coordinate
axis = np.array([-plane[1], plane[0], 0.0])
axis /= np.linalg.norm(axis)
angle = np.arccos(plane[2] / np.linalg.norm(plane[:3]))
trans = np.identity(4)
trans[:3, :3] = t3d.axangles.axangle2mat(axis, angle)
pc.transform_(trans.T)

# Divide a plane and objects
plane_pts = pc.points[mask, :]
not_plane_pts = pc.points[~mask, :]
hull = ConvexHull(plane_pts[:, :2])

# Cluster objects
db = DBSCAN(eps=0.1).fit(not_plane_pts)
labels = db.labels_
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
print("Number of class:", n_clusters)
mask_0 = labels == 0
mask_1 = labels == 1

result_pts = np.r_[np.c_[plane_pts, np.tile([0.0, 0.0, 1.0], (len(plane_pts), 1))],
                   np.c_[not_plane_pts[mask_0], np.tile([0.0, 1.0, 0.0], (len(not_plane_pts[mask_0]), 1))],
                   np.c_[not_plane_pts[mask_1], np.tile([1.0, 0.0, 1.0], (len(not_plane_pts[mask_1]), 1))]]

# Draw results
result_pc = pp.PointCloud(result_pts, pp.pointcloud.PointXYZRGBField())
client = pp.VizClient()
res = client.post_pointcloud(result_pc, 'test')
print(res)

pp.vizloop()
