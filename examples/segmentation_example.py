import p3dpy as pp

import numpy as np
from scipy.spatial import ConvexHull
from sklearn.cluster import DBSCAN
import transforms3d as t3d


def calc_triangle_area(p0, p1, p2):
    return 0.5 * (p0[0] * (p1[1] - p2[1]) + p1[0] * (p2[1] - p0[1]) + p2[0] * (p0[1] - p1[1]))


def calc_convexhull_area(points):
    hull = ConvexHull(points[:, :2])
    hull_points = points[hull.vertices, :2]
    hull_points = np.r_[hull_points, [hull_points[0]]]
    hull_mean = hull_points.mean(axis=0)
    area = 0
    for i in range(len(hull_points) - 1):
        area += calc_triangle_area(hull_mean, hull_points[i], hull_points[i + 1])
    return area


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

# Cluster objects
db = DBSCAN(eps=0.1).fit(not_plane_pts)
labels = db.labels_
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
print("Number of class:", n_clusters)
mask_0 = labels == 0
mask_1 = labels == 1

# Draw results
client = pp.VizClient()
res1 = client.post_pointcloud(pp.PointCloud(np.c_[plane_pts, np.tile([0.0, 0.0, 1.0], (len(plane_pts), 1))],
                                            pp.pointcloud.PointXYZRGBField()),
                              'plane')
res2 = client.post_pointcloud(pp.PointCloud(np.c_[not_plane_pts[mask_0], np.tile([0.0, 1.0, 0.0], (len(not_plane_pts[mask_0]), 1))],
                                            pp.pointcloud.PointXYZRGBField()),
                              'obj1')
res3 = client.post_pointcloud(pp.PointCloud(np.c_[not_plane_pts[mask_1], np.tile([1.0, 0.0, 1.0], (len(not_plane_pts[mask_1]), 1))],
                                            pp.pointcloud.PointXYZRGBField()),
                              'obj2')
plane_area = calc_convexhull_area(plane_pts[:, :2])
obj0_area = calc_convexhull_area(not_plane_pts[mask_0, :2])
obj1_area = calc_convexhull_area(not_plane_pts[mask_1, :2])
client.add_log(f"Plane Area: {plane_area:.3f}")
client.add_log(f"Obj1 Area: {obj0_area:.3f}")
client.add_log(f"Obj2 Area: {obj1_area:.3f}")
print(res1)

pp.vizloop()
