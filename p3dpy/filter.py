import numpy as np
from . import pointcloud
from scipy.spatial import KDTree


def radius_outlier_removal(pc: pointcloud.PointCloud, radius: float, neighbor_counts: int) -> pointcloud.PointCloud:
    tree = KDTree(pc.points)
    mask = [len(tree.query_ball_point(p, radius)) > neighbor_counts for p in pc.points]
    return pointcloud.PointCloud(pc._points[mask, :], pc._field)


def statistical_outlier_removal(pc: pointcloud.PointCloud, k_neighbors: int, std_ratio: float) -> pointcloud.PointCloud:
    tree = KDTree(pc.points)
    dd, ii = tree.query(pc.points, k_neighbors)
    avg_d = dd.mean(axis=1)
    std_d = dd.std(axis=1)
    dist_thresh = avg_d.mean() + std_ratio * std_d
    return pointcloud.PointCloud(pc._points[avg_d < dist_thresh, :], pc._field)