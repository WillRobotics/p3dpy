from collections import defaultdict
from typing import DefaultDict, Tuple

import numpy as np
from scipy.spatial import cKDTree

from . import pointcloud


def remove_invalid_points(pc: pointcloud.PointCloud) -> pointcloud.PointCloud:
    return pointcloud.PointCloud(pc.finalize().data[np.isfinite(pc.data).any(axis=1)], field=pc.field)


def random_sampling(pc: pointcloud.PointCloud, n_sample: int) -> pointcloud.PointCloud:
    return pointcloud.PointCloud(pc.finalize().data[np.random.randint(len(pc), size=n_sample), :], field=pc.field)


def pass_through_filter(
    pc: pointcloud.PointCloud, min_lim: float, max_lim: float, axis: int = 0
) -> pointcloud.PointCloud:
    axis_pc = pc.points[:, axis]
    return pointcloud.PointCloud(pc.data[(axis_pc >= min_lim) & (axis_pc <= max_lim), :], field=pc.field)


def radius_outlier_removal(pc: pointcloud.PointCloud, radius: float, neighbor_counts: int) -> pointcloud.PointCloud:
    tree = cKDTree(pc.points)
    mask = [len(tree.query_ball_point(p, radius)) > neighbor_counts for p in pc.points]
    return pointcloud.PointCloud(pc.data[mask, :], pc.field)


def statistical_outlier_removal(
    pc: pointcloud.PointCloud, k_neighbors: int, std_ratio: float
) -> pointcloud.PointCloud:
    tree = cKDTree(pc.points)
    dd, _ = tree.query(pc.points, k_neighbors)
    avg_d = dd.mean(axis=1)
    std_d = dd.std(axis=1)
    dist_thresh = avg_d.mean() + std_ratio * std_d
    return pointcloud.PointCloud(pc.data[avg_d < dist_thresh, :], pc.field)


def voxel_grid_filter(pc: pointcloud.PointCloud, voxel_size: float) -> pointcloud.PointCloud:
    """Voxel grid filter

    Parameters
    ----------
    pc: pointcloud.PointCloud
        Input point cloud.
    voxel_size: float
        The length of one side of the voxel.

    Returns
    -------
    pointcloud.PointCloud
        Voxel-averaged point cloud.
    """

    class SumPoints:
        def __init__(self, dim: int = pc.data.shape[1]):
            self._num_points = 0
            self._point = np.zeros(dim)

        def add_point(self, point: np.ndarray):
            self._point += point
            self._num_points += 1

        def get_mean(self):
            return self._point / self._num_points

    min_bound = pc.points.min(axis=0) - voxel_size * 0.5
    voxel_dic: DefaultDict[Tuple[int, int, int], SumPoints] = defaultdict(SumPoints)

    def func(i):
        coord = tuple(
            np.floor((pc.data[i, pc.field.slices["point"]] - min_bound) / voxel_size).astype(np.int32).tolist()
        )
        voxel_dic[coord].add_point(pc.data[i])

    func_v = np.frompyfunc(func, 1, 0)
    func_v(np.arange(len(pc)))
    return pointcloud.PointCloud([v.get_mean() for v in voxel_dic.values()], pc.field)
