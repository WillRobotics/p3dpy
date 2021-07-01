from typing import List

import numpy as np
from scipy.spatial import cKDTree
from . import pointcloud


def _kabsh(
    source: pointcloud.PointCloud,
    target: pointcloud.PointCloud,
    corres: np.ndarray,
) -> np.ndarray:
    src_avg = source.points.mean(axis=0)
    trg_avg = target.points[corres, :].mean(axis=0)
    hh = np.dot((source.points - src_avg).T, target.points[corres, :] - trg_avg)
    hh /= corres.shape[0]
    u, s, vh = np.linalg.svd(hh, full_matrices=True)
    ss = np.identity(3)
    ss[2, 2] = np.linalg.det(np.dot(u, vh.T))
    tr = np.identity(4)
    tr[:3, :3] = np.dot(np.dot(vh.T, ss), u.T)
    tr[:3, 3] = trg_avg
    tr[:3, 3] -= np.dot(tr[:3, :3], src_avg)
    return tr


def _compute_rmse(source_pts: np.ndarray, target_tree: cKDTree, top_k: int = -1) -> float:
    if top_k <= 0:
        return sum(target_tree.query(source_pts)[0]) / source_pts.shape[0]
    else:
        return sum(sorted(target_tree.query(source_pts)[0])[:top_k]) / source_pts.shape[0]


def compute_rmse(source_pts: np.ndarray, target_pts: np.ndarray, top_k: int = -1) -> float:
    target_tree = cKDTree(target_pts)
    return _compute_rmse(source_pts, target_tree, top_k)


def icp_registration(
    source: pointcloud.PointCloud,
    target: pointcloud.PointCloud,
    dist_thresh: float,
    initial_pos: np.ndarray = np.identity(4),
    tol: float = 1.0e-6,
    max_itr: int = 30,
) -> np.ndarray:
    target_tree = cKDTree(target.points)
    cur_pc = pointcloud.PointCloud(source.points, field=pointcloud.PointXYZField())
    trans = initial_pos
    cur_pc.transform(trans)
    rmse = None
    for _ in range(max_itr):
        dd, ii = target_tree.query(cur_pc.points, k=1, distance_upper_bound=dist_thresh)
        ii = ii[~np.isinf(dd)]
        tr = _kabsh(cur_pc, target, ii)
        trans = np.dot(tr, trans)
        cur_pc.transform_(tr)
        tmp_rmse = _compute_rmse(cur_pc.points, target_tree)
        if rmse is not None and abs(tmp_rmse - rmse) < tol:
            break
        rmse = tmp_rmse
    return trans