from typing import List
import numpy as np
from scipy.spatial import KDTree
from . import pointcloud


def _kabsh(
    source: pointcloud.PointCloud,
    target: pointcloud.PointCloud,
    corres: List[int],
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


def compute_rmse(source: pointcloud.PointCloud, target_tree: KDTree) -> float:
    return sum(target_tree.query(source.points)[0]) / len(source)


def icp_registration(
    source: pointcloud.PointCloud,
    target: pointcloud.PointCloud,
    dist_thresh: float,
    initial_pos: np.ndarray = np.identity(4),
    tol: float = 1.0e-6,
    max_itr: int = 30,
) -> np.ndarray:
    target_tree = KDTree(target.points)
    cur_pc = pointcloud.PointCloud(source.points, field=pointcloud.PointXYZField())
    trans = initial_pos
    cur_pc.transform(trans)
    rmse = None
    for _ in range(max_itr):
        dd, ii = target_tree.query(cur_pc.points, k=1, distance_upper_bound=dist_thresh)
        ii = ii[~np.isinf(dd)]
        tr = _kabsh(cur_pc.points, target, ii)
        trans = np.dot(tr, trans)
        cur_pc.transform_(tr)
        tmp_rmse = compute_rmse(cur_pc, target_tree)
        if rmse is not None and abs(tmp_rmse - rmse) < tol:
            break
        rmse = tmp_rmse
    return trans