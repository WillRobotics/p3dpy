from typing import List
import numpy as np
from . import pointcloud


class RANSACResult(object):
    def __init__(self, fitness: float = 0.0, inlier_rmse: float = 0.0):
        self._fitness = fitness
        self._inlier_rmse = inlier_rmse


def _compute_plane(pc: pointcloud.PointCloud, inliers: List[int]) -> np.ndarray:
    cands = pc.points[inliers, :]
    mean = cands.mean(axis=0)
    cands0 -= mean
    u, s, v = np.linalg.svd(cands0)
    nv = v[-1, :]
    param = np.r_[nv, -np.dot(mean, nv)]
    return param


def _evaluate_ransac(pc: pointcloud.PointCloud, plane: np.ndarray, dist_thresh: float) -> (RANSACResult, np.ndarray, float):
    dists = np.abs(np.dot(np.c_[pc.points, np.ones(len(pc))], plane))
    mask = dists < dist_thresh
    dists = dists[mask]
    error = dists.sum()
    if len(dists) == 0:
        return RANSACResult(), mask, error
    else:
        return RANSACResult(len(dists) / len(pc), error / np.sqrt(len(dists))), mask, error


def segmentation_plane(pc: pointcloud.PointCloud, dist_thresh: float = 0.1, ransac_n: int = 3, num_iter: int = 100) -> (np.ndarray, np.ndarray):
    res = RANSACResult()
    best_plane = np.zeros(4)
    for n in range(num_iter):
        inliers = np.random.choice(len(pc), ransac_n, replace=False)
        plane = _compute_plane(pc, inliers)
        tmp_res, mask, error = _evaluate_ransac(pc, plane, dist_thresh)
        if tmp_res._fitness > res._fitness or\
           (res._fitness == tmp_res._fitness and\
            tmp_res._inlier_rmse < res._inlier_rmse):
            res = tmp_res
            best_plane = plane

    best_plane = _compute_plane(pc, mask)
    _, mask, _ = _evaluate_ransac(pc, best_plane, dist_thresh)
    return best_plane, mask