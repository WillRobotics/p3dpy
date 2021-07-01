from typing import List, Tuple, Union
import numpy as np
from . import pointcloud


class RANSACResult(object):
    def __init__(self, fitness: float = 0.0, inlier_rmse: float = 0.0) -> None:
        self._fitness = fitness
        self._inlier_rmse = inlier_rmse


def _compute_plane(pc: pointcloud.PointCloud, inliers: Union[np.ndarray, List[int]]) -> np.ndarray:
    cands = pc.points[inliers, :]
    if len(cands) == 3:
        e0 = cands[1] - cands[0]
        e1 = cands[2] - cands[0]
        abc = np.cross(e0, e1)
        norm = np.linalg.norm(abc)
        if norm == 0:
            return np.zeros(4)
        abc /= norm
        return np.r_[abc, -np.dot(abc, cands[0])]
    elif len(cands) > 3:
        mean = cands.mean(axis=0)
        cands -= mean
        u, s, v = np.linalg.svd(cands)
        nv = v[-1, :]
        param = np.r_[nv, -np.dot(mean, nv)]
        return param
    else:
        raise ValueError("The number of inliers must be 3 or more.")


def _evaluate_ransac(
    pc: pointcloud.PointCloud, plane: np.ndarray, dist_thresh: float
) -> Tuple[RANSACResult, np.ndarray, float]:
    dists = np.abs(np.dot(np.c_[pc.points, np.ones(len(pc))], plane))
    mask = dists < dist_thresh
    dists = dists[mask]
    error = dists.sum()
    if len(dists) == 0:
        return RANSACResult(), mask, error
    else:
        return RANSACResult(len(dists) / len(pc), error / np.sqrt(len(dists))), mask, error


def segmentation_plane(
    pc: pointcloud.PointCloud, dist_thresh: float = 0.1, ransac_n: int = 3, num_iter: int = 100
) -> Tuple[np.ndarray, np.ndarray]:
    res = RANSACResult()
    best_plane = np.zeros(4)
    for n in range(num_iter):
        inliers = np.random.choice(len(pc), ransac_n, replace=False)
        plane = _compute_plane(pc, inliers)
        tmp_res, mask, error = _evaluate_ransac(pc, plane, dist_thresh)
        if tmp_res._fitness > res._fitness or (
            res._fitness == tmp_res._fitness and tmp_res._inlier_rmse < res._inlier_rmse
        ):
            res = tmp_res
            best_plane = plane

    _, mask, _ = _evaluate_ransac(pc, best_plane, dist_thresh)
    try:
        best_plane = _compute_plane(pc, mask)
    except ValueError:
        pass
    return best_plane, mask