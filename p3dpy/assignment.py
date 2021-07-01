from typing import Callable, Deque, List, Optional
from collections import deque

from typing import Any, Deque
import numpy as np
from scipy.optimize import linear_sum_assignment

from . import pointcloud


class ClusterAssigner:
    class Cluster:
        def __init__(self, name: str, pointcloud: pointcloud.PointCloud, feature: np.ndarray) -> None:
            self.name = name
            self.pointcloud = pointcloud
            self.feature = feature

    def __init__(self, dist_fn: Callable, dist_thresh: float) -> None:
        self._db: Deque[Any] = deque()
        self._dist_fn = dist_fn
        self._dist_thresh = dist_thresh

    @property
    def db(self) -> Deque[Cluster]:
        return self._db

    def register_and_assign(self, clusters: List[Cluster]) -> List[int]:
        if len(self._db) > 0 and len(clusters) > 0:
            dist_mat = np.zeros((len(clusters), len(self._db)))
            for i, d in enumerate(self._db):
                for j, c in enumerate(clusters):
                    dist_mat[j, i] = self._dist_fn(d, c)
            row_ind, col_ind = linear_sum_assignment(dist_mat)
            dists = dist_mat[row_ind, col_ind]
            for j, d in enumerate(dists):
                if d > self._dist_thresh:
                    col_ind[j] = len(self._db)
                    self._db.append(clusters[j])
            return col_ind
        else:
            self._db.extend(clusters)
            return [i for i in range(len(clusters))]
