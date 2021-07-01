from typing import List, Tuple

import numpy as np
from scipy.spatial import cKDTree

from . import pointcloud


PST_RAD_45 = 0.78539816339744830961566084581988
PST_RAD_90 = 1.5707963267948966192313216916398
PST_RAD_135 = 2.3561944901923449288469825374596
PST_RAD_PI_7_8 = 2.7488935718910690836548129603691


def _compute_shot_lrfs(pc: pointcloud.PointCloud, neighbors: List[Tuple], radius: float) -> np.ndarray:
    def shot_lrf(i, nb):
        n_nb = len(nb)
        if n_nb < 5:
            np.zeros((3, 3))
        q = pc.points[nb, :] - pc.points[i, :]
        dists = np.linalg.norm(q)
        w = radius - dists
        cov = np.dot(np.multiply(q.T, w), q)
        cov /= w.sum()
        _, evec = np.linalg.eigh(cov)
        z = evec[:, 0]
        x = evec[:, 2]
        n_px = np.count_nonzero(np.dot(q, x) >= 0)
        n_pz = np.count_nonzero(np.dot(q, z) >= 0)
        if n_px < n_nb - n_px:
            x *= -1.0
        if n_pz < n_nb - n_pz:
            z *= -1.0
        return np.c_[x, np.cross(z, x), z]

    shot_lrf_v = np.frompyfunc(shot_lrf, 2, 1)
    lrfs = shot_lrf_v(np.arange(len(neighbors)), np.array(neighbors, dtype=object))
    return lrfs


# https://github.com/PointCloudLibrary/pcl/blob/master/features/include/pcl/features/impl/shot.hpp#L237
def compute_shot_descriptors(
    pc: pointcloud.PointCloud,
    radius: float,
    min_neighbors: int = 5,
    n_grid_sectors: int = 32,
) -> np.ndarray:
    """Compute shot descriptors

    Parameters
    ----------
    pc: pointcloud.PointCloud
        Input point cloud.
    radius: float
        Radius for calculating the feature value of a point.
    min_neighbors: int
        Number of neighbors needed to compute the features.
    n_grid_sectors: int
        Number of sectors.

    Returns
    -------
    np.ndarray
        Shot descriptors.
    """
    if not pc.has_field("normal"):
        raise ValueError("Given pointcoud doesn't have normals.")

    n_bins = 10
    radius1_2 = radius * 0.5
    radius3_4 = radius * 3.0 / 4.0
    radius1_4 = radius * 0.25
    tree = cKDTree(pc.points)
    neighbors = [tree.query_ball_point(p, radius) for p in pc.points]
    lrfs = _compute_shot_lrfs(pc, neighbors, radius)

    def shot_fn(i, nb):
        shots = np.zeros(n_grid_sectors * (n_bins + 1))
        n_nb = len(nb)
        if n_nb < min_neighbors:
            return shots
        for n in nb:
            q = pc.points[n, :] - pc.points[i, :]
            dist = np.linalg.norm(q)
            if dist == 0:
                continue

            cos_desc = min(max(np.dot(lrfs[i][:, 2], pc.normals[i, :]), -1.0), 1.0)
            bindist = ((1.0 + cos_desc) * n_bins) / 2.0

            x_lrf = np.dot(q, lrfs[i][:, 0])
            y_lrf = np.dot(q, lrfs[i][:, 1])
            z_lrf = np.dot(q, lrfs[i][:, 2])
            if abs(x_lrf) < 1e-30:
                x_lrf = 0.0
            if abs(y_lrf) < 1e-30:
                y_lrf = 0.0
            if abs(z_lrf) < 1e-30:
                z_lrf = 0.0
            bit4 = True if (y_lrf > 0 or (y_lrf == 0 and x_lrf < 0)) else False
            bit3 = not bit4 if (x_lrf > 0 or (x_lrf == 0 and y_lrf > 0)) else bit4
            bit3, bit4 = int(bit3), int(bit4)
            desc_index = (bit4 << 3) + (bit3 << 2)
            desc_index = desc_index << 1
            if x_lrf * y_lrf > 0 or x_lrf == 0:
                desc_index += 0 if abs(x_lrf) >= abs(y_lrf) else 4
            else:
                desc_index += 4 if abs(x_lrf) > abs(y_lrf) else 0
            desc_index += 1 if z_lrf > 0 else 0
            # RADII
            desc_index += 2 if dist > radius1_2 else 0
            step_index = int(np.ceil(bindist - 0.5) if bindist < 0.0 else np.floor(bindist + 0.5))
            volume_index = desc_index * (n_bins + 1)
            bindist -= step_index
            init_weight = 1.0 - abs(bindist)
            if bindist > 0:
                shots[volume_index + ((step_index + 1) % n_bins)] += bindist
            else:
                shots[volume_index + ((step_index - 1 + n_bins) % n_bins)] -= bindist

            if dist > radius1_2:
                radius_dist = (dist - radius3_4) / radius1_2
                if dist > radius3_4:
                    init_weight += 1 - radius_dist
                else:
                    init_weight += 1 + radius_dist
                    shots[(desc_index - 2) * (n_bins + 1) + step_index] -= radius_dist
            else:
                radius_dist = (dist - radius1_4) / radius1_2
                if dist < radius1_4:
                    init_weight += 1 + radius_dist
                else:
                    init_weight += 1 - radius_dist
                    shots[(desc_index + 2) * (n_bins + 1) + step_index] += radius_dist
            inclination_cos = max(min(z_lrf / dist, 1.0), -1.0)
            inclination = np.arccos(inclination_cos)
            if inclination > PST_RAD_90 or (abs(inclination - PST_RAD_90) < 1e-30 and z_lrf <= 0):
                inclination_dist = (inclination - PST_RAD_135) / PST_RAD_90
                if inclination > PST_RAD_135:
                    init_weight += 1 - inclination_dist
                else:
                    init_weight += 1 + inclination_dist
                    shots[(desc_index + 1) * (n_bins + 1) + step_index] -= inclination_dist
            else:
                inclination_dist = (inclination - PST_RAD_45) / PST_RAD_90
                if inclination < PST_RAD_45:
                    init_weight += 1 + inclination_dist
                else:
                    init_weight += 1 - inclination_dist
                    shots[(desc_index - 1) * (n_bins + 1) + step_index] += inclination_dist
            if y_lrf != 0.0 or x_lrf != 0.0:
                azimuth = np.arctan2(y_lrf, x_lrf)
                sel = desc_index >> 2
                angular_sector_span = PST_RAD_45
                angular_sector_start = -PST_RAD_PI_7_8
                azimuth_dist = (azimuth - (angular_sector_start + angular_sector_span * sel)) / angular_sector_span
                azimuth_dist = max(-0.5, min(azimuth_dist, 0.5))
                if azimuth_dist > 0:
                    init_weight += 1 - azimuth_dist
                    interp_index = (desc_index + 4) % n_grid_sectors
                    shots[interp_index * (n_bins + 1) + step_index] += azimuth_dist
                else:
                    init_weight += 1 + azimuth_dist
                    interp_index = (desc_index - 4 + n_grid_sectors) % n_grid_sectors
                    shots[interp_index * (n_bins + 1) + step_index] -= azimuth_dist
            shots[volume_index + step_index] += init_weight
            shots /= np.linalg.norm(shots)
            return shots

    shot_fn_v = np.frompyfunc(shot_fn, 2, 1)
    shots = shot_fn_v(np.arange(len(neighbors)), np.array(neighbors, dtype=object))
    return np.array([shot.astype(np.float32) for shot in shots])
