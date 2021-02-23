from typing import Optional
import numpy as np
from . import pointcloud


class RGBDImage(object):
    def __init__(self, depth_img: np.ndarray, color_img: Optional[np.ndarray] = None):
        if color_img is not None and depth_img.shape != color_img.shape[:3]:
            raise ValueError("The image sizes of the depth image and the color image do not match.")
        self._depth_img = depth_img
        self._color_img = color_img

    def pointcloud(self, intrinsic: np.ndarray, extrinsic: np.ndarray = np.identity(4)) -> pointcloud.PointCloud:
        field = pointcloud.PointXYZRGBField()
        pc = pointcloud.PointCloud()
        pc._points = np.zeros(self._depth_img.shape + [field.size()])
        pc._points[:, :, field.Z] = self._depth_img
        row, col = np.indices(self._depth_img.shape)
        pc._points[:, :, field.X] = (col - intrinsic[0, 3]) * self._depth_img / intrinsic[0, 0]
        pc._points[:, :, field.Y] = (row - intrinsic[1, 3]) * self._depth_img / intrinsic[1, 1]
        pc._points = pc._points.reshape((-1, 6))
        pc._points = pc._points[np.isfinite(pc._points).all(axis=1), :]
        pc.transform_(extrinsic)
        return pc
