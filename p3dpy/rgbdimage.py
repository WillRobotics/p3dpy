from typing import Optional
import numpy as np
from . import pointcloud


class RGBDImage(object):
    def __init__(
        self,
        depth_img: np.ndarray,
        color_img: Optional[np.ndarray] = None,
        depth_scale: float = 1000.0,
    ) -> None:
        if (
            color_img is not None
            and depth_img.shape[0] != color_img.shape[0]
            and depth_img.shape[1] != color_img.shape[1]
        ):
            raise ValueError("The image sizes of the depth image and the color image do not match.")
        self._depth_img = depth_img.astype(np.float32) / depth_scale
        if color_img is not None:
            self._color_img = color_img.astype(np.float32)
            if color_img.dtype == np.uint8:
                self._color_img /= 255.0

    @property
    def depth(self) -> np.ndarray:
        return self._depth_img

    @property
    def color(self) -> np.ndarray:
        return self._color_img

    def pointcloud(self, intrinsic: np.ndarray, extrinsic: np.ndarray = np.identity(4)) -> pointcloud.PointCloud:
        """Pointcloud from RGBD Image.

        Parameters
        ----------
        intrinsic: np.ndarray
            Camera intrinsic parameters.
        extrinsic: np.ndarray
            Camera extrinsic parameters.

        Returns
        -------
        pointcloud.PointCloud
            Pointcloud.
        """
        field = pointcloud.PointXYZRGBField()
        pc = pointcloud.PointCloud(np.zeros(list(self._depth_img.shape) + [field.size()]), field)
        pc._points[:, :, field.Z] = self._depth_img
        row, col = np.indices(self._depth_img.shape)
        pc._points[:, :, field.X] = (col - intrinsic[0, 2]) * self._depth_img / intrinsic[0, 0]
        pc._points[:, :, field.Y] = (row - intrinsic[1, 2]) * self._depth_img / intrinsic[1, 1]
        pc._points[:, :, field.slices["color"]] = self._color_img
        pc._points = pc._points.reshape((-1, 6))
        pc._points = pc._points[np.isfinite(pc._points).all(axis=1), :]
        pc.transform_(extrinsic)
        return pc
