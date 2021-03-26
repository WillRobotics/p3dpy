import copy
import numpy as np


class FieldBase(object):
    slices = {}
    def size(self):
        return 0


class PointXYZField(FieldBase):
    X = 0
    Y = 1
    Z = 2
    slices = {"point": slice(3)}
    def size(self):
        return 3


class PointXYZRGBField(PointXYZField):
    R = 3
    G = 4
    B = 5
    slices = {"point": slice(3), "color": slice(3, 6)}
    def size(self):
        return 6


class PointXYZRGBAField(PointXYZField):
    R = 3
    G = 4
    B = 5
    A = 6
    slices = {"point": slice(3), "color": slice(3, 6), "alpha": slice(6, 7)}
    def size(self):
        return 7


class PointXYZNormalField(PointXYZField):
    NX = 3
    NY = 4
    NZ = 5
    slices = {"point": slice(3), "normal": slice(3, 6)}
    def size(self):
        return 6


class PointXYZRGBNormalField(PointXYZRGBField):
    NX = 6
    NY = 7
    NZ = 8
    slices = {"point": slice(3), "color": slice(3, 6), "normal": slice(6, 9)}
    def size(self):
        return 9


class PointCloud(object):
    def __init__(self, points=[], field=PointXYZField()):
        self._field = field
        self._points = points

    def __len__(self):
        return len(self._points)

    def finalize(self):
        self._points = np.array(self._points)

    @property
    def points(self):
        if isinstance(self._points, list):
            self._points = np.array(self._points)
        return self._points[:, self._field.slices['point']]

    @property
    def normals(self):
        if isinstance(self._points, list):
            self._points = np.array(self._points)
        if "normal" in self._field.slices:
            return self._points[:, self._field.slices['normal']]
        else:
            return None

    @property
    def colors(self):
        if isinstance(self._points, list):
            self._points = np.array(self._points)
        if "color" in self._field.slices:
            return self._points[:, self._field.slices['color']]
        else:
            return None

    def append(self, point: np.ndarray):
        if isinstance(self._points, np.ndarray):
            self._points = list(self._points)
        self._points.append(point)

    def extend(self, points: np.ndarray):
        if isinstance(self._points, np.ndarray):
            self._points = list(self._points)
        self._points.extend(points)

    def transform_(self, trans: np.ndarray):
        if isinstance(self._points, list):
            self._points = np.array(self._points)
        self._points[:, self._field.slices['point']] = np.dot(self.points, trans[:3, :3].T) + trans[:3, 3]
        if "normal" in self._field.slices:
            self._points[:, self._field.slices['normal']] = np.dot(self.normals, trans[:3, :3].T)

    def transform(self, trans: np.ndarray):
        pc = PointCloud(copy.deepcopy(self._points), self._field)
        pc.transform_(trans)
        return pc