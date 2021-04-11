from typing import Optional
import copy
import numpy as np


class FieldBase(object):
    def __init__(self):
        self.slices = {}

    def size(self) -> int:
        return 0

    def has_field(self, name: str) -> bool:
        return name in self.slices


class PointXYZField(FieldBase):
    X = 0
    Y = 1
    Z = 2
    def __init__(self):
        self.slices = {"point": slice(3)}

    def size(self) -> int:
        return 3


class PointXYZRGBField(PointXYZField):
    R = 3
    G = 4
    B = 5
    def __init__(self):
        self.slices = {"point": slice(3), "color": slice(3, 6)}

    def size(self) -> int:
        return 6


class PointXYZRGBAField(PointXYZField):
    R = 3
    G = 4
    B = 5
    A = 6
    def __init__(self):
        self.slices = {"point": slice(3), "color": slice(3, 6), "alpha": slice(6, 7)}

    def size(self) -> int:
        return 7


class PointXYZNormalField(PointXYZField):
    NX = 3
    NY = 4
    NZ = 5
    def __init__(self):
        self.slices = {"point": slice(3), "normal": slice(3, 6)}

    def size(self) -> int:
        return 6


class PointXYZRGBNormalField(PointXYZRGBField):
    NX = 6
    NY = 7
    NZ = 8
    def __init__(self):
        self.slices = {"point": slice(3), "color": slice(3, 6), "normal": slice(6, 9)}

    def size(self) -> int:
        return 9


class DynamicField(FieldBase):
    def __init__(self, init_field: Optional[FieldBase] = None):
        if init_field is None:
            self.slices = {}
        else:
            self.slices = init_field.slices

    def add_field(self, name: str, n_elem: slice):
        size = self.size()
        self.slices.update({name: slice(size, size + n_elem)})

    def size(self) -> int:
        return max([s.stop for s in self.slices.values()])


class PointCloud(object):
    """Point cloud class."""

    def __init__(self, points=[], field=PointXYZField()):
        """Constructor

        Parameters
        ----------
        points: list or np.ndarray
            2D ndarray or list of 1D ndarray.
            Each row represents one point of the point cloud.
            Each column represents one scalar field associated to its corresponding point.

        field: FieldBase
            The field of data contained in each point.
        """
        self._field = field
        self._points = points

    def __len__(self):
        return len(self._points)

    def has_field(self, name: str) -> bool:
        return self._field.has_field(name)

    def finalize(self):
        self._points = np.array(self._points)

    def mean(self) -> np.ndarray:
        if isinstance(self._points, list):
            self._points = np.array(self._points)
        return self._points.mean(axis=0)

    def min_point(self) -> np.ndarray:
        return self.points.min(axis=0)

    def max_point(self) -> np.ndarray:
        return self.points.max(axis=0)

    def bounding_box(self) -> (np.ndarray, np.ndarray):
        return self.min_point(), self.max_point()

    @property
    def points(self):
        if isinstance(self._points, list):
            self._points = np.array(self._points)
        return self._points[:, self._field.slices["point"]]

    @property
    def normals(self):
        if isinstance(self._points, list):
            self._points = np.array(self._points)
        if self.has_field("normal"):
            return self._points[:, self._field.slices["normal"]]
        else:
            return None

    @property
    def colors(self):
        if isinstance(self._points, list):
            self._points = np.array(self._points)
        if self.has_field("color"):
            return self._points[:, self._field.slices["color"]]
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
        self._points[:, self._field.slices["point"]] = np.dot(self.points, trans[:3, :3].T) + trans[:3, 3]
        if self.has_field("normal"):
            self._points[:, self._field.slices["normal"]] = np.dot(self.normals, trans[:3, :3].T)

    def transform(self, trans: np.ndarray):
        pc = PointCloud(copy.deepcopy(self._points), self._field)
        pc.transform_(trans)
        return pc

    def set_uniform_color(self, color: np.ndarray):
        if self.has_field("colors"):
            self._points[:, self._field.slices["color"]] = color
        else:
            self._field = DynamicField(self._field)
            self._field.add_field("color", 3)
            self._points = np.c_[self._points, np.tile(color, (len(self), 1))]
