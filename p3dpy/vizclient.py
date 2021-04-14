import urllib.parse

import numpy as np
import requests

from .pointcloud import PointCloud


class VizClient(object):
    def __init__(self, host: str = "localhost", port: int = 8000):
        self._url = "http://%s:%d" % (host, port)

    def post_pointcloud(self, pointcloud: PointCloud, name: str = ""):
        points = pointcloud.points.tolist()
        colors = pointcloud.colors
        colors = (colors * 255).astype(np.uint8).tolist() if colors is not None else []
        name = id(pointcloud) if name == "" else name
        response = requests.post(
            urllib.parse.urljoin(self._url, "pointcloud/store"),
            json={"name": name, "points": points, "colors": colors},
        )
        return response.json()

    def update_pointcloud(self, name: str, pointcloud: PointCloud):
        points = pointcloud.points.tolist()
        colors = pointcloud.colors
        colors = (colors * 255).astype(np.uint8).tolist() if colors is not None else []
        response = requests.put(
            urllib.parse.urljoin(self._url, f"pointcloud/update/{name}"),
            json={"name": "", "points": points, "colors": colors},
        )
        return response.json()

    def get_pointcloud(self, name: str):
        response = requests.get(urllib.parse.urljoin(self._url, f"pointcloud/{name}"))
        points = response.json()
        pointcloud = PointCloud()
        pointcloud.points_ = np.array(points[0])
        return pointcloud

    def add_log(self, message: str):
        response = requests.post(
            urllib.parse.urljoin(self._url, "log/store"),
            json={"log": message},
        )
        return response.json()

    def clear_log(self):
        response = requests.get(
            urllib.parse.urljoin(self._url, "log/clear"),
        )
        return response.json()
