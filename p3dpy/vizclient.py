import base64
import urllib.parse

import numpy as np
import requests

from .pointcloud import PointCloud


class VizClient(object):
    def __init__(self, host: str = "localhost", port: int = 8000):
        self._url = "http://%s:%d" % (host, port)

    def _encode(self, s: str):
        return base64.b64encode(s).decode("utf-8")

    def post_pointcloud(self, pointcloud: PointCloud, name: str = ""):
        points = pointcloud.points.astype(np.float32).tobytes("C")
        colors = pointcloud.colors
        colors = (colors * 255).astype(np.uint8).tobytes("C") if colors is not None else (np.ones((len(pointcloud), 3), dtype=np.uint8) * 255).tobytes("C")
        name = id(pointcloud) if name == "" else name
        response = requests.post(
            urllib.parse.urljoin(self._url, "pointcloud/store"),
            json={"name": name, "points": self._encode(points), "colors": self._encode(colors)},
        )
        return response.json()

    def update_pointcloud(self, name: str, pointcloud: PointCloud):
        points = pointcloud.points.astype(np.float32).tobytes("C")
        colors = pointcloud.colors
        colors = (colors * 255).astype(np.uint8).tobytes("C") if colors is not None else (np.ones((len(pointcloud), 3), dtype=np.uint8) * 255).tobytes("C")
        response = requests.put(
            urllib.parse.urljoin(self._url, f"pointcloud/update/{name}"),
            json={"name": "", "points": self._encode(points), "colors": self._encode(colors)},
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
            json={"log": "<p>" + message + "</p>"},
        )
        return response.json()

    def clear_log(self):
        response = requests.get(
            urllib.parse.urljoin(self._url, "log/clear"),
        )
        return response.json()

    def get_parameters(self):
        response = requests.get(
            urllib.parse.urljoin(self._url, "parameters"),
        )
        return response.json()
