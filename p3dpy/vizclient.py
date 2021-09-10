from typing import List
import base64
import urllib.parse

import numpy as np
import requests

from .pointcloud import PointCloud


class VizClient(object):
    def __init__(self, host: str = "localhost", port: int = 8000) -> None:
        self._url = "http://%s:%d" % (host, port)

    def _encode(self, s: bytes) -> str:
        return base64.b64encode(s).decode("utf-8")

    def post_pointcloud(self, pointcloud: PointCloud, name: str = "") -> requests.models.Response:
        points = pointcloud.points.astype(np.float32).tobytes("C")
        colors = (
            (pointcloud.colors * 255).astype(np.uint8).tobytes("C")
            if pointcloud.colors is not None
            else (np.ones((len(pointcloud), 3), dtype=np.uint8) * 255).tobytes("C")
        )
        name = str(id(pointcloud)) if name == "" else name
        response = requests.post(
            urllib.parse.urljoin(self._url, "pointcloud/store"),
            json={"name": name, "points": self._encode(points), "colors": self._encode(colors)},
        )
        return response

    def post_pointcloud_array(self, pointclouds: List[PointCloud], names: List[str] = [], clear: bool = False) -> requests.models.Response:
        pointcloud_arr = []
        for i, pc in enumerate(pointclouds):
            points = pc.points.astype(np.float32).tobytes("C")
            colors = (
                (pc.colors * 255).astype(np.uint8).tobytes("C")
                if pc.colors is not None
                else (np.ones((len(pc), 3), dtype=np.uint8) * 255).tobytes("C")
            )
            name = id(pc) if len(names) == 0 else names[i]
            pointcloud_arr.append({"name": name, "points": self._encode(points), "colors": self._encode(colors)})
        response = requests.post(
            urllib.parse.urljoin(self._url, "pointcloud/store_array"),
            json={"array": pointcloud_arr, "clear": clear},
        )
        return response

    def update_pointcloud(self, name: str, pointcloud: PointCloud) -> requests.models.Response:
        points = pointcloud.points.astype(np.float32).tobytes("C")
        colors = (
            (pointcloud.colors * 255).astype(np.uint8).tobytes("C")
            if pointcloud.colors is not None
            else (np.ones((len(pointcloud), 3), dtype=np.uint8) * 255).tobytes("C")
        )
        response = requests.put(
            urllib.parse.urljoin(self._url, f"pointcloud/update/{name}"),
            json={"name": "", "points": self._encode(points), "colors": self._encode(colors)},
        )
        return response

    def get_pointcloud(self, name: str) -> PointCloud:
        response = requests.get(urllib.parse.urljoin(self._url, f"pointcloud/{name}"))
        points = response.json()
        pointcloud = PointCloud()
        pointcloud._points = np.array(points[0])
        return pointcloud

    def add_log(self, message: str, clear: bool = False) -> requests.models.Response:
        message = message.replace("\n", "<br/>")
        response = requests.post(
            urllib.parse.urljoin(self._url, "log"),
            json={"log": "<p>" + message + "</p>", "clear": clear},
        )
        return response

    def get_parameters(self) -> requests.models.Response:
        response = requests.get(
            urllib.parse.urljoin(self._url, "parameters"),
        )
        return response
