import urllib.parse

import numpy as np
import requests

from .pointcloud import PointCloud


class VizClient(object):
    def __init__(self, host: str = "localhost", port: int = 8000):
        self._url = "http://%s:%d" % (host, port)

    def post_pointcloud(self, pointcloud: PointCloud, name: str = ""):
        points = pointcloud.points.tolist()
        name = id(pointcloud) if name == "" else name
        response = requests.post(urllib.parse.urljoin(self._url, "pointcloud/store"), json={"name": name, "points": points})
        return response.json()

    def update_pointcloud(self, name: str, pointcloud: PointCloud):
        points = pointcloud.points.tolist()
        response = requests.put(urllib.parse.urljoin(self._url, f"pointcloud/update/{name}"), json={"name": "", "points": points})
        return response.json()

    def get_pointcloud(self, name: str):
        response = requests.get(urllib.parse.urljoin(self._url, f"pointcloud/{name}"))
        points = response.json()
        pointcloud = PointCloud()
        pointcloud.points_ = np.array(points)
        return pointcloud