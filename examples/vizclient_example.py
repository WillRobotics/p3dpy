import numpy as np
import p3dpy as pp
from p3dpy import PointCloud, VizClient
import argparse
parser = argparse.ArgumentParser(description='Visualization client example.')
parser.add_argument('--host', type=str, default='localhost', help="Host address.")
args = parser.parse_args()

client = VizClient(host=args.host)
pc = pp.io.load_pcd('data/bunny.pcd')
pc._field = pp.pointcloud.PointXYZRGBField()
colors = np.tile([1.0, 0.0, 0.0], (len(pc), 1))
pc._points = np.c_[pc._points, colors]
res = client.post_pointcloud(pc, 'test')
res_pc = client.get_pointcloud(res["name"])
print(res_pc.points)