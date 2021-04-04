import numpy as np
import p3dpy as pp
from p3dpy import PointCloud, VizClient
import argparse
parser = argparse.ArgumentParser(description='Visualization client example.')
parser.add_argument('--host', type=str, default='localhost', help="Host address.")
args = parser.parse_args()


pp.vizspawn(host=args.host)

client = VizClient(host=args.host)
pc = pp.io.load_pcd('data/bunny.pcd')
pc.set_uniform_color([1.0, 0.0, 0.0])
res = client.post_pointcloud(pc, 'test')
res_pc = client.get_pointcloud(res["name"])
print(res_pc.points)

pp.vizloop()