import numpy as np
import p3dpy as pp
from p3dpy import PointCloud, VizClient
import argparse
parser = argparse.ArgumentParser(description='Visualization client example.')
parser.add_argument('--host', type=str, default='localhost', help="Host address.")
args = parser.parse_args()


pp.vizspawn(host=args.host)

client = VizClient(host=args.host)
pc1 = pp.io.load_pcd('data/bunny.pcd')
pc2 = pp.io.load_pcd('data/bunny.pcd')
pc1.set_uniform_color([1.0, 0.0, 0.0])
pc2.set_uniform_color([0.0, 1.0, 0.0])
res1 = client.post_pointcloud(pc1, 'test1')
res2 = client.post_pointcloud(pc2, 'test2')

pp.vizloop()