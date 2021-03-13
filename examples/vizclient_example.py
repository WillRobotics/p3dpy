import p3dpy as pp
from p3dpy import PointCloud, VizClient

client = VizClient()
pc = pp.io.load_pcd('data/bunny.pcd')
res = client.post_pointcloud(pc)
res_pc = client.get_pointcloud(res["name"])
print(res_pc.points)