import numpy as np
import p3dpy as pp

pc = pp.io.load_pcd('data/bunny.pcd')
pc.compute_normals(0.05)
shots1 = pp.feature.compute_shot_descriptors(pc, 0.05)
print(shots1)

pc = pp.filter.voxel_grid_filter(pc, 0.01)
trans = np.identity(4)
trans[0, 3] += 1.0
pc.transform_(trans)
shots2 = pp.feature.compute_shot_descriptors(pc, 0.05)

pc = pp.io.load_ply('data/sphere.ply')
pc.compute_normals(0.3)
shots3 = pp.feature.compute_shot_descriptors(pc, 0.3)

rmse = pp.registration.compute_rmse(shots1, shots2)
print(rmse)
rmse = pp.registration.compute_rmse(shots1, shots3)
print(rmse)
