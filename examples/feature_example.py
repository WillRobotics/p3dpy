import p3dpy as pp

pc = pp.io.load_pcd('data/bunny.pcd')
pc.compute_normals(0.05)
shots = pp.feature.compute_shot_descriptors(pc, 0.05)
print(shots)