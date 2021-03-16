import p3dpy as pp

pc = pp.io.load_pcd('data/bunny.pcd')
pc1 = pp.filter.radius_outlier_removal(pc, 0.05, 30)
print(len(pc), len(pc1))

pc2 = pp.filter.statistical_outlier_removal(pc, 10, 0.6)
print(len(pc), len(pc2))