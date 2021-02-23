import p3dpy as pp

pc = pp.io.load_pcd('data/bunny.pcd')
print(pc._points)