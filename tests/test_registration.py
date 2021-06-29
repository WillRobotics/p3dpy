import numpy as np

import p3dpy as pp
import unittest


class TestRegistration(unittest.TestCase):
    def test_icp_registration(self):
        source = pp.PointCloud(np.random.rand(10, 3), pp.pointcloud.PointXYZField())
        angle = np.deg2rad(10.0)
        trans = np.identity(4)
        trans[0, 0] = np.cos(angle)
        trans[0, 1] = -np.sin(angle)
        trans[1, 0] = np.sin(angle)
        trans[1, 1] = np.cos(angle)
        target = source.transform(trans)
        res = pp.registration.icp_registration(source, target, 0.3)
        np.testing.assert_almost_equal(res, trans)


if __name__ == '__main__':
    unittest.main()
