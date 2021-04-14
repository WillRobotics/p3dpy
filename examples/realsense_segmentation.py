import sys
import pyrealsense2 as rs
import numpy as np
from enum import IntEnum
from scipy.spatial import ConvexHull
from sklearn.cluster import DBSCAN
import transforms3d as t3d

import p3dpy as pp

import argparse
parser = argparse.ArgumentParser(description='Visualization client example.')
parser.add_argument('--host', type=str, default='localhost', help="Host address.")
args = parser.parse_args()

colors = [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.5, 0.5, 1.0]]

def calc_triangle_area(p0, p1, p2):
    return 0.5 * (p0[0] * (p1[1] - p2[1]) + p1[0] * (p2[1] - p0[1]) + p2[0] * (p0[1] - p1[1]))


def calc_convexhull_area(points):
    hull = ConvexHull(points[:, :2])
    hull_points = points[hull.vertices, :2]
    hull_points = np.r_[hull_points, [hull_points[0]]]
    hull_mean = hull_points.mean(axis=0)
    area = 0
    for i in range(len(hull_points) - 1):
        area += calc_triangle_area(hull_mean, hull_points[i], hull_points[i + 1])
    return area


class Preset(IntEnum):
    Custom = 0
    Default = 1
    Hand = 2
    HighAccuracy = 3
    HighDensity = 4
    MediumDensity = 5


def get_intrinsic_matrix(frame):
    intrinsics = frame.profile.as_video_stream_profile().intrinsics
    out = np.identity(3)
    out[0, 0] = intrinsics.fx
    out[1, 1] = intrinsics.fy
    out[0, 2] = intrinsics.ppx
    out[1, 2] = intrinsics.ppy
    return out


if __name__ == "__main__":

    pp.vizspawn(host=args.host)

    # Create a pipeline
    pipeline = rs.pipeline()

    #Create a config and configure the pipeline to stream
    #  different resolutions of color and depth streams
    config = rs.config()

    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)

    # Start streaming
    profile = pipeline.start(config)
    depth_sensor = profile.get_device().first_depth_sensor()

    # Using preset HighAccuracy for recording
    depth_sensor.set_option(rs.option.visual_preset, Preset.HighAccuracy)

    # Getting the depth sensor's depth scale (see rs-align example for explanation)
    depth_scale = depth_sensor.get_depth_scale()

    # We will not display the background of objects more than
    #  clipping_distance_in_meters meters away
    clipping_distance_in_meters = 3  # 3 meter
    clipping_distance = clipping_distance_in_meters / depth_scale
    # print(depth_scale)

    # Create an align object
    # rs.align allows us to perform alignment of depth frames to others frames
    # The "align_to" is the stream type to which we plan to align depth frames.
    align_to = rs.stream.color
    align = rs.align(align_to)

    pcd = pp.PointCloud()
    flip_transform = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    client = pp.VizClient(host=args.host)

    try:
        while True:
            # Get frameset of color and depth
            frames = pipeline.wait_for_frames()

            # Align the depth frame to color frame
            aligned_frames = align.process(frames)

            # Get aligned frames
            aligned_depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()
            intrinsic = get_intrinsic_matrix(color_frame)

            # Validate that both frames are valid
            if not aligned_depth_frame or not color_frame:
                continue

            depth_image = np.array(aligned_depth_frame.get_data())
            color_image = np.asarray(color_frame.get_data())
            rgbd_image = pp.RGBDImage(depth_image, color_image)
            pcd = rgbd_image.pointcloud(intrinsic)
            pcd.transform_(flip_transform)
            pcd = pp.filter.voxel_grid_filter(pcd, 0.01)
            plane, mask = pp.segmentation.segmentation_plane(pcd, dist_thresh=0.03)
            axis = np.array([-plane[1], plane[0], 0.0])
            axis /= np.linalg.norm(axis)
            angle = np.arccos(plane[2] / np.linalg.norm(plane[:3]))
            trans = np.identity(4)
            trans[:3, :3] = t3d.axangles.axangle2mat(axis, angle)
            pcd.transform_(trans.T)

            # Divide a plane and objects
            plane_pts = pcd._points[mask, :]
            not_plane_pts = pcd._points[~mask, :]
            # Cluster objects
            db = DBSCAN(eps=0.1).fit(not_plane_pts[:, :3])
            labels = db.labels_
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            print("Number of class:", n_clusters)
            masks = [labels == i for i in range(n_clusters)]

            plane_pts[:, 3:] = [0.0, 0.0, 1.0]
            for i in range(n_clusters):
                not_plane_pts[masks[i], 3:] = colors[i % len(colors)]
            result_pts = np.vstack([plane_pts, not_plane_pts])

            # Draw results
            result_pc = pp.PointCloud(result_pts, pp.pointcloud.PointXYZRGBField())
            res = client.post_pointcloud(result_pc, 'test')
            plane_area = calc_convexhull_area(plane_pts[:, :2])
            client.clear_log()
            client.add_log(f"Plane Area: {plane_area}")
            for i in range(n_clusters):
                obj_area = calc_convexhull_area(not_plane_pts[masks[i], :2])
                client.add_log(f"Obj{i} Area: {obj_area}")
            print(res)

    finally:
        pipeline.stop()
