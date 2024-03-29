import sys
import pyrealsense2 as rs
import numpy as np
from enum import IntEnum
import datetime
import h5py

import p3dpy as pp

import argparse
parser = argparse.ArgumentParser(description='Visualization client example.')
parser.add_argument('--host', type=str, default='localhost', help="Host address.")
parser.add_argument('--scale', type=float, default=1000, help="Depth scale.")
args = parser.parse_args()

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
    outfh = h5py.File("sensor_data.h5", "a")
    if "pointclouds" in outfh:
        grp = outfh["pointclouds"]
    else:
        grp = outfh.create_group("pointclouds")

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

            depth_image = np.array(aligned_depth_frame.get_data()).astype(np.float32)
            color_image = np.asarray(color_frame.get_data())
            rgbd_image = pp.RGBDImage(depth_image, color_image, args.scale)
            pcd = rgbd_image.pointcloud(intrinsic)
            pcd.transform_(flip_transform)
            grp.create_dataset(datetime.datetime.now().strftime("%Y%m%d%H%M%S"), data=pcd._points)
            pcd = pp.filter.voxel_grid_filter(pcd, 0.01)
            res = client.post_pointcloud(pcd, 'test')
            print(res)

    finally:
        pipeline.stop()
