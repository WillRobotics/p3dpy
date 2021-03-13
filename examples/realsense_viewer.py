import sys
import pyrealsense2 as rs
import numpy as np
from enum import IntEnum

import p3dpy as pp

import matplotlib.pylab as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D


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

    fig = plt.figure()
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    ax2 = fig.add_subplot(2, 2, 2)
    ax3 = fig.add_subplot(2, 2, 4)
    dummy_points = np.zeros((1, 3))
    pos = ax1.plot(*list(zip(*dummy_points)), 'o', markersize=0.5)[0]
    ax1.invert_yaxis()
    ax1.set_xlim(-3, 3)
    ax1.set_ylim(-3, 3)
    ax1.set_zlim(-1, 6)
    depth_im = ax2.imshow(np.zeros((480, 640), dtype=np.uint8), cmap="gray", vmin=0, vmax=255)
    color_im = ax3.imshow(np.zeros((480, 640, 3), dtype=np.uint8))

    try:
        def update(frame):
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
                return

            depth_image = np.array(aligned_depth_frame.get_data())
            color_image = np.asarray(color_frame.get_data())
            rgbd_image = pp.RGBDImage(depth_image, color_image)
            pcd = rgbd_image.pointcloud(intrinsic)
            pcd.transform_(flip_transform)
            pos.set_data(pcd.points[:, 0], pcd.points[:, 1])
            pos.set_3d_properties(pcd.points[:, 2])
            depth_offset = rgbd_image.depth.min()
            depth_scale = rgbd_image.depth.max() - depth_offset
            depth_temp = np.clip((rgbd_image.depth - depth_offset) / depth_scale, 0.0, 1.0)
            depth_im.set_array((255.0 * depth_temp).astype(np.uint8))
            color_im.set_array((255.0 * rgbd_image.color).astype(np.uint8))

        anim = animation.FuncAnimation(fig, update, interval=10)
        plt.show()

    finally:
        pipeline.stop()
