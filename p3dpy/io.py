from typing import BinaryIO, Optional, TextIO, Tuple, Union
import struct
import numpy as np
import stl
from plyfile import PlyData
import lzf
from . import pointcloud


_field_dict = {
    "I1": "b",
    "I2": "h",
    "I4": "i",
    "U1": "B",
    "U2": "H",
    "U4": "I",
    "F4": "f",
    "F8": "d",
}
_type_dict = {
    "I1": int,
    "I2": int,
    "I4": int,
    "U1": int,
    "U2": int,
    "U4": int,
    "F4": float,
    "F8": float,
}


def _parse_pcd_header(lines: list) -> Tuple[dict, str]:
    config = {}
    data_type = "ascii"
    for c in lines:
        c = c.split()
        if len(c) == 0:
            continue
        if c[0] == "FIELDS" or c[0] == "SIZE" or c[0] == "TYPE" or c[0] == "COUNT":
            config[c[0]] = c[1:]
        elif c[0] == "WIDTH" or c[0] == "POINTS":
            config[c[0]] = int(c[1])
        elif c[0] == "DATA":
            data_type = c[1]
            break
        else:
            continue
    return config, data_type


def load_pcd(fd: Union[BinaryIO, TextIO, str]) -> pointcloud.PointCloud:
    """Load PCD file format

    Parameters
    ----------
    fd: BinaryIO, TextIO or str
        Input file name or StringIO data type.
    """
    if isinstance(fd, str):
        fd = open(fd, "rb")
    lines = []
    while True:
        ln = fd.readline().strip().decode()
        lines.append(ln)
        if ln.startswith("DATA"):
            break
    config, data_type = _parse_pcd_header(lines)

    has_point = False
    has_color = False
    has_normal = False
    if "x" in config["FIELDS"] and "y" in config["FIELDS"] and "z" in config["FIELDS"]:
        has_point = True
    if "rgb" in config["FIELDS"]:
        has_color = True
    if "normal_x" in config["FIELDS"] and "normal_y" in config["FIELDS"] and "normal_z" in config["FIELDS"]:
        has_normal = True

    field: Optional[pointcloud.FieldBase] = None
    if has_point and has_color and has_normal:
        field = pointcloud.PointXYZRGBNormalField()
    elif has_point and has_color:
        field = pointcloud.PointXYZRGBField()
    elif has_point and has_normal:
        field = pointcloud.PointXYZNormalField()
    elif has_point:
        field = pointcloud.PointXYZField()
    else:
        raise ValueError("Unsupport field type.")

    pc = pointcloud.PointCloud(points=[], field=field)
    fmt = ""
    for i in range(len(config["FIELDS"])):
        fmt += config["COUNT"][i] if int(config["COUNT"][i]) > 1 else ""
        fmt += _field_dict[config["TYPE"][i] + config["SIZE"][i]]

    loaddata = []
    if data_type == "ascii":
        lines = fd.read().splitlines()
        for d in lines:
            d = d.split()
            cnt = 0
            data = []
            for i in range(len(config["FIELDS"])):
                fcnt = int(config["COUNT"][i])
                tp_s = config["TYPE"][i] + config["SIZE"][i]
                if fcnt == 1:
                    data.append(_type_dict[tp_s](d[cnt]))
                else:
                    data.append([_type_dict[tp_s](d[cnt + j]) for j in range(fcnt)])
                cnt += fcnt
            loaddata.append(data)
    elif data_type == "binary":
        bytedata = fd.read()
        size = struct.calcsize(fmt)
        for i in range(len(bytedata) // size):
            loaddata.append(list(struct.unpack(fmt, bytedata[(i * size) : ((i + 1) * size)])))
    elif data_type == "binary_compressed":
        compressed_size, uncompressed_size = struct.unpack("II", fd.read(8))
        compressed_data = fd.read(compressed_size)
        buf = lzf.decompress(compressed_data, uncompressed_size)
        size = struct.calcsize(fmt)
        for i in range(len(buf) // size):
            loaddata.append(list(struct.unpack(fmt, buf[(i * size) : ((i + 1) * size)])))
    else:
        raise ValueError(f"Unsupported data type {data_type}.")

    for data in loaddata:
        pc._points.append(np.zeros(pc._field.size()))
        for f, d in zip(config["FIELDS"], data):
            if f == "x":
                pc._points[-1][pc._field.X] = d
            elif f == "y":
                pc._points[-1][pc._field.Y] = d
            elif f == "z":
                pc._points[-1][pc._field.Z] = d
            elif f == "rgb":
                d = int(d)
                pc._points[-1][pc._field.R] = float((d >> 16) & 0x000FF) / 255.0
                pc._points[-1][pc._field.G] = float((d >> 8) & 0x000FF) / 255.0
                pc._points[-1][pc._field.B] = float((d) & 0x000FF) / 255.0
            elif f == "normal_x":
                pc._points[-1][pc._field.NX] = d
            elif f == "normal_y":
                pc._points[-1][pc._field.NY] = d
            elif f == "normal_z":
                pc._points[-1][pc._field.NZ] = d

    pc.finalize()
    return pc


def load_stl(fd: Union[BinaryIO, TextIO, str], scale: float = 1.0) -> pointcloud.PointCloud:
    if isinstance(fd, str):
        fd = open(fd, "rb")
    mesh = stl.mesh.Mesh.from_file("", fh=fd)
    return pointcloud.PointCloud(points=mesh.points.reshape((-1, 3)) * scale, field=pointcloud.PointXYZField())


def load_ply(fd: Union[BinaryIO, TextIO, str]) -> pointcloud.PointCloud:
    if isinstance(fd, str):
        fd = open(fd, "rb")
    plydata = PlyData.read(fd)
    points = plydata["vertex"][["x", "y", "z"]]
    return pointcloud.PointCloud(
        points=points.view("<f4").reshape(points.shape + (-1,)), field=pointcloud.PointXYZField()
    )
