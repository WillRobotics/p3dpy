# p3dpy

[![PyPI version](https://badge.fury.io/py/p3dpy.svg)](https://badge.fury.io/py/p3dpy)

Numpy based simple pointcloud tools.

## Core features

* Basic pointcloud operations (Transformation, Registration, Filtering, Feature,...)
* Simple dependencies (numpy, scipy,... other basic python packages)
* Browser based viewer
* Easy use for single board computers (Raspberry Pi, Jetson,...)

## Installation

```
pip install p3dpy
```

## Getting Started

This is a simple example to vizualize a pcd file.

```py
import numpy as np
import p3dpy as pp
from p3dpy import VizClient
import argparse
parser = argparse.ArgumentParser(description='Simple example.')
parser.add_argument('--host', type=str, default='localhost', help="Host address.")
args = parser.parse_args()


pp.vizspawn(host=args.host)

client = VizClient(host=args.host)
pc = pp.io.load_pcd('data/bunny.pcd')
pc.set_uniform_color([1.0, 0.0, 0.0])
res = client.post_pointcloud(pc, 'test')

pp.vizloop()
```

## Visualization
![demo](https://raw.githubusercontent.com/WillRobotics/p3dpy/master/assets/p3dpy_demo.gif)
