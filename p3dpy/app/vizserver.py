#!/usr/bin/env python3
import asyncio
import base64
import json
import os
from typing import Any, Dict, List, Union

import numpy as np
from fastapi import APIRouter, Body, FastAPI, Request, WebSocket
from fastapi.encoders import jsonable_encoder
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

import p3dpy


class PointCloudData(BaseModel):
    name: str
    points: str
    colors: str


class PointCloudDataArray(BaseModel):
    array: List[PointCloudData]
    clear: bool


def _encode(s: bytes) -> str:
    return base64.b64encode(s).decode("utf-8")


def _decode(s: str) -> bytes:
    return base64.b64decode(s.encode())


router = APIRouter()
stored_data: Dict[str, Any] = {"pointcloud": {}, "log": "", "clearLog": False}
parameters: Dict[str, Any] = {"max_points": 500000, "gui_params": {}}
templates = Jinja2Templates(directory=os.path.join(os.path.dirname(p3dpy.__file__), "app/templates"))


@router.get("/", response_class=HTMLResponse)
async def get_webpage(request: Request):
    return templates.TemplateResponse("index.html", context={"request": request, "parameters": parameters})


@router.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            data = await ws.receive_json()
            data_pack: List[List[np.ndarray]] = [[], []]
            for d in data:
                if d in stored_data["pointcloud"]:
                    data_pack[0].append(stored_data["pointcloud"][d][0])
                    data_pack[1].append(stored_data["pointcloud"][d][1])
            send_data: List[Union[List, str]] = [[], []]
            if len(data_pack[0]) > 0:
                send_data[0] = _encode(np.concatenate(data_pack[0], axis=0).tobytes("C"))
            if len(data_pack[1]) > 0:
                send_data[1] = _encode(np.concatenate(data_pack[1], axis=0).tobytes("C"))
            await ws.send_json(send_data)
    except:
        await ws.close()


@router.websocket("/info")
async def info_endpoint(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            await ws.send_json(
                {
                    "keys": list(stored_data["pointcloud"].keys()),
                    "log": stored_data["log"],
                    "clearLog": stored_data["clearLog"],
                }
            )
            stored_data["log"] = ""
            stored_data["clearLog"] = False
            await asyncio.sleep(1)
    except:
        await ws.close()


@router.get("/pointcloud/{name}")
async def get_data(name: str):
    points, colors = stored_data["pointcloud"][name]
    json_data = jsonable_encoder([points.tolist(), colors.tolist()])
    return JSONResponse(content=json_data)


@router.post("/pointcloud/store")
async def store_data(data: PointCloudData):
    points = np.frombuffer(_decode(data.points), dtype=np.float32)
    points = points.reshape((-1, 3))
    colors = np.frombuffer(_decode(data.colors), dtype=np.uint8)
    colors = colors.reshape((-1, 3))
    stored_data["pointcloud"][data.name] = [points, colors]
    return {"res": "ok", "name": data.name}


@router.post("/pointcloud/store_array")
async def store_data_array(data: PointCloudDataArray):
    if data.clear:
        stored_data["pointcloud"] = {}
    names = []
    for d in data.array:
        points = np.frombuffer(_decode(d.points), dtype=np.float32)
        points = points.reshape((-1, 3))
        colors = np.frombuffer(_decode(d.colors), dtype=np.uint8)
        colors = colors.reshape((-1, 3))
        stored_data["pointcloud"][d.name] = [points, colors]
        names.append(d.name)
    return {"res": "ok", "names": names}


@router.put("/pointcloud/update/{name}")
async def update_data(name: str, data: PointCloudData):
    points = np.frombuffer(_decode(data.points), dtype=np.float32)
    points = points.reshape((-1, 3))
    colors = np.frombuffer(_decode(data.colors), dtype=np.uint8)
    colors = colors.reshape((-1, 3))
    stored_data["pointcloud"][data.name] = [points, colors]
    return {"res": "ok", "name": data.name}


@router.post("/log")
async def store_log(body: dict = Body(...)):
    if isinstance(body["log"], str):
        stored_data["clearLog"] = body["clear"]
        if body["clear"]:
            stored_data["log"] = body["log"]
        else:
            stored_data["log"] += body["log"]
        return {"res": "ok"}
    return {"res": "error"}


@router.post("/parameters/store")
async def post_parameters(body: dict = Body(...)):
    for k, v in body.items():
        if k in parameters["gui_params"]:
            parameters["gui_params"][k][0] = v
    return {"res": "ok"}


@router.get("/parameters")
async def get_parameters():
    data = dict([(k, v[0]) for k, v in parameters["gui_params"].items()])
    json_data = jsonable_encoder(data)
    return JSONResponse(content=json_data)


def create_app():
    app = FastAPI()
    app.mount(
        "/static",
        StaticFiles(directory=os.path.join(os.path.dirname(p3dpy.__file__), "app/static"), html=True),
        name="static",
    )
    app.include_router(router)
    return app


def set_parameters(params: Dict):
    global parameters
    parameters["gui_params"] = json.loads(params)
