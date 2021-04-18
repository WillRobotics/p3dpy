#!/usr/bin/env python3
import os
from typing import List
import asyncio
from fastapi import Body, FastAPI, Request, WebSocket
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.encoders import jsonable_encoder
from fastapi.logger import logger
from fastapi.responses import JSONResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

import p3dpy


class PointCloudData(BaseModel):
    name: str
    points: List[List[float]]
    colors: List[List[int]]


app = FastAPI()
stored_data = {"pointcloud": {}, "log": "", "clearLog": False}
parameters = {"max_points": 500000}

app.mount(
    "/static",
    StaticFiles(directory=os.path.join(os.path.dirname(p3dpy.__file__), "app/static"), html=True),
    name="static",
)
templates = Jinja2Templates(directory=os.path.join(os.path.dirname(p3dpy.__file__), "app/templates"))


@app.get("/", response_class=HTMLResponse)
async def get_webpage(request: Request):
    return templates.TemplateResponse("index.html", context={"request": request, "parameters": parameters})


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            data = await ws.receive_text()
            if data in stored_data["pointcloud"]:
                await ws.send_json({data: stored_data["pointcloud"][data]})
    except:
        await ws.close()


@app.websocket("/info")
async def info_endpoint(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            await ws.send_json(
                {"keys": list(stored_data["pointcloud"].keys()),
                 "log": stored_data["log"],
                 "clearLog": stored_data["clearLog"]})
            stored_data["log"] = ""
            stored_data["clearLog"] = False
            await asyncio.sleep(1)
    except:
        await ws.close()


@app.get("/pointcloud/{name}")
async def get_data(name: str):
    json_data = jsonable_encoder(stored_data["pointcloud"][name])
    return JSONResponse(content=json_data)


@app.post("/pointcloud/store")
async def store_data(data: PointCloudData):
    stored_data["pointcloud"][data.name] = [data.points, data.colors]
    return {"res": "ok", "name": data.name}


@app.put("/pointcloud/update/{name}")
async def update_data(name: str, data: PointCloudData):
    stored_data["pointcloud"][data.name] = [data.points, data.colors]
    return {"res": "ok", "name": data.name}


@app.post("/log/store")
async def store_log(body: dict = Body(...)):
    if isinstance(body["log"], str):
        stored_data["log"] += body["log"]
        return {"res": "ok"}
    return {"res": "error"}


@app.get("/log/clear")
async def clear_log():
    stored_data["clearLog"] = True
    return {"res": "ok"}


def main():
    import uvicorn
    import argparse

    parser = argparse.ArgumentParser(description="Visualization server for p3dpy.")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host address.")
    parser.add_argument("--port", type=int, default=8000, help="Port number.")
    args = parser.parse_args()
    uvicorn.run(app=app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
