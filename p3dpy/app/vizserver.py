#!/usr/bin/env python3
import os
from typing import List
from fastapi import FastAPI, Request, WebSocket
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


app = FastAPI()
stored_data = {"test": [[1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [3.0, 3.0, 3.0]]}

app.mount("/static", StaticFiles(directory=os.path.join(os.path.dirname(p3dpy.__file__), "app/static"), html=True), name="static")
templates = Jinja2Templates(directory=os.path.join(os.path.dirname(p3dpy.__file__), "app/templates"))


@app.get('/', response_class=HTMLResponse)
async def get_webpage(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    while True:
        data = await ws.receive_text()
        if data in stored_data:
            await ws.send_json({data: stored_data[data]})


@app.get("/pointcloud/{name}")
async def get_data(name: str):
    json_data = jsonable_encoder(stored_data[name])
    return JSONResponse(content=json_data)


@app.post("/pointcloud/store")
async def store_data(data: PointCloudData):
    stored_data[data.name] = data.points
    return {"res": "ok", "name": data.name}


@app.put("/pointcloud/update/{name}")
async def update_data(name: str, data: PointCloudData):
    stored_data[data.name] = data.points
    return {"res": "ok", "name": data.name}


def main():
    import uvicorn
    import argparse
    parser = argparse.ArgumentParser(description='Visualization server for p3dpy.')
    parser.add_argument('--host', type=str, default='127.0.0.1', help="Host address.")
    parser.add_argument('--port', type=int, default=8000, help="Port number.")
    args = parser.parse_args()
    uvicorn.run(app=app, host=args.host, port=args.port)

if __name__ == '__main__':
    main()
