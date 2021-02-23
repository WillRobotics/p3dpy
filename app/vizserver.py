from typing import List
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from starlette.templating import Jinja2Templates
from pydantic import BaseModel
import uvicorn


class PointCloudData(BaseModel):
    name: str
    points: List[List[float]]


app = FastAPI()
stored_data = {}

app.mount("/", StaticFiles(directory="static", html=True), name="static")

@app.get("/{name}")
async def get_data(name: str):
    json_data = jsonable_encoder(stored_data[name])
    return JSONResponse(content=json_data)


@app.post("/store")
async def store_data(data: PointCloudData):
    stored_data[data.name] = data.points
    return {"res": "ok", "name": data.name}


@app.put("/update/{name}")
async def update_data(name: str, data: PointCloudData):
    stored_data[data.name].extend(data.points)
    return {"res": "ok", "name": data.name}


if __name__ == '__main__':
    uvicorn.run(app=app)