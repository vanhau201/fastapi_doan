from fastapi import FastAPI, UploadFile, File
import cv2
import torch
import numpy as np
import time
from typing import List

model = torch.hub.load('ultralytics/yolov5', 'custom',
                       path='model/best.pt')
app = FastAPI()


@app.post("/api")
async def index(file: List[UploadFile] = File(...)):
    start_time = time.time()
    data = []
    for f in file:
        contents = await f.read()
        nparr = np.fromstring(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        # predict

        img = img[..., ::-1]
        results = model(img)
        if len(results.pandas().xyxy[0]) > 0:
            for i in results.pandas().xyxy[0].values:
                data.append(i[6])
    print("Time :", time.time()-start_time)
    return {"data": data}


@app.post("/")
async def home(file: UploadFile = File(...)):
    start_time = time.time()
    data = []

    contents = await file.read()
    nparr = np.fromstring(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    # predict

    img = img[..., ::-1]
    results = model(img)
    if len(results.pandas().xyxy[0]) > 0:
        for i in results.pandas().xyxy[0].values:
            data.append(i[6])
    print("Time :", time.time()-start_time)
    return {"data": data}
