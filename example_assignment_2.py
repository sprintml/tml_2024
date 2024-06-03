import requests
import torch
import torch.nn as nn
# Do install: 
# conda install onnx
# conda install onnxruntime
import onnxruntime as ort
import numpy as np
import json
import io
import sys
import base64
from torch.utils.data import Dataset
from typing import Tuple
import pickle
import os

cwd = os.getcwd()
print('cwd: ', cwd)

class TaskDataset(Dataset):
    def __init__(self, transform=None):

        self.ids = []
        self.imgs = []
        self.labels = []

        self.transform = transform

    def __getitem__(self, index) -> Tuple[int, torch.Tensor, int]:
        id_ = self.ids[index]
        img = self.imgs[index]
        if not self.transform is None:
            img = self.transform(img)
        label = self.labels[index]
        return id_, img, label

    def __len__(self):
        return len(self.ids)

### REQUESTING NEW API ###
TOKEN = "1" # to be changed according to your token (given to you for the assignments)

response = requests.get("http://34.71.138.79:9090" + "/stealing_launch", headers={"token": TOKEN})
answer = response.json()

print(answer)  # {"seed": "SEED", "port": PORT}
if 'detail' in answer:
    sys.exit(1)

# save the values
SEED = str(answer['seed'])
PORT = str(answer['port'])

# SEED = "1868949"
# PORT = "9002"

### QUERYING THE API ###

def model_stealing(images, port):
    endpoint = "/query"
    url = f"http://34.71.138.79:{port}" + endpoint
    image_data = []
    for img in images:
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)
        img_base64 = base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')
        image_data.append(img_base64)
    
    payload = json.dumps(image_data)
    response = requests.get(url, files={"file": payload}, headers={"token": TOKEN})
    if response.status_code == 200:
        representation = response.json()["representations"]
        return representation
    else:
        raise Exception(
            f"Model stealing failed. Code: {response.status_code}, content: {response.json()}"
        )

dataset = torch.load("ModelStealingPub.pt")
out = model_stealing([dataset.imgs[idx] for idx in np.random.permutation(1000)], port=PORT)

# Store the output in a file.
# Be careful to store all the outputs from the API since the number of queries is limited.
with open('out.pickle', 'wb') as handle:
    pickle.dump(out, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Restore the output from the file.
with open('out.pickle', 'rb') as handle:
    out = pickle.load(handle)

print(len(out))

#### SUBMISSION ####

# Create a dummy model
model = nn.Sequential(nn.Flatten(), nn.Linear(32*32*3, 1024))

path = 'dummy_submission.onnx'

torch.onnx.export(
    model,
    torch.randn(1, 3, 32, 32),
    path,
    export_params=True,
    input_names=["x"],
)

#### Tests ####

# (these are being ran on the eval endpoint for every submission)
with open(path, "rb") as f:
    model = f.read()
    try:
        stolen_model = ort.InferenceSession(model)
    except Exception as e:
        raise Exception(f"Invalid model, {e=}")
    try:
        out = stolen_model.run(
            None, {"x": np.random.randn(1, 3, 32, 32).astype(np.float32)}
        )[0][0]
    except Exception as e:
        raise Exception(f"Some issue with the input, {e=}")
    assert out.shape == (1024,), "Invalid output shape"

# Send the model to the server
response = requests.post("http://34.71.138.79:9090/stealing", files={"file": open(path, "rb")}, headers={"token": TOKEN, "seed": SEED})
print(response.json())