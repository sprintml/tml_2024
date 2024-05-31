import requests
import torch
import torch.nn as nn
# Do install: conda install onnx
import onnxruntime as ort
import numpy as np

#### SUBMISSION ####

# Create a dummy model
model = nn.Sequential(nn.Flatten(), nn.Linear(32*32*3, 1024))

path = 'dummy_submission.onnx'
TOKEN = "1" # to be changed according to your token (given to you for the assignments)
SEED = "1" # to be changed according to your seed (from the API endpoint that you run locally)

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