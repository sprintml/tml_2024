import requests
import torch
import torch.nn as nn

#### SUBMISSION ####

# Create a dummy model
model = nn.Sequential(nn.Flatten(), nn.Linear(32*32*3, 1024))

torch.onnx.export(
    model,
    torch.randn(1, 3, 32, 32),
    "out/models/dummy_submission.onnx",
    export_params=True,
    input_names=["x"],
)

# Send the model to the server
response = requests.post("http://35.184.239.3:9090/stealing", files={"file": open("out/models/dummy_submission.onnx", "rb")}, headers={"token": "TOKEN", "seed": "SEED"})
print(response.json())