import torch
from torch.utils.data import Dataset
from typing import Tuple
import numpy as np
import requests
import pandas as pd

#### LOADING THE MODEL

from torchvision.models import resnet18

model = resnet18(pretrained=False)
model.fc = torch.nn.Linear(512, 44)

ckpt = torch.load("./01_MIA_67.pt", map_location="cpu")

model.load_state_dict(ckpt)

#### DATASETS

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


class MembershipDataset(TaskDataset):
    def __init__(self, transform=None):
        super().__init__(transform)
        self.membership = []

    def __getitem__(self, index) -> Tuple[int, torch.Tensor, int, int]:
        id_, img, label = super().__getitem__(index)
        return id_, img, label, self.membership[index]


data: MembershipDataset = torch.load("./priv_out.pt")
    
#### EXAMPLE SUBMISSION

df = pd.DataFrame(
    {
        "ids": data.ids,
        "score": np.random.randn(len(data.ids)),
    }
)
df.to_csv("test.csv", index=None)
response = requests.post("http://35.184.239.3:9090/mia", files={"file": open("test.csv", "rb")}, headers={"token": "TOKEN"})
print(response.json())
