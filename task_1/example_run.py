import pandas as pd
import torch
import torch.nn as nn

from torch.utils.data import Dataset
from torchvision import models
from typing import Tuple

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 1
MEMBERSHIP_DATASET_PATH = "data/priv_out.pt"       # Path to priv_out_.pt
MIA_CKPT_PATH = "data/01_MIA_69.pt"                 # Path to 01_MIA_69.pt
PUB_DATASET_PATH = "data/pub.pt"              # Path to pub.pt


allowed_models = {
    "resnet18": models.resnet18,
    "resnet34": models.resnet34,
    "resnet50": models.resnet50,
}


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


def inference_dataloader(dataset: MembershipDataset, batch_size):
    return torch.utils.data.DataLoader(dataset, batch_size, shuffle=False)


def load_model(model_name, model_path):
    try:
        model: nn.Module = allowed_models[model_name](weights=None)
        model.fc = nn.Linear(model.fc.weight.shape[1], 44)
    except Exception as e:
        raise Exception(
            f"Invalid model class, {e}, only {allowed_models.keys()} are allowed"
        )

    try:
        model_state_dict = torch.load(model_path, map_location=DEVICE, weights_only=False)
        model.load_state_dict(model_state_dict, strict=True)
        model.eval()
    except Exception as e:
        raise Exception(f"Invalid model, {e}")

    return model


def membership_prediction(model, dataset_path):
    with torch.serialization.safe_globals([MembershipDataset]):
        dataset: MembershipDataset = torch.load(dataset_path)
    dataloader = inference_dataloader(dataset, BATCH_SIZE)

    outputs_list = []

    for _, img, _, _ in dataloader:
        img = img.to(DEVICE)

        with torch.no_grad():
            membership_output = model(img)

        outputs_list += membership_output.tolist()

    return pd.DataFrame(
        {
            "ids": dataset.ids,
            "score": outputs_list,
        }
    )


if __name__ == '__main__':
    model = load_model(model_name="resnet18", model_path=MIA_CKPT_PATH)
    preds = membership_prediction(model, PUB_DATASET_PATH)
    preds.to_csv("data/pub_submission.csv", index=False)

    print("Outputs saved to pub_submission.csv")