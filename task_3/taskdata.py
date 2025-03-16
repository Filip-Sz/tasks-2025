from typing import Tuple
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


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
    
T = transforms.Compose(
    [
        transforms.Resize((32, 32)),
        transforms.Lambda(lambda x: x.convert("RGB")),
        transforms.ToTensor()
    ]
)

def get_data_loaders(test_frac=0.1, PATH='data.pt'):
    data = torch.load('data.pt', weights_only=False)
    data.transform = transforms.Compose(
        [
            transforms.Resize((32, 32)),
            transforms.Lambda(lambda x: x.convert("RGB")),
            transforms.ToTensor()
        ]
    )
    train_data, test_data = torch.utils.data.random_split(data, [len(data)-int(test_frac*len(data)), int(test_frac*len(data))])
    train_loader = DataLoader(train_data, batch_size=100, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_data, batch_size=100, shuffle=True, num_workers=2)

    return train_loader, test_loader