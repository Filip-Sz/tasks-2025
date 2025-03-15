import datetime
import os

import torch
import yaml
from adversary import *
from taskdata import *
from torch.utils.data import DataLoader
from torchvision.models import resnet50
from train_utils import *


def main():
    with open("config.yaml", "r") as file:
        config = yaml.safe_load(file)

    DEVICE = "cpu"

    model = resnet50(weights="IMAGENET1K_V1")
    model.fc = torch.nn.Linear(model.fc.weight.shape[1], 10)

    data = torch.load("data.pt", weights_only=False)
    data.transform = transforms.Compose(
        [
            transforms.Resize((32, 32)),
            transforms.Lambda(lambda x: x.convert("RGB")),
            transforms.ToTensor(),
        ]
    )

    # train_data, test_data = torch.utils.data.random_split(data, [99000, 1000])
    train_data, test_data = torch.utils.data.random_split(data, [99990, 10])
    train_loader = DataLoader(
        train_data,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
    )
    test_loader = DataLoader(
        test_data,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    criterion = torch.nn.CrossEntropyLoss()

    model.to(DEVICE)

    num_epochs = config["num_epochs"]
    save_dir = config["save_dir"]
    use_early_stopping = config["use_early_stopping"]
    patience = config["patience"]
    min_delta = config["min_delta"]

    save_dir = os.path.join(save_dir, str(datetime.datetime.now()))
    os.mkdir(save_dir)

    lines = [
        f"Optimizer: {type(optimizer).__name__}",
        f"Device: {DEVICE}",
        f"epochs: {num_epochs}",
        f'Learning rate: {optimizer.state_dict()["param_groups"][0]["lr"]}',
        f'L2 penalty: {optimizer.state_dict()["param_groups"][0]["weight_decay"]}',
        f'Batch size: {config["batch_size"]}',
        f"use_early_stopping: {use_early_stopping}",
        f"patience: {patience}",
        f"min_delta: {min_delta}",
    ]

    with open(os.path.join(save_dir, "hyperparams.txt"), "w") as f:
        f.write("\n".join(lines))

    # Training model
    train(
        model=model,
        train_dataloader=test_loader,
        optimizer=optimizer,
        loss_fn=criterion,
        device=DEVICE,
        test_dataloader=test_loader,
        epochs=num_epochs,
        save_n_epochs=config["save_n_epochs"],
        save_dir=save_dir,
        verbose=True,
        use_early_stopping=use_early_stopping,
        patience=patience,
        min_delta=min_delta,
    )


if __name__ == "__main__":
    main()
