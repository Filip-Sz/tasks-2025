
import torch
from torch.utils.data import DataLoader
from torchvision.models import resnet50
from taskdata import *
from adversary import *
import os
from train_utils import *
import datetime

def main():
    DEVICE = "cuda"

    model = resnet50(weights='IMAGENET1K_V1')
    model.fc = torch.nn.Linear(model.fc.weight.shape[1], 10)

    data = torch.load('data.pt', weights_only=False)
    data.transform = transforms.Compose(
        [
            transforms.Resize((32, 32)),
            transforms.Lambda(lambda x: x.convert("RGB")),
            transforms.ToTensor(),
        ]
    )

    batch_size = 64
    train_data, test_data = torch.utils.data.random_split(data, [90000, 10000])
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True, num_workers=2)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    criterion = torch.nn.CrossEntropyLoss()

    num_epochs = 50

    model.to(DEVICE)
    
    # Saving hyperparameters to file
    save_dir = './training_logs'

    save_dir = os.path.join(save_dir, str(datetime.datetime.now()))
    os.mkdir(save_dir)

    use_early_stopping = True
    patience = 5
    min_delta = 0

    lines = [f'Optimizer: {type(optimizer).__name__}',
             f'Device: {DEVICE}',
             f'epochs: {num_epochs}',
             f'Learning rate: {optimizer.state_dict()["param_groups"][0]["lr"]}',
             f'L2 penalty: {optimizer.state_dict()["param_groups"][0]["weight_decay"]}',
             f'Batch size: {batch_size}',
             f'use_early_stopping: {use_early_stopping}',
             f'patience: {patience}',
             f'min_delta: {min_delta}']

    with open(os.path.join(save_dir, 'hyperparams.txt'), 'w') as f:
        f.write('\n'.join(lines))

    # Training model
    train(model=model,
        train_dataloader=train_loader,
        optimizer=optimizer,
        loss_fn=criterion,
        device=DEVICE,
        test_dataloader=test_loader,
        epochs=num_epochs,
        save_n_epochs=5,
        save_dir=save_dir,
        verbose=True,
        use_early_stopping=use_early_stopping,
        patience=patience,
        min_delta=min_delta)

if __name__ == "__main__":
    main()