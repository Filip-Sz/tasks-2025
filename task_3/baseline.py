import torch
from torch.utils.data import DataLoader
from torchvision.models import resnet50
from taskdata import *
from adversary import *

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

train_data, test_data = torch.utils.data.random_split(data, [90000, 10000])
train_loader = DataLoader(train_data, batch_size=100, shuffle=True, num_workers=2)
rest_loader = DataLoader(test_data, batch_size=100, shuffle=True, num_workers=2)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.CrossEntropyLoss()

n_epochs = 10

n_epochs = 3

model.to(DEVICE)
for epoch in range(n_epochs):
    running_loss = 0
    correct = 0
    total = 0

    test_correct = 0
    test_total = 0

    model.train()
    for i, x, y in train_loader:
        x = x.to(DEVICE)
        y = y.to(DEVICE)

        optimizer.zero_grad()
        pred = model(x)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()

        correct += (torch.argmax(pred, dim=1) == y).sum().item()
        total += len(y)

        running_loss += loss.item()

    model.eval()
    with torch.no_grad():
        for i, x, y in train_loader:
            x = x.to(DEVICE)
            y = y.to(DEVICE)
            correct += (torch.argmax(pred, dim=1) == y).sum().item()
            total += len(y)
            running_loss += loss.item()
    
    print(f"{epoch:0>2} Loss: {running_loss:.3f} | Accuracy: {correct/total: .3f}")

torch.save(model.state_dict(), 'models/baseline.pt')