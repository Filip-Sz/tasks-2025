import torch
import typing
import random

def FGSM(
     model: torch.nn.Module,
     x: torch.Tensor,
     y: torch.Tensor,
     epsilon: float = 1/255,
 ) -> torch.Tensor: 
    was_training = model.training
    model.eval()
    x = x.clone().detach()

    x.requires_grad = True

    logits = model(x)
    loss = torch.nn.CrossEntropyLoss()(logits, y)
    x_grad = torch.autograd.grad(loss, x)[0].sign()

    perturbation = x_grad*epsilon
    perturbated_img = torch.clamp(x + perturbation, 0, 1).detach()

    if was_training:
        model.train()

    return perturbated_img.requires_grad_(False)


def PGD(
    model: torch.nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    epsilon: float = 2 / 255,
    n_iters: int = 1,
    random_init: bool = False,
) -> torch.Tensor:
    was_training = model.training
    model.eval()
    x = x.clone().detach()
    with torch.no_grad():
        x_min = torch.clamp(x - 2*epsilon, min=0)
        x_max = torch.clamp(x + 2*epsilon, max=1)

        if random_init:
            x = x + torch.empty_like(x).uniform_(-2*epsilon, 2*epsilon)
            x.clamp_(x_min, x_max)

    for _ in range(n_iters):
        x.requires_grad = True
        logits = model(x)
        loss = torch.nn.CrossEntropyLoss()(logits, y)
        with torch.no_grad():
            x_grad = torch.autograd.grad(loss, x)[0]
            x = x.detach() + epsilon * x_grad.sign()
            x.clamp_(x_min, x_max)
            x = x.detach()

    if was_training:
        model.train()
    return x.requires_grad_(False)

def get_adversary_dataset(model, dataloader, DEVICE):
    pgds = []
    fgsms = []
    for i, x, y in dataloader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        pgds.append(PGD(model, x, y, n_iters=random.choice([1, 2, 3, 4]), epsilon=random.choice([1, 2, 3, 4])/255))
        fgsms.append(FGSM(model, x, y, epsilon=random.choice([1, 2, 3, 4])/255))
    
    return torch.stack(pgds), torch.stack(fgsms)