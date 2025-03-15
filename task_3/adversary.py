import torch
import typing

MEAN = torch.Tensor([0.2980, 0.2962, 0.2987])
STD = torch.Tensor([0.2886, 0.2875, 0.2889])

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
    epsilon: float = 4 / 255,
    alpha: float = 2 / 255,
    n_iters: int = 1,
    random_init: bool = False,
) -> torch.Tensor:
    was_training = model.training
    model.eval()
    x = x.clone().detach()
    with torch.no_grad():
        x_min = torch.clamp(x - epsilon, min=0)
        x_max = torch.clamp(x + epsilon, max=1)

        if random_init:
            x = x + torch.empty_like(x).uniform_(-epsilon, epsilon)
            x.clamp_(x_min, x_max)

    for _ in range(n_iters):
        x.requires_grad = True
        logits = model(x)
        loss = torch.nn.CrossEntropyLoss()(logits, y)
        with torch.no_grad():
            x_grad = torch.autograd.grad(loss, x)[0]
            x = x.detach() + alpha * x_grad.sign()
            x.clamp_(x_min, x_max)
            x = x.detach()

    if was_training:
        model.train()
    return x.requires_grad_(False)