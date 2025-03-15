import copy
import os

import pandas as pd
import torch
import torch.nn as nn
from adversary import *
from torchvision.transforms import v2
from tqdm import tqdm

"""
source: https://github.com/jeffheaton/app_deep_learning/blob/main/t81_558_class_03_4_early_stop.ipynb
"""


class EarlyStopping:
    def __init__(self, patience=5, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.best_model = None
        self.best_loss = None
        self.counter = 0
        self.status = ""

    def __call__(self, model, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_model = copy.deepcopy(model.state_dict())
        elif self.best_loss - val_loss > self.min_delta:
            self.best_model = copy.deepcopy(model.state_dict())
            self.best_loss = val_loss
            self.counter = 0
            # self.status = f"Improvement found, counter reset to {self.counter}"
        else:
            self.counter += 1
            # self.status = f"No improvement in the last {self.counter} epochs"
            if self.counter >= self.patience:
                # self.status = f"Early stopping triggered after {self.counter} epochs."
                model.load_state_dict(self.best_model)
                return True
        return False


"""
source: https://www.learnpytorch.io/04_pytorch_custom_datasets/#75-create-train-test-loop-functions
"""


def train_step(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str,
):

    model.train()
    train_loss, train_total = 0, 0
    train_score, train_score_fgsm, train_score_pgd = 0, 0, 0
    clean_loss_total, fgsm_loss_total, pgd_loss_total = 0, 0, 0
    for _, X, y in dataloader:
        X, y = X.to(device), y.to(device)
        fgsm = FGSM(model, X, y)
        pgd = PGD(model, X, y)

        optimizer.zero_grad()

        y_pred = model(X)
        clean_loss = loss_fn(y_pred, y)

        fgsm_pred = model(fgsm)
        fgsm_loss = loss_fn(fgsm_pred, y)

        pgd_pred = model(pgd)
        pgd_loss = loss_fn(pgd_pred, y)

        loss = 0.5 * clean_loss + 0.25 * fgsm_loss + 0.25 * pgd_loss
        train_loss += loss.item()
        clean_loss_total += clean_loss.item()
        fgsm_loss_total += fgsm_loss.item()
        pgd_loss_total += pgd_loss.item()

        loss.backward()
        optimizer.step()

        train_score += (torch.argmax(y_pred, dim=1) == y).sum().item()
        train_score_fgsm += (torch.argmax(fgsm_pred, dim=1) == y).sum().item()
        train_score_pgd += (torch.argmax(pgd_pred, dim=1) == y).sum().item()
        train_total += len(y)

    train_loss = train_loss
    train_score = train_score / train_total
    train_score_fgsm = train_score_fgsm / train_total
    train_score_pgd = train_score_pgd / train_total

    return (
        train_loss,
        train_score,
        train_score_fgsm,
        train_score_pgd,
        clean_loss_total,
        fgsm_loss_total,
        pgd_loss_total,
    )


"""
source: https://www.learnpytorch.io/04_pytorch_custom_datasets/#75-create-train-test-loop-functions
"""


def test_step(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    loss_fn: nn.Module,
    device: str,
):

    model.eval()
    test_loss, test_total = 0, 0
    test_score, test_score_fgsm, test_score_pgd = 0, 0, 0
    clean_loss_total, fgsm_loss_total, pgd_loss_total = 0, 0, 0
    # with torch.inference_mode():
    for _, X, y in dataloader:
        X, y = X.to(device), y.to(device)
        fgsm = FGSM(model, X, y)
        pgd = PGD(model, X, y)

        y_pred = model(X)
        clean_loss = loss_fn(y_pred, y)

        fgsm_pred = model(fgsm)
        fgsm_loss = loss_fn(fgsm_pred, y)

        pgd_pred = model(pgd)
        pgd_loss = loss_fn(pgd_pred, y)

        loss = 0.5 * clean_loss + 0.25 * fgsm_loss + 0.25 * pgd_loss
        test_loss += loss.item()
        clean_loss_total += clean_loss.item()
        fgsm_loss_total += fgsm_loss.item()
        pgd_loss_total += pgd_loss.item()

        test_score += (torch.argmax(y_pred, dim=1) == y).sum().item()
        test_score_fgsm += (torch.argmax(fgsm_pred, dim=1) == y).sum().item()
        test_score_pgd += (torch.argmax(pgd_pred, dim=1) == y).sum().item()
        test_total += len(y)

    test_loss = test_loss
    test_score = test_score / test_total
    test_score_fgsm = test_score_fgsm / test_total
    test_score_pgd = test_score_pgd / test_total

    return (
        test_loss,
        test_score,
        test_score_fgsm,
        test_score_pgd,
        clean_loss_total,
        fgsm_loss_total,
        pgd_loss_total,
    )


def save_model(model, save_path, file_name):
    torch.save(obj=model.state_dict(), f=os.path.join(save_path, file_name))


def save_training_results(results, save_path, file_name):
    results.to_csv(os.path.join(save_path, file_name), index=False, sep=",")


def train(
    model: nn.Module,
    train_dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: str,
    test_dataloader: torch.utils.data.DataLoader = None,
    epochs: int = 20,
    save_n_epochs: int = 5,
    save_dir: str = "./training_logs",
    verbose: bool = True,
    use_early_stopping: bool = True,
    patience: int = 5,
    min_delta: int = 0,
):

    results = {
        "epoch": [],
        "train_loss": [],
        "train_score": [],
        "train_score_fgsm": [],
        "train_score_pgd": [],
        "train_clean_loss": [],
        "train_fgsm_loss": [],
        "train_pgd_loss": [],
        "test_loss": [],
        "test_score": [],
        "test_score_fgsm": [],
        "test_score_pgd": [],
        "test_clean_loss": [],
        "test_fgsm_loss": [],
        "test_pgd_loss": [],
    }
    results = pd.DataFrame(results)

    # Creating directories for saving results and models
    results_dir_path = os.path.join(save_dir, "results")
    models_dir_path = os.path.join(save_dir, "models")
    os.mkdir(results_dir_path)
    os.mkdir(models_dir_path)

    # Set early stopping object
    early_stop = EarlyStopping(patience=patience, min_delta=min_delta)
    early_stopping_used = False
    use_early_stopping = (test_dataloader is not None) and use_early_stopping

    progress = tqdm(range(epochs))

    for epoch in progress:
        (
            train_loss,
            train_score,
            train_score_fgsm,
            train_score_pgd,
            train_clean_loss,
            train_fgsm_loss,
            train_pgd_loss,
        ) = train_step(
            model=model,
            dataloader=train_dataloader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=device,
        )
        if not test_dataloader == None:
            (
                test_loss,
                test_score,
                test_score_fgsm,
                test_score_pgd,
                test_clean_loss,
                test_fgsm_loss,
                test_pgd_loss,
            ) = test_step(
                model=model, dataloader=test_dataloader, loss_fn=loss_fn, device=device
            )
        else:
            test_loss = None
            test_score = None

        if verbose:
            if not test_dataloader == None:
                progress.set_description(
                    f"Epoch: {epoch+1} | "
                    f"test_score: {test_score:.4f} | "
                    f"test_score_fgsm: {test_score_fgsm:.4f} | "
                    f"test_score_pgd: {test_score_pgd:.4f}"
                )
            else:
                progress.set_description(
                    f"Epoch: {epoch+1} | " f"train_score: {train_score:.4f}"
                )

        results.loc[len(results.index)] = [
            int(epoch + 1),
            train_loss,
            train_score,
            train_score_fgsm,
            train_score_pgd,
            train_clean_loss,
            train_fgsm_loss,
            train_pgd_loss,
            test_loss,
            test_score,
            test_score_fgsm,
            test_score_pgd,
            test_clean_loss,
            test_fgsm_loss,
            test_pgd_loss,
        ]

        # Saving training results
        save_training_results(results, results_dir_path, f"training_results.csv")
        if (epoch + 1) % save_n_epochs == 0 and not (epoch + 1 == epochs):
            save_model(model, models_dir_path, f"model_epoch_{epoch+1}")

        if use_early_stopping and early_stop(model, test_loss):
            early_stopping_used = True
            save_training_results(
                results, results_dir_path, f"training_results_early_stopping.csv"
            )
            save_model(model, models_dir_path, "model_early_stopping")
            print("Early stopping!")
            break

    if not early_stopping_used:
        save_training_results(results, results_dir_path, "training_results_final.csv")
        save_model(model, models_dir_path, "model_final")
        model.load_state_dict(early_stop.best_model)
        save_model(model, models_dir_path, "model_best")
