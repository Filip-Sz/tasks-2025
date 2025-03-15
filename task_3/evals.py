from adversary import *
import typing
import torch
from torch.utils.data import Dataset, DataLoader
import tqdm

def evaluate(
    model: torch.nn.Module,
    dataloader: Dataset | DataLoader,
    description: str = "eval",
    device: str = "cuda",
    quiet: bool = False,
) -> float:
    """Evaluate a model on a dataset or dataloader, returning the accuracy (0..1)."""
    model = model.to(device).eval()

    clean, clean_done, clean_correct = 0.0, 0, 0
    fgsm, fgsm_done, fgsm_correct = 0.0, 0, 0
    pgd, pgd_done, pgd_correct = 0.0, 0, 0


    progress_bar = tqdm(dataloader, desc=description, disable=quiet, delay=0.5)
    with torch.no_grad(), progress_bar:
        for i, image_batch, label_batch in progress_bar:
            logits_batch = model(image_batch.to(device))
            predictions = logits_batch.argmax(dim=1)

            clean_done += len(label_batch)
            clean_correct += (predictions == label_batch.to(device)).sum().item()
            clean = clean_correct / clean_done if clean_done else 0

            fgsm_batch = FGSM(model, image_batch, label_batch)
            logits_batch = model(fgsm_batch.to(device))
            predictions = logits_batch.argmax(dim=1)

            fgsm_done += len(label_batch)
            fgsm_correct += (predictions == label_batch.to(device)).sum().item()
            fgsm = fgsm_correct / fgsm_done if fgsm_done else 0

            pgd_batch = PGD(model, image_batch, label_batch)
            logits_batch = model(pgd_batch.to(device))
            predictions = logits_batch.argmax(dim=1)

            pgd_done += len(label_batch)
            pgd_correct += (predictions == label_batch.to(device)).sum().item()
            pgd = pgd_correct / pgd_done if pgd_done else 0

            progress_bar.set_postfix({"Clean": f"{clean * 100:.1f} %", "FGSM": f"{fgsm * 100:.1f} %", "PGD": f"{pgd * 100:.1f} %"})

    return clean, fgsm, pgd