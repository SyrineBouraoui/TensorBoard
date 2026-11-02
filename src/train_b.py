"""Entraîne le détecteur (option B), évalue (IoU, % IoU > 0.5, accuracy de classe, matrice de confusion)."""

import torch
from torch.utils.data import DataLoader

from src.detector_b import Detector, N_CLASSES, compute_loss, iou, predict


def train(model: Detector, loader: DataLoader, optimizer: torch.optim.Optimizer) -> float:
    """Entraîne le modèle sur une époque et retourne la perte moyenne par image."""
    model.train()
    total_loss = 0.0
    for images, targets in loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = compute_loss(outputs, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)   
    return total_loss / len(loader.dataset)          


def evaluate(model: Detector, loader: DataLoader, size: int, n_classes: int = N_CLASSES):
    """Retourne (IoU moyen, proportion d'IoU > 0.5, accuracy de classe, matrice de confusion).

    Matrice de confusion : lignes = vraie classe, colonnes = classe prédite.
    """
    model.eval()
    ious = []
    confusion = torch.zeros(n_classes, n_classes, dtype=torch.long)
    with torch.no_grad():
        for images, targets in loader:
            logits = model(images)
            grid = targets.shape[-1]
            for logit, target in zip(logits, targets):
                box, _, pred_cls = predict(logit, size)
                row, col = divmod(int(target[0].argmax()), grid)
                _, dx, dy, w, h = target[:5, row, col].tolist()
                true_cls = int(target[5:, row, col].argmax())
                cx, cy = (col + dx) * size / grid, (row + dy) * size / grid
                truth = torch.tensor([cx - w * size / 2, cy - h * size / 2,
                                      cx + w * size / 2, cy + h * size / 2])
                ious.append(float(iou(box, truth[None])[0]))
                confusion[true_cls, pred_cls] += 1
    t = torch.tensor(ious)
    accuracy = confusion.diag().sum().item() / confusion.sum().item()
    return t.mean().item(), (t > 0.5).float().mean().item(), accuracy, confusion