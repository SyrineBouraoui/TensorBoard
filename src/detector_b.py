"""Détecteur option B : feature maps -> prédiction par cellule -> NMS -> une boîte + une classe.

Image 64x64 -> 3 blocs conv stride 2 -> feature map (C, 8, 8) (stride 8).
Une conv 1x1 prédit pour chaque cellule 5 + N_CLASSES valeurs :
  (objectness, dx, dy, w, h, score_cercle, score_carre, score_triangle)
  - dx, dy in [0,1] : décalage du centre dans la cellule
  - w, h in [0,1]   : taille relative à l'image
  - scores de classe : valeurs brutes (logits), softmax inclus dans la CrossEntropy
"""

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import Dataset

STRIDE = 8
N_CLASSES = 3
CLASS_NAMES = ("cercle", "carré", "triangle")   # ordre de dataset_generator2 : circle=0, square=1, triangle=2


POS_WEIGHT = 10.0     
BOX_WEIGHT = 10.0      
CLASS_WEIGHT = 1.0    


def iou(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """IoU entre une boîte a (4,) et des boîtes b (N, 4)."""
    inter_w = (torch.minimum(a[2], b[:, 2]) - torch.maximum(a[0], b[:, 0])).clamp(min=0)
    inter_h = (torch.minimum(a[3], b[:, 3]) - torch.maximum(a[1], b[:, 1])).clamp(min=0)
    inter = inter_w * inter_h
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (area_a + area_b - inter).clamp(min=1e-8)


def nms(boxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float = 0.5) -> list[int]:
    """Garde la meilleure boîte, supprime celles qui la chevauchent, recommence.
    Retourne les indices des boîtes conservées (seuil 0.5 : un seuil > 1 ne supprimerait rien)."""
    order = scores.argsort(descending=True)
    kept = []
    while len(order):
        best = order[0]
        kept.append(int(best))
        order = order[1:]
        order = order[iou(boxes[best], boxes[order]) < iou_threshold]
    return kept


def code(cx: float, cy: float, w: float, h: float, cls: int, size: int = 64) -> torch.Tensor:
    """Boîte [cx, cy, w, h] (pixels) + classe -> cible (5 + N_CLASSES, G, G).

    Canaux 0..4 : objectness, dx, dy, w/size, h/size (seulement dans la cellule du centre).
    Canaux 5.. : one-hot de la classe (seulement dans la cellule du centre).
    """
    grid = size // STRIDE
    gx, gy = cx / size * grid, cy / size * grid
    col, row = min(int(gx), grid - 1), min(int(gy), grid - 1)
    target = torch.zeros(5 + N_CLASSES, grid, grid)
    target[:5, row, col] = torch.tensor([1.0, gx - col, gy - row, w / size, h / size])
    target[5 + int(cls), row, col] = 1.0
    return target


def decode(logits: torch.Tensor, size: int = 64) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(5 + N_CLASSES, G, G) -> boîtes (G*G, 4) en pixels [x1, y1, x2, y2], scores (G*G,), classes (G*G,)."""
    grid = logits.shape[-1]
    p = torch.sigmoid(logits[:5])                     
    rows, cols = torch.meshgrid(torch.arange(grid), torch.arange(grid), indexing="ij")
    cx = (cols + p[1]) * STRIDE
    cy = (rows + p[2]) * STRIDE
    w, h = p[3] * size, p[4] * size
    boxes = torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=-1)
    classes = logits[5:].argmax(dim=0)             
    return boxes.reshape(-1, 4), p[0].reshape(-1), classes.reshape(-1)


class ShapeDataset(Dataset):
    """Cible (5 + N_CLASSES, G, G) : tout à 0 sauf la cellule contenant le centre.
    Fichier .txt : "cx cy w h class_id" (dataset_generator2)."""

    def __init__(self, folder: Path, size: int = 64) -> None:
        self.paths = sorted(Path(folder).glob("*.png"))
        if not self.paths:
            raise FileNotFoundError(f"Aucune image dans {folder} (lancer dataset_generator2).")
        self.grid = size // STRIDE

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        path = self.paths[i]
        image = torch.from_numpy(
            np.asarray(Image.open(path).convert("RGB"), dtype=np.float32).copy()
        ).permute(2, 0, 1) / 255.0
        cx, cy, w, h, cls = map(float, path.with_suffix(".txt").read_text().split())
        target = code(cx, cy, w, h, int(cls), size=image.shape[-1])
        return image, target


class Detector(nn.Module):
    def __init__(self, channels: tuple = (8, 16, 32), n_out: int = 5 + N_CLASSES) -> None:
        super().__init__()
        # Chaque entrée de `channels` = un bloc conv stride 2 -> stride total = 2 ** len(channels)
        assert 2 ** len(channels) == STRIDE, "STRIDE doit valoir 2 ** len(channels)"
        layers, c_in = [], 3
        for c_out in channels:
            layers += [nn.Conv2d(c_in, c_out, kernel_size=3, stride=2, padding=1),
                       nn.BatchNorm2d(c_out), nn.ReLU()]
            c_in = c_out
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Conv2d(c_in, n_out, kernel_size=1, stride=1)

    def feature_maps(self, images: torch.Tensor) -> torch.Tensor:
        return self.backbone(images)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.head(self.feature_maps(images))


def compute_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """BCE (présence, toutes les cellules) + MSE (boîte, cellule positive)
    + CrossEntropy (classe, cellule positive)."""
    obj_loss = nn.functional.binary_cross_entropy_with_logits(
        logits[:, 0], target[:, 0], pos_weight=torch.tensor(POS_WEIGHT))
    positive = target[:, 0] > 0.5                                              # (B, G, G)
    pred_box = torch.sigmoid(logits[:, 1:5]).permute(0, 2, 3, 1)[positive]     # (N, 4)
    true_box = target[:, 1:5].permute(0, 2, 3, 1)[positive]                    # (N, 4)
    pred_cls = logits[:, 5:].permute(0, 2, 3, 1)[positive]                     # (N, N_CLASSES) scores bruts
    true_cls = target[:, 5:].permute(0, 2, 3, 1)[positive].argmax(dim=-1)      # (N,) indices de classe
    return (obj_loss
            + BOX_WEIGHT * nn.functional.mse_loss(pred_box, true_box)
            + CLASS_WEIGHT * nn.functional.cross_entropy(pred_cls, true_cls))


def predict(logits: torch.Tensor, size: int = 64) -> tuple[torch.Tensor, float, int]:
    """Une seule figure par image : boîte, score et classe de la cellule de plus fort score après NMS."""
    boxes, scores, classes = decode(logits, size)
    best = nms(boxes, scores)[0]
    return boxes[best], float(scores[best]), int(classes[best])