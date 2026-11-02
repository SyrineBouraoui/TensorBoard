"""Visualisation : images, feature maps, boîtes prédites (avec classe) et matrice de confusion."""

import matplotlib.pyplot as plt
import torch


def show_images(images: list, title: str = "Images", ncols: int = 5) -> None:
    """Affiche des images PIL ou des tenseurs (3, H, W)."""
    nrows = -(-len(images) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.6 * ncols, 2.6 * nrows), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for ax, img in zip(axes.flat, images):
        ax.imshow(img.permute(1, 2, 0) if torch.is_tensor(img) else img)
    fig.suptitle(title)
    fig.tight_layout()
    plt.show()


def show_feature_maps(image: torch.Tensor, maps: torch.Tensor, n_maps: int = 8) -> None:
    """Image d'entrée + les n premiers canaux de la feature map (C, G, G)."""
    fig, axes = plt.subplots(1, n_maps + 1, figsize=(2.2 * (n_maps + 1), 2.4))
    axes[0].imshow(image.permute(1, 2, 0))
    axes[0].set_title("image")
    for k in range(n_maps):
        axes[k + 1].imshow(maps[k].detach(), cmap="viridis")
        axes[k + 1].set_title(f"canal {k}")
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    plt.show()


def show_detections(images: list[torch.Tensor], boxes: list[torch.Tensor],
                    scores: list[float], labels: list[str] | None = None,
                    ncols: int = 5) -> None:
    """Images avec la boîte finale [x1, y1, x2, y2] (pixels), son score et (option B) sa classe prédite."""
    nrows = -(-len(images) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.6 * ncols, 2.8 * nrows), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for i, (ax, img, box, score) in enumerate(zip(axes.flat, images, boxes, scores)):
        x1, y1, x2, y2 = box.tolist()
        ax.imshow(img.permute(1, 2, 0))
        ax.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, color="lime", lw=2))
        name = f"{labels[i]} " if labels else ""
        ax.set_title(f"{name}{score:.2f}", fontsize=9)
    fig.tight_layout()
    plt.show()


def show_confusion(confusion: torch.Tensor, class_names: tuple, title: str = "Matrice de confusion") -> None:
    """Matrice de confusion (lignes = vraie classe, colonnes = classe prédite)."""
    n = len(class_names)
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(confusion, cmap="Blues")
    ax.set_xticks(range(n), class_names)
    ax.set_yticks(range(n), class_names)
    ax.set_xlabel("classe prédite")
    ax.set_ylabel("vraie classe")
    ax.set_title(title)
    top = confusion.max().item()
    for i in range(n):
        for j in range(n):
            v = int(confusion[i, j])
            ax.text(j, i, v, ha="center", va="center",
                    color="white" if v > top / 2 else "black")
    fig.tight_layout()
    plt.show()