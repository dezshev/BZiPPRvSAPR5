"""Рисунки для README: топология сети и восстановление образов при разном шуме.

Запуск: python tools/make_docs_images.py
"""

import math
import os
import random
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Circle, FancyArrowPatch

from hopfield.network import Network, add_noise, single_trial
from hopfield.patterns import default_patterns

OUT = os.path.join(ROOT, "docs", "img")
ACCENT = "#1f6feb"
DARK = "#16233a"
GOOD = "#1a7f37"
BAD = "#b3261e"
PIXELS = ListedColormap(["#ffffff", "#16233a"])


def topology() -> None:
    """Топология сети: каждый нейрон связан со всеми остальными."""
    figure, axes = plt.subplots(figsize=(7.5, 6.4))
    count = 8
    radius = 2.4
    points = []
    for index in range(count):
        angle = 2 * math.pi * index / count - math.pi / 2
        points.append((radius * math.cos(angle), radius * math.sin(angle)))

    for i in range(count):
        for j in range(i + 1, count):
            axes.plot([points[i][0], points[j][0]], [points[i][1], points[j][1]],
                      color="#c7d6ee", linewidth=0.9, zorder=1)
    for index, (x, y) in enumerate(points):
        axes.add_patch(Circle((x, y), 0.42, facecolor="white", edgecolor=ACCENT,
                              linewidth=1.8, zorder=3))
        axes.text(x, y, f"a{index + 1}", ha="center", va="center", fontsize=10,
                  color=DARK, zorder=4)
        outer_x, outer_y = x * 1.42, y * 1.42
        axes.add_patch(FancyArrowPatch((x, y), (outer_x, outer_y), arrowstyle="-|>",
                                       mutation_scale=12, color="#6b7c93", linewidth=1.2, zorder=2))

    axes.text(0, 0.35, "каждый нейрон связан\nсо всеми остальными", ha="center", va="center",
              fontsize=11, color=DARK)
    axes.text(0, -0.55, "w(i, j) = w(j, i),  w(i, i) = 0", ha="center", va="center",
              fontsize=10.5, color=ACCENT)
    axes.set_title("Топология сети Хопфилда", fontsize=13, color=DARK, fontweight="bold")
    axes.set_xlim(-4, 4)
    axes.set_ylim(-3.6, 3.6)
    axes.set_aspect("equal")
    axes.axis("off")
    figure.tight_layout()
    figure.savefig(os.path.join(OUT, "topology.png"), dpi=150, facecolor="white")
    plt.close(figure)


def _draw(axes, cells, size, title, color=DARK):
    grid = [[1 if cells[row * size + column] > 0 else 0 for column in range(size)]
            for row in range(size)]
    axes.imshow(grid, cmap=PIXELS, vmin=0, vmax=1)
    axes.set_xticks([])
    axes.set_yticks([])
    axes.set_title(title, fontsize=9.5, color=color)
    for spine in axes.spines.values():
        spine.set_edgecolor("#9fb3cd")


def restoration() -> None:
    """Восстановление буквы при разном уровне зашумления."""
    patterns = default_patterns("ДЕШ")
    network = Network.train(patterns)
    levels = [20, 40, 50, 70]
    figure, panels = plt.subplots(len(levels), 3, figsize=(7.6, 2.5 * len(levels)))
    for row, percent in enumerate(levels):
        generator = random.Random(percent)
        trial = single_trial(network, 0, percent, generator)
        _draw(panels[row][0], patterns[0].cells, patterns[0].size, "эталон «Д»")
        _draw(panels[row][1], trial.noisy.cells, patterns[0].size, f"шум {percent} %")
        color = GOOD if trial.correct else (ACCENT if trial.inverted else BAD)
        _draw(panels[row][2], trial.result, patterns[0].size,
              f"после сети: {trial.verdict}", color)
    figure.suptitle("Что возвращает сеть при разном зашумлении", fontsize=12, color=DARK,
                    fontweight="bold")
    figure.tight_layout()
    figure.savefig(os.path.join(OUT, "restoration.png"), dpi=150, facecolor="white")
    plt.close(figure)


def patterns_picture() -> None:
    """Три запомненных образа рядом."""
    patterns = default_patterns("ДЕШ")
    figure, panels = plt.subplots(1, len(patterns), figsize=(2.6 * len(patterns), 3.0))
    for panel, pattern in zip(panels, patterns):
        _draw(panel, pattern.cells, pattern.size,
              f"«{pattern.name}»: {pattern.filled()} из {pattern.size ** 2} клеток")
    figure.suptitle("Образы для обучения сети (буквы фамилии)", fontsize=12, color=DARK,
                    fontweight="bold")
    figure.tight_layout()
    figure.savefig(os.path.join(OUT, "letters.png"), dpi=150, facecolor="white")
    plt.close(figure)


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    topology()
    patterns_picture()
    restoration()
    print("рисунки сохранены в", OUT)


if __name__ == "__main__":
    main()
