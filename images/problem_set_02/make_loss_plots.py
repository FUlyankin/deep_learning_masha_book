"""Параболы функции потерь для задач 1 и 5 листочка 2.

Запуск: python3.11 make_loss_plots.py
Генерирует img01_loss_theory.png, img01_loss_gd.png, img01_loss_sgd.png,
img05_lr_regimes.png в стиле исходных matplotlib-картинок книги.
"""
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.style.use("seaborn-v0_8-whitegrid")

BLACK = "#111111"
BLUE = "#0072B2"
CRIMSON = "#C2185B"
RED = "#D55E00"

DPI = 150


def mse(w):
    return 2.5 * w**2 - 8 * w + 6.5


def base_parabola(ax):
    w = np.linspace(-0.25, 3.45, 300)
    ax.plot(w, mse(w), color=BLACK, lw=2.2, zorder=2)
    ax.set_xlabel("$w$", fontsize=13)
    ax.set_ylabel("$MSE(w)$", fontsize=13)
    ax.set_ylim(-0.45, 8.4)


def save(fig, name):
    fig.tight_layout()
    fig.savefig(name, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("saved", name)


# ---------- теоретический минимум ----------
fig, ax = plt.subplots(figsize=(5, 3.4))
base_parabola(ax)
ax.scatter([1.6], [mse(1.6)], s=90, color=CRIMSON, zorder=5)
ax.annotate(
    "$\\hat{w} = 1.6$", (1.6, mse(1.6)), xytext=(1.85, 0.75), fontsize=12
)
save(fig, "img01_loss_theory.png")

# ---------- шаги GD ----------
steps_gd = np.array([0, 0.8, 1.2, 1.4])
fig, ax = plt.subplots(figsize=(5, 3.4))
base_parabola(ax)
ax.scatter(steps_gd, mse(steps_gd), s=55, color=BLUE, zorder=5)
ax.scatter([1.6], [mse(1.6)], s=90, color=CRIMSON, zorder=4)
for i, w in enumerate(steps_gd):
    ax.annotate(
        f"$w_{i}$", (w, mse(w)), xytext=(w + 0.12, mse(w) + 0.35), fontsize=11
    )
save(fig, "img01_loss_gd.png")

# ---------- шаги SGD ----------
steps_sgd = np.array([0, 0.4, 1.28, 1.424, 1.4848])
fig, ax = plt.subplots(figsize=(5, 3.4))
base_parabola(ax)
ax.scatter(steps_sgd, mse(steps_sgd), s=55, color=RED, zorder=5)
ax.scatter([1.6], [mse(1.6)], s=90, color=CRIMSON, zorder=4)
for i, w in zip([0, 1, 2, 4], steps_sgd[[0, 1, 2, 4]]):
    dx, dy = (0.12, 0.35) if i < 4 else (0.1, -0.55)
    ax.annotate(f"$w_{i}$", (w, mse(w)), xytext=(w + dx, mse(w) + dy), fontsize=11)
save(fig, "img01_loss_sgd.png")

# ---------- четыре режима скорости обучения ----------
def L(w):
    return (w - 3.0) ** 2


trajs = {
    "$\\eta = 0.25$": [0, 1.5, 2.25, 2.625],
    "$\\eta = 0.5$": [0, 3],
    "$\\eta = 1$": [0, 6, 0, 6],
    "$\\eta = 1.25$": [0, 7.5, -3.75, 13.125],
}

fig, axes = plt.subplots(2, 2, figsize=(9, 6))
for ax, (title, traj) in zip(axes.flat, trajs.items()):
    traj = np.array(traj, dtype=float)
    lo = min(-0.8, traj.min() - 1.2)
    hi = max(6.8, traj.max() + 1.2)
    w = np.linspace(lo, hi, 300)
    ax.plot(w, L(w), color=BLACK, lw=2.0, zorder=2)
    ax.scatter([3], [0], s=80, color=CRIMSON, zorder=4)
    for a, b in zip(traj[:-1], traj[1:]):
        ax.annotate(
            "",
            xy=(b, L(b)),
            xytext=(a, L(a)),
            arrowprops=dict(
                arrowstyle="->", color=BLUE, lw=1.6,
                connectionstyle="arc3,rad=0.18", shrinkA=6, shrinkB=6,
            ),
            zorder=3,
        )
    ax.scatter(traj, L(traj), s=48, color=BLUE, zorder=5)
    ymax = L(w).max()
    ax.annotate("$w_0$", (traj[0], L(traj[0])),
                xytext=(traj[0] + (hi - lo) * 0.03, L(traj[0])), fontsize=11)
    ax.set_title(title, fontsize=13)
    ax.set_ylim(-0.08 * ymax, ymax * 1.08)
    ax.set_xticks([])
    ax.set_yticks([])
save(fig, "img05_lr_regimes.png")
