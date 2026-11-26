"""Схема Momentum vs Nesterov для задачи 2 листочка 2.

Запуск: python3.11 make_nesterov_plot.py
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.style.use("seaborn-v0_8-white")

GREEN = "#008000"
RED = "#D55E00"
BLUE = "#0072B2"

fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))

mom = np.array([2.8, 0.5])    # накопленная инерция
grad_here = np.array([0.7, 1.5])   # шаг по градиенту в текущей точке
grad_ahead = np.array([0.4, 1.7])  # шаг по градиенту в точке после инерции


def vec(ax, start, delta, color, lw=2.4, ls="-", zorder=3):
    ax.annotate(
        "",
        xy=start + delta,
        xytext=start,
        arrowprops=dict(
            arrowstyle="-|>,head_width=0.22,head_length=0.45",
            color=color, lw=lw, linestyle=ls, shrinkA=0, shrinkB=0,
        ),
        zorder=zorder,
    )


for ax in axes:
    ax.set_xlim(-0.7, 4.4)
    ax.set_ylim(-0.6, 3.1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.scatter([0], [0], s=70, color="black", zorder=5)
    ax.annotate("$w_{t-1}$", (0, 0), xytext=(-0.25, -0.42), fontsize=13)

# --- Momentum: градиент считаем там, где стоим ---
ax = axes[0]
ax.set_title("Momentum", fontsize=14)
vec(ax, np.zeros(2), mom, GREEN, ls="--", lw=2.0)
vec(ax, np.zeros(2), grad_here, RED)
vec(ax, np.zeros(2), mom + grad_here, BLUE, lw=3.2, zorder=2)
ax.annotate("инерция $\\alpha h_{t-1}$", mom * 0.55 + [0.08, -0.33],
            fontsize=12, color=GREEN)
ax.annotate("градиент в $w_{t-1}$", grad_here + [-1.35, 0.25],
            fontsize=12, color=RED)
ax.annotate("итоговый шаг $h_t$", (mom + grad_here) * 0.72 + [0.1, -0.05],
            fontsize=12, color=BLUE)

# --- Nesterov: сначала инерция, градиент считаем в новой точке ---
ax = axes[1]
ax.set_title("Momentum с коррекцией Нестерова", fontsize=14)
vec(ax, np.zeros(2), mom, GREEN, ls="--", lw=2.0)
ax.scatter([mom[0]], [mom[1]], s=45, color="black", zorder=5)
vec(ax, mom, grad_ahead, RED)
vec(ax, np.zeros(2), mom + grad_ahead, BLUE, lw=3.2, zorder=2)
ax.annotate("инерция $\\alpha h_{t-1}$", mom * 0.55 + [0.08, -0.33],
            fontsize=12, color=GREEN)
ax.annotate("градиент в точке\n$w_{t-1} - \\alpha h_{t-1}$",
            mom + grad_ahead * 0.35 + [0.12, 0.05], fontsize=12, color=RED)
ax.annotate("итоговый шаг $h_t$", (mom + grad_ahead) * 0.6 + [-0.25, 0.28],
            fontsize=12, color=BLUE)

fig.tight_layout()
fig.savefig("img02_nesterov.png", dpi=150, bbox_inches="tight")
print("saved img02_nesterov.png")
