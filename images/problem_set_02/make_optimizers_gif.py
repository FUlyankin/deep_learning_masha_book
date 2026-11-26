"""Сборка img02_optimizers.gif: траектории оптимизаторов из задачи 2 листочка 2.

Запуск: python3.11 make_optimizers_gif.py
Функция потерь — вытянутая квадратичная чаша 0.06 x^2 + 0.9 y^2, стартуем из одной точки.
Вдоль долины методы растягиваются, и траектории не сливаются.
"""
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

rng = np.random.default_rng(7)


# вытянутая чаша: пологая вдоль x, крутая поперёк y
AX, AY = 0.06, 0.9


def f(x, y):
    return AX * x**2 + AY * y**2


def grad(p, noisy=False):
    g = np.array([2 * AX * p[0], 2 * AY * p[1]])
    if noisy:
        g = g + rng.normal(0, 0.14, size=2)
    return g


START = np.array([-9.3, 1.25])
STEPS = 120


def run_sgd(lr=0.33):
    p, path = START.copy(), [START.copy()]
    for _ in range(STEPS):
        p = p - lr * grad(p, noisy=True)
        path.append(p.copy())
    return np.array(path)


def run_momentum(lr=0.05, alpha=0.9, nesterov=False):
    p, h, path = START.copy(), np.zeros(2), [START.copy()]
    for _ in range(STEPS):
        g = grad(p - alpha * h) if nesterov else grad(p)
        h = alpha * h + lr * g
        p = p - h
        path.append(p.copy())
    return np.array(path)


def run_rmsprop(lr=0.2, alpha=0.95, eps=1e-8):
    p, v, path = START.copy(), np.zeros(2), [START.copy()]
    for _ in range(STEPS):
        g = grad(p)
        v = alpha * v + (1 - alpha) * g**2
        p = p - lr / np.sqrt(v + eps) * g
        path.append(p.copy())
    return np.array(path)


def run_adam(lr=0.33, b1=0.9, b2=0.999, eps=1e-8, belief=False):
    p, h, v, path = START.copy(), np.zeros(2), np.zeros(2), [START.copy()]
    for t in range(1, STEPS + 1):
        g = grad(p)
        h = b1 * h + (1 - b1) * g
        v = b2 * v + (1 - b2) * ((g - h) ** 2 if belief else g**2)
        h_hat = h / (1 - b1**t)
        v_hat = v / (1 - b2**t)
        p = p - lr / (np.sqrt(v_hat) + eps) * h_hat
        path.append(p.copy())
    return np.array(path)


runs = [
    ("SGD", run_sgd(), "#888888"),
    ("Momentum", run_momentum(), "#D55E00"),
    ("Nesterov", run_momentum(nesterov=True), "#56B4E9"),
    ("RMSprop", run_rmsprop(), "#008000"),
    ("Adam", run_adam(), "#9400D3"),
    ("AdaBelief", run_adam(lr=0.33, belief=True), "#B8860B"),
]

for name, path, _ in runs:
    print(f"{name:10s} final = {np.round(path[-1], 3)}  loss = {f(*path[-1]):.4f}")

xs = np.linspace(-10.5, 2.5, 400)
ys = np.linspace(-2.0, 2.0, 250)
X, Y = np.meshgrid(xs, ys)
Z = f(X, Y)

fig, ax = plt.subplots(figsize=(9, 3.6), dpi=110)
ax.contour(X, Y, Z, levels=np.logspace(-1, 1.5, 12),
           colors="grey", linewidths=0.5, alpha=0.6)
ax.plot(0, 0, marker="*", color="black", markersize=16, zorder=6)
ax.set_xlim(-10.5, 2.5)
ax.set_ylim(-2.0, 2.0)
ax.set_xticks([])
ax.set_yticks([])
for side in ax.spines.values():
    side.set_visible(False)

lines, dots = [], []
for name, path, color in runs:
    lw, al = (1.5, 0.75) if name == "SGD" else (2.2, 0.95)
    ls = "--" if name == "AdaBelief" else "-"
    (ln,) = ax.plot([], [], color=color, lw=lw, ls=ls, label=name, zorder=3, alpha=al)
    (dt,) = ax.plot([], [], marker="o", color=color, markersize=7, zorder=4)
    lines.append((ln, path))
    dots.append((dt, path))
ax.legend(loc="upper right", fontsize=9, ncol=2, framealpha=0.95)
fig.tight_layout()

# каждый 2-й шаг оптимизации -> немного кадров -> лёгкий gif;
# в конце держим последний кадр паузой, потом цикл начинается заново
frames = list(range(0, STEPS + 1, 2)) + [STEPS] * 10


def update(frame):
    k = min(frame + 1, STEPS + 1)
    artists = []
    for (ln, path), (dt, _) in zip(lines, dots):
        ln.set_data(path[:k, 0], path[:k, 1])
        dt.set_data([path[k - 1, 0]], [path[k - 1, 1]])
        artists.extend([ln, dt])
    return artists


anim = FuncAnimation(fig, update, frames=frames, interval=140, blit=True)
# fps=9 -> каждый кадр ~110 мс: точки ползут медленно, кадров немного
anim.save("img02_optimizers.gif", writer=PillowWriter(fps=9))
print("saved img02_optimizers.gif")
