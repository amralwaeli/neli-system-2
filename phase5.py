from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv
import time

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

OUT = Path(__file__).resolve().parent / "outputs" / "phase5"
OUT.mkdir(parents=True, exist_ok=True)


@dataclass
class Result:
    name: str
    category: str
    rel_l2: float
    max_abs: float
    train_seconds: float
    problem: Problem
    model: tf.keras.Model
    history: dict[str, list[float]]
    grid: dict[str, np.ndarray]


@dataclass
class Problem:
    name: str
    category: str
    input_dim: int
    bounds: list[tuple[float, float]]
    sample_batch: callable
    loss_fn: callable
    exact_fn: callable
    eval_grid: callable
    plot_fn: callable


def set_seed(seed: int = 7):
    np.random.seed(seed)
    tf.random.set_seed(seed)


def build_model(input_dim: int, bounds: list[tuple[float, float]], hidden_layers: int = 3, neurons: int = 32) -> tf.keras.Model:
    mins = np.array([b[0] for b in bounds], dtype=np.float32)
    spans = np.array([b[1] - b[0] for b in bounds], dtype=np.float32)

    def normalize(inputs: tf.Tensor) -> tf.Tensor:
        return 2.0 * (inputs - mins) / spans - 1.0

    x_in = tf.keras.Input(shape=(input_dim,))
    x = tf.keras.layers.Lambda(normalize)(x_in)
    for _ in range(hidden_layers):
        x = tf.keras.layers.Dense(neurons, activation="tanh")(x)
    out = tf.keras.layers.Dense(1)(x)
    return tf.keras.Model(x_in, out)


def batched_predict(model: tf.keras.Model, pts: np.ndarray, batch_size: int = 4096) -> np.ndarray:
    outs = []
    for i in range(0, len(pts), batch_size):
        chunk = tf.convert_to_tensor(pts[i : i + batch_size], dtype=tf.float32)
        outs.append(model(chunk, training=False).numpy().reshape(-1, 1))
    return np.vstack(outs)


def relative_l2(pred: np.ndarray, true: np.ndarray) -> float:
    pred = np.asarray(pred).reshape(-1)
    true = np.asarray(true).reshape(-1)
    denom = np.linalg.norm(true)
    if denom == 0:
        return float(np.linalg.norm(pred))
    return float(np.linalg.norm(pred - true) / denom)


# -------- Problem definitions --------

def make_exponential_decay_problem(k: float = 2.0) -> Problem:
    bounds = [(0.0, 1.0)]

    def sample(n_colloc: int, n_ic: int):
        t_colloc = np.random.rand(n_colloc, 1).astype(np.float32)
        t0 = np.zeros((n_ic, 1), dtype=np.float32)
        return t_colloc, t0

    def loss_fn(model, batch):
        t_colloc, t0 = batch
        t_colloc_t = tf.convert_to_tensor(t_colloc)
        with tf.GradientTape() as tape:
            tape.watch(t_colloc_t)
            raw = model(t_colloc_t)
            y = 1.0 + t_colloc_t * raw
        y_t = tape.gradient(y, t_colloc_t)
        pde = tf.reduce_mean(tf.square(y_t + k * y))
        y0 = 1.0 + tf.zeros_like(t0) * model(tf.convert_to_tensor(t0))
        ic = tf.reduce_mean(tf.square(y0 - 1.0))
        return pde + 20.0 * ic, pde, ic, tf.constant(0.0)

    def exact_fn(pts):
        t = pts[:, 0]
        return np.exp(-k * t)

    def eval_grid(model):
        t = np.linspace(0.0, 1.0, 200).reshape(-1, 1).astype(np.float32)
        y_true = exact_fn(t)
        y_pred = (1.0 + t * model(tf.convert_to_tensor(t)).numpy()).reshape(-1)
        return {"t": t.ravel(), "true": y_true, "pred": y_pred}

    def plot_fn(result: Result):
        t = result.grid["t"]
        true = result.grid["true"].ravel()
        pred = result.grid["pred"].ravel()
        err = np.abs(pred - true)
        fig1, ax1 = plt.subplots(figsize=(6.5, 4))
        ax1.plot(t, true, label="Reference", linewidth=2)
        ax1.plot(t, pred, "--", label="PINN", linewidth=2)
        ax1.set_title(result.name)
        ax1.set_xlabel("t")
        ax1.set_ylabel("y(t)")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        fig2, ax2 = plt.subplots(figsize=(6.5, 4))
        ax2.plot(t, err, color="crimson", linewidth=2)
        ax2.set_title("Absolute error")
        ax2.set_xlabel("t")
        ax2.set_ylabel("|error|")
        ax2.grid(True, alpha=0.3)
        fig3, ax3 = plt.subplots(figsize=(6.5, 4))
        ax3.semilogy(result.history["total"], label="Total")
        ax3.semilogy(result.history["pde"], label="PDE")
        ax3.semilogy(result.history["ic"], label="IC")
        ax3.set_title("Loss curves")
        ax3.set_xlabel("Epoch")
        ax3.set_ylabel("Loss")
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        return fig1, fig2, fig3

    return Problem("Exponential decay", "ODE", 1, bounds, sample, loss_fn, exact_fn, eval_grid, plot_fn)


def make_harmonic_oscillator_problem(omega: float = 2.0) -> Problem:
    bounds = [(0.0, 1.0)]

    def sample(n_colloc: int, n_ic: int):
        t_colloc = np.random.rand(n_colloc, 1).astype(np.float32)
        t0 = np.zeros((n_ic, 1), dtype=np.float32)
        return t_colloc, t0

    def loss_fn(model, batch):
        t_colloc, t0 = batch
        t_colloc_t = tf.convert_to_tensor(t_colloc)
        with tf.GradientTape(persistent=True) as tape2:
            tape2.watch(t_colloc_t)
            with tf.GradientTape() as tape1:
                tape1.watch(t_colloc_t)
                raw = model(t_colloc_t)
                y = t_colloc_t + tf.square(t_colloc_t) * raw
            y_t = tape1.gradient(y, t_colloc_t)
        y_tt = tape2.gradient(y_t, t_colloc_t)
        del tape2
        pde = tf.reduce_mean(tf.square(y_tt + (omega**2) * y))
        y0 = tf.zeros_like(t0)
        with tf.GradientTape() as tape0:
            t0_t = tf.convert_to_tensor(t0)
            tape0.watch(t0_t)
            y0p = t0_t + tf.square(t0_t) * model(t0_t)
        y0_t = tape0.gradient(y0p, t0_t)
        ic = tf.reduce_mean(tf.square(y0 - 0.0)) + tf.reduce_mean(tf.square(y0_t - 1.0))
        return pde + 10.0 * ic, pde, ic, tf.constant(0.0)

    def exact_fn(pts):
        t = pts[:, 0]
        return np.sin(omega * t) / omega

    def eval_grid(model):
        t = np.linspace(0.0, 1.0, 200).reshape(-1, 1).astype(np.float32)
        y_true = exact_fn(t)
        y_pred = (t + (t**2) * model(tf.convert_to_tensor(t)).numpy()).reshape(-1)
        return {"t": t.ravel(), "true": y_true, "pred": y_pred}

    def plot_fn(result: Result):
        t = result.grid["t"]
        true = result.grid["true"].ravel()
        pred = result.grid["pred"].ravel()
        err = np.abs(pred - true)
        fig1, ax1 = plt.subplots(figsize=(6.5, 4))
        ax1.plot(t, true, label="Reference", linewidth=2)
        ax1.plot(t, pred, "--", label="PINN", linewidth=2)
        ax1.set_title(result.name)
        ax1.set_xlabel("t")
        ax1.set_ylabel("y(t)")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        fig2, ax2 = plt.subplots(figsize=(6.5, 4))
        ax2.plot(t, err, color="crimson", linewidth=2)
        ax2.set_title("Absolute error")
        ax2.set_xlabel("t")
        ax2.set_ylabel("|error|")
        ax2.grid(True, alpha=0.3)
        fig3, ax3 = plt.subplots(figsize=(6.5, 4))
        ax3.semilogy(result.history["total"], label="Total")
        ax3.semilogy(result.history["pde"], label="PDE")
        ax3.semilogy(result.history["ic"], label="IC")
        ax3.set_title("Loss curves")
        ax3.set_xlabel("Epoch")
        ax3.set_ylabel("Loss")
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        return fig1, fig2, fig3

    return Problem("Harmonic oscillator", "ODE", 1, bounds, sample, loss_fn, exact_fn, eval_grid, plot_fn)


def make_logistic_growth_problem(r: float = 2.0, k: float = 1.0, y0: float = 0.1) -> Problem:
    bounds = [(0.0, 1.0)]

    def sample(n_colloc: int, n_ic: int):
        t_colloc = np.random.rand(n_colloc, 1).astype(np.float32)
        t0 = np.zeros((n_ic, 1), dtype=np.float32)
        return t_colloc, t0

    def loss_fn(model, batch):
        t_colloc, t0 = batch
        t_colloc_t = tf.convert_to_tensor(t_colloc)
        with tf.GradientTape() as tape:
            tape.watch(t_colloc_t)
            raw = model(t_colloc_t)
            y = y0 + t_colloc_t * raw
        y_t = tape.gradient(y, t_colloc_t)
        pde = tf.reduce_mean(tf.square(y_t - r * y * (1.0 - y / k)))
        y_init = y0 + tf.zeros_like(t0) * model(tf.convert_to_tensor(t0))
        ic = tf.reduce_mean(tf.square(y_init - y0))
        return pde + 20.0 * ic, pde, ic, tf.constant(0.0)

    def exact_fn(pts):
        t = pts[:, 0]
        return k * y0 * np.exp(r * t) / (k + y0 * (np.exp(r * t) - 1.0))

    def eval_grid(model):
        t = np.linspace(0.0, 1.0, 200).reshape(-1, 1).astype(np.float32)
        y_true = exact_fn(t)
        y_pred = (y0 + t * model(tf.convert_to_tensor(t)).numpy()).reshape(-1)
        return {"t": t.ravel(), "true": y_true, "pred": y_pred}

    def plot_fn(result: Result):
        t = result.grid["t"]
        true = result.grid["true"].ravel()
        pred = result.grid["pred"].ravel()
        err = np.abs(pred - true)
        fig1, ax1 = plt.subplots(figsize=(6.5, 4))
        ax1.plot(t, true, label="Reference", linewidth=2)
        ax1.plot(t, pred, "--", label="PINN", linewidth=2)
        ax1.set_title(result.name)
        ax1.set_xlabel("t")
        ax1.set_ylabel("y(t)")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        fig2, ax2 = plt.subplots(figsize=(6.5, 4))
        ax2.plot(t, err, color="crimson", linewidth=2)
        ax2.set_title("Absolute error")
        ax2.set_xlabel("t")
        ax2.set_ylabel("|error|")
        ax2.grid(True, alpha=0.3)
        fig3, ax3 = plt.subplots(figsize=(6.5, 4))
        ax3.semilogy(result.history["total"], label="Total")
        ax3.semilogy(result.history["pde"], label="PDE")
        ax3.semilogy(result.history["ic"], label="IC")
        ax3.set_title("Loss curves")
        ax3.set_xlabel("Epoch")
        ax3.set_ylabel("Loss")
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        return fig1, fig2, fig3

    return Problem("Logistic growth", "ODE", 1, bounds, sample, loss_fn, exact_fn, eval_grid, plot_fn)


def make_wave_problem(c: float = 1.0) -> Problem:
    bounds = [(0.0, 1.0), (0.0, 1.0)]

    def sample(n_colloc: int, n_ic: int, n_bc: int):
        x_c = np.random.rand(n_colloc, 1).astype(np.float32)
        t_c = np.random.rand(n_colloc, 1).astype(np.float32)
        x_i = np.random.rand(n_ic, 1).astype(np.float32)
        t_i = np.zeros((n_ic, 1), dtype=np.float32)
        x_l = np.zeros((n_bc, 1), dtype=np.float32)
        x_r = np.ones((n_bc, 1), dtype=np.float32)
        t_b = np.random.rand(n_bc, 1).astype(np.float32)
        return x_c, t_c, x_i, t_i, x_l, t_b, x_r, t_b

    def loss_fn(model, batch):
        x_c, t_c, x_i, t_i, x_l, t_b, x_r, _ = batch
        xc = tf.convert_to_tensor(np.hstack([x_c, t_c]))
        xi = tf.convert_to_tensor(np.hstack([x_i, t_i]))
        xl = tf.convert_to_tensor(np.hstack([x_l, t_b]))
        xr = tf.convert_to_tensor(np.hstack([x_r, t_b]))
        with tf.GradientTape(persistent=True) as tape2:
            tape2.watch(xc)
            with tf.GradientTape(persistent=True) as tape1:
                tape1.watch(xc)
                raw = model(xc)
                x = xc[:, 0:1]
                t = xc[:, 1:2]
                u = tf.sin(np.pi * x) + tf.square(t) * x * (1.0 - x) * raw
            du = tape1.gradient(u, xc)
            u_x = du[:, 0:1]
            u_t = du[:, 1:2]
        u_xx = tape2.gradient(u_x, xc)[:, 0:1]
        u_tt = tape2.gradient(u_t, xc)[:, 1:2]
        del tape1, tape2
        pde = tf.reduce_mean(tf.square(u_tt - (c**2) * u_xx))
        ic = tf.constant(0.0)
        b = tf.constant(0.0)
        return pde, pde, ic, b

    def exact_fn(pts):
        x = pts[:, 0]
        t = pts[:, 1]
        return np.sin(np.pi * x) * np.cos(np.pi * c * t)

    def eval_grid(model):
        x = np.linspace(0.0, 1.0, 80)
        t = np.linspace(0.0, 1.0, 80)
        X, T = np.meshgrid(x, t, indexing="ij")
        pts = np.column_stack([X.ravel(), T.ravel()]).astype(np.float32)
        pred = (np.sin(np.pi * pts[:, 0:1]) + (pts[:, 1:2] ** 2) * pts[:, 0:1] * (1.0 - pts[:, 0:1]) * model(tf.convert_to_tensor(pts)).numpy()).reshape(X.shape)
        true = exact_fn(pts).reshape(X.shape)
        return {"x": x, "t": t, "pred": pred, "true": true}

    def plot_fn(result: Result):
        X, T = np.meshgrid(result.grid["x"], result.grid["t"], indexing="ij")
        true = result.grid["true"]
        pred = result.grid["pred"]
        err = np.abs(pred - true)
        mid = len(result.grid["t"]) // 2
        fig1, ax1 = plt.subplots(figsize=(6.5, 4))
        ax1.plot(result.grid["x"], true[:, mid], label="Reference", linewidth=2)
        ax1.plot(result.grid["x"], pred[:, mid], "--", label="PINN", linewidth=2)
        ax1.set_title(f"Wave equation slice at t={result.grid['t'][mid]:.2f}")
        ax1.set_xlabel("x")
        ax1.set_ylabel("u(x,t)")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        fig2, ax2 = plt.subplots(figsize=(6.5, 4))
        cax = ax2.contourf(X, T, err, levels=24, cmap="magma")
        fig2.colorbar(cax, ax=ax2)
        ax2.set_title("Absolute error")
        ax2.set_xlabel("x")
        ax2.set_ylabel("t")
        fig3, ax3 = plt.subplots(figsize=(6.5, 4))
        ax3.semilogy(result.history["total"], label="Total")
        ax3.semilogy(result.history["pde"], label="PDE")
        ax3.semilogy(result.history["ic"], label="IC")
        ax3.semilogy(result.history["bc"], label="Boundary")
        ax3.set_title("Loss curves")
        ax3.set_xlabel("Epoch")
        ax3.set_ylabel("Loss")
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        return fig1, fig2, fig3

    return Problem("1D wave equation", "PDE", 2, bounds, sample, loss_fn, exact_fn, eval_grid, plot_fn)


def make_burgers_problem(nu: float = 0.01 / np.pi) -> Problem:
    bounds = [(0.0, 1.0), (0.0, 1.0)]

    def exact_u(x, t):
        return np.sin(np.pi * x) * np.exp(-t)

    def sample(n_colloc: int, n_ic: int, n_bc: int):
        x_c = np.random.rand(n_colloc, 1).astype(np.float32)
        t_c = np.random.rand(n_colloc, 1).astype(np.float32)
        x_i = np.random.rand(n_ic, 1).astype(np.float32)
        t_i = np.zeros((n_ic, 1), dtype=np.float32)
        x_l = np.zeros((n_bc, 1), dtype=np.float32)
        x_r = np.ones((n_bc, 1), dtype=np.float32)
        t_b = np.random.rand(n_bc, 1).astype(np.float32)
        return x_c, t_c, x_i, t_i, x_l, t_b, x_r, t_b

    def loss_fn(model, batch):
        x_c, t_c, x_i, t_i, x_l, t_b, x_r, _ = batch
        xc = tf.convert_to_tensor(np.hstack([x_c, t_c]))
        xi = tf.convert_to_tensor(np.hstack([x_i, t_i]))
        xl = tf.convert_to_tensor(np.hstack([x_l, t_b]))
        xr = tf.convert_to_tensor(np.hstack([x_r, t_b]))
        with tf.GradientTape(persistent=True) as tape2:
            tape2.watch(xc)
            with tf.GradientTape(persistent=True) as tape1:
                tape1.watch(xc)
                raw = model(xc)
                x = xc[:, 0:1]
                t = xc[:, 1:2]
                u = tf.sin(np.pi * x) + t * x * (1.0 - x) * raw
            du = tape1.gradient(u, xc)
            u_x = du[:, 0:1]
            u_t = du[:, 1:2]
        u_xx = tape2.gradient(u_x, xc)[:, 0:1]
        del tape1, tape2
        forcing = tf.reshape(tf.convert_to_tensor(
            -np.sin(np.pi * x_c) * np.exp(-t_c)
            + (np.sin(np.pi * x_c) * np.exp(-t_c)) * (np.pi * np.cos(np.pi * x_c) * np.exp(-t_c))
            - nu * (-np.pi**2 * np.sin(np.pi * x_c) * np.exp(-t_c)),
            dtype=tf.float32,
        ), (-1, 1))
        pde = tf.reduce_mean(tf.square(u_t + u * u_x - nu * u_xx - forcing))
        ic = tf.constant(0.0)
        b = tf.constant(0.0)
        return pde, pde, ic, b

    def eval_grid(model):
        x = np.linspace(0.0, 1.0, 80)
        t = np.linspace(0.0, 1.0, 80)
        X, T = np.meshgrid(x, t, indexing="ij")
        pts = np.column_stack([X.ravel(), T.ravel()]).astype(np.float32)
        pred = (np.sin(np.pi * pts[:, 0:1]) + pts[:, 1:2] * pts[:, 0:1] * (1.0 - pts[:, 0:1]) * model(tf.convert_to_tensor(pts)).numpy()).reshape(X.shape)
        true = exact_u(X, T)
        return {"x": x, "t": t, "pred": pred, "true": true}

    def plot_fn(result: Result):
        X, T = np.meshgrid(result.grid["x"], result.grid["t"], indexing="ij")
        true = result.grid["true"]
        pred = result.grid["pred"]
        err = np.abs(pred - true)
        mid = len(result.grid["t"]) // 2
        fig1, ax1 = plt.subplots(figsize=(6.5, 4))
        ax1.plot(result.grid["x"], true[:, mid], label="Reference", linewidth=2)
        ax1.plot(result.grid["x"], pred[:, mid], "--", label="PINN", linewidth=2)
        ax1.set_title(f"Burgers slice at t={result.grid['t'][mid]:.2f}")
        ax1.set_xlabel("x")
        ax1.set_ylabel("u(x,t)")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        fig2, ax2 = plt.subplots(figsize=(6.5, 4))
        cax = ax2.contourf(X, T, err, levels=24, cmap="magma")
        fig2.colorbar(cax, ax=ax2)
        ax2.set_title("Absolute error")
        ax2.set_xlabel("x")
        ax2.set_ylabel("t")
        fig3, ax3 = plt.subplots(figsize=(6.5, 4))
        ax3.semilogy(result.history["total"], label="Total")
        ax3.semilogy(result.history["pde"], label="PDE")
        ax3.semilogy(result.history["ic"], label="IC")
        ax3.semilogy(result.history["bc"], label="Boundary")
        ax3.set_title("Loss curves")
        ax3.set_xlabel("Epoch")
        ax3.set_ylabel("Loss")
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        return fig1, fig2, fig3

    def exact_fn(pts):
        return exact_u(pts[:, 0], pts[:, 1])

    return Problem("Viscous Burgers", "PDE", 2, bounds, sample, loss_fn, exact_fn, eval_grid, plot_fn)


def make_poisson_problem() -> Problem:
    bounds = [(0.0, 1.0), (0.0, 1.0)]

    def exact_fn(pts):
        x = pts[:, 0]
        y = pts[:, 1]
        return np.sin(np.pi * x) * np.sin(np.pi * y)

    def sample(n_colloc: int, n_bc: int):
        x_c = np.random.rand(n_colloc, 1).astype(np.float32)
        y_c = np.random.rand(n_colloc, 1).astype(np.float32)
        t = np.zeros((n_bc, 1), dtype=np.float32)
        side = np.random.rand(n_bc, 1).astype(np.float32)
        left = np.hstack([np.zeros_like(side), side])
        right = np.hstack([np.ones_like(side), side])
        bottom = np.hstack([side, np.zeros_like(side)])
        top = np.hstack([side, np.ones_like(side)])
        return x_c, y_c, left, right, bottom, top

    def loss_fn(model, batch):
        x_c, y_c, left, right, bottom, top = batch
        xc = tf.convert_to_tensor(np.hstack([x_c, y_c]))
        with tf.GradientTape(persistent=True) as tape2:
            tape2.watch(xc)
            with tf.GradientTape(persistent=True) as tape1:
                tape1.watch(xc)
                raw = model(xc)
                x = xc[:, 0:1]
                y = xc[:, 1:2]
                u = x * (1.0 - x) * y * (1.0 - y) * raw
            du = tape1.gradient(u, xc)
            u_x = du[:, 0:1]
            u_y = du[:, 1:2]
        u_xx = tape2.gradient(u_x, xc)[:, 0:1]
        u_yy = tape2.gradient(u_y, xc)[:, 1:2]
        del tape1, tape2
        forcing = 2.0 * (np.pi**2) * tf.reshape(tf.convert_to_tensor(exact_fn(np.hstack([x_c, y_c])), dtype=tf.float32), (-1, 1))
        pde = tf.reduce_mean(tf.square(u_xx + u_yy + forcing))
        b = tf.constant(0.0)
        return pde, pde, tf.constant(0.0), b

    def eval_grid(model):
        x = np.linspace(0.0, 1.0, 70)
        y = np.linspace(0.0, 1.0, 70)
        X, Y = np.meshgrid(x, y, indexing="ij")
        pts = np.column_stack([X.ravel(), Y.ravel()]).astype(np.float32)
        pred = (pts[:, 0:1] * (1.0 - pts[:, 0:1]) * pts[:, 1:2] * (1.0 - pts[:, 1:2]) * model(tf.convert_to_tensor(pts)).numpy()).reshape(X.shape)
        true = exact_fn(pts).reshape(X.shape)
        return {"x": x, "y": y, "pred": pred, "true": true}

    def plot_fn(result: Result):
        X, Y = np.meshgrid(result.grid["x"], result.grid["y"], indexing="ij")
        true = result.grid["true"]
        pred = result.grid["pred"]
        err = np.abs(pred - true)
        fig1, ax1 = plt.subplots(figsize=(6.5, 4))
        c1 = ax1.contourf(X, Y, true, levels=24, cmap="viridis")
        fig1.colorbar(c1, ax=ax1)
        ax1.set_title("Reference solution")
        ax1.set_xlabel("x")
        ax1.set_ylabel("y")
        fig2, ax2 = plt.subplots(figsize=(6.5, 4))
        c2 = ax2.contourf(X, Y, pred, levels=24, cmap="viridis")
        fig2.colorbar(c2, ax=ax2)
        ax2.set_title("PINN prediction")
        ax2.set_xlabel("x")
        ax2.set_ylabel("y")
        fig3, ax3 = plt.subplots(figsize=(6.5, 4))
        c3 = ax3.contourf(X, Y, err, levels=24, cmap="magma")
        fig3.colorbar(c3, ax=ax3)
        ax3.set_title("Absolute error")
        ax3.set_xlabel("x")
        ax3.set_ylabel("y")
        return fig1, fig2, fig3

    return Problem("2D Poisson", "PDE", 2, bounds, sample, loss_fn, exact_fn, eval_grid, plot_fn)


# -------- Training --------

def train(
    problem: Problem,
    epochs: int,
    hidden_layers: int = 4,
    neurons: int = 64,
    seed: int = 7,
    n_collocation: int | None = None,
    n_initial: int | None = None,
    n_boundary: int | None = None,
) -> Result:
    set_seed(seed)
    model = build_model(problem.input_dim, problem.bounds, hidden_layers=hidden_layers, neurons=neurons)
    opt = tf.keras.optimizers.Adam(learning_rate=2e-3)
    history = {"total": [], "pde": [], "ic": [], "bc": []}
    start = time.time()
    for _ in range(epochs):
        with tf.GradientTape() as tape:
            if problem.category == "ODE":
                batch = problem.sample_batch(n_collocation or 512, n_initial or 64)
            elif problem.name == "2D Poisson":
                batch = problem.sample_batch(n_collocation or 1024, n_boundary or 256)
            else:
                batch = problem.sample_batch(n_collocation or 768, n_initial or 96, n_boundary or 96)
            total, pde, ic, bc = problem.loss_fn(model, batch)
        grads = tape.gradient(total, model.trainable_variables)
        opt.apply_gradients(zip(grads, model.trainable_variables))
        history["total"].append(float(total.numpy()))
        history["pde"].append(float(pde.numpy()))
        history["ic"].append(float(ic.numpy()))
        history["bc"].append(float(bc.numpy()))
    grid = problem.eval_grid(model)
    pred = grid["pred"]
    true = grid["true"]
    rel = relative_l2(pred, true)
    max_abs = float(np.max(np.abs(pred - true)))
    return Result(problem.name, problem.category, rel, max_abs, time.time() - start, problem, model, history, grid)


# -------- Reporting --------

def save_1d_plots(result: Result):
    fig1, fig2, fig3 = result.problem.plot_fn(result)
    base = OUT / result.name.lower().replace(" ", "_")
    fig1.savefig(base.with_name(base.name + "_reference_vs_pinn.png"), dpi=160, bbox_inches="tight")
    fig2.savefig(base.with_name(base.name + "_error.png"), dpi=160, bbox_inches="tight")
    fig3.savefig(base.with_name(base.name + "_loss.png"), dpi=160, bbox_inches="tight")
    plt.close(fig1)
    plt.close(fig2)
    plt.close(fig3)


def save_2d_plots(result: Result):
    fig1, fig2, fig3 = result.problem.plot_fn(result)
    fig4, ax4 = plt.subplots(figsize=(6.5, 4))
    ax4.semilogy(result.history["total"], label="Total")
    ax4.semilogy(result.history["pde"], label="PDE")
    ax4.semilogy(result.history["bc"], label="Boundary")
    ax4.set_title("Loss curves")
    ax4.set_xlabel("Epoch")
    ax4.set_ylabel("Loss")
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    base = OUT / result.name.lower().replace(" ", "_")
    fig1.savefig(base.with_name(base.name + "_reference.png"), dpi=160, bbox_inches="tight")
    fig2.savefig(base.with_name(base.name + "_pinn.png"), dpi=160, bbox_inches="tight")
    fig3.savefig(base.with_name(base.name + "_error.png"), dpi=160, bbox_inches="tight")
    fig4.savefig(base.with_name(base.name + "_loss.png"), dpi=160, bbox_inches="tight")
    plt.close(fig1)
    plt.close(fig2)
    plt.close(fig3)
    plt.close(fig4)


def main():
    problems = [
        make_exponential_decay_problem(),
        make_harmonic_oscillator_problem(),
        make_logistic_growth_problem(),
        make_wave_problem(),
        make_burgers_problem(),
        make_poisson_problem(),
    ]
    configs = {
        "Exponential decay": 400,
        "Harmonic oscillator": 220,
        "Logistic growth": 220,
        "1D wave equation": 220,
        "Viscous Burgers": 220,
        "2D Poisson": 220,
    }
    all_rows = []
    for problem in problems:
        result = train(problem, epochs=configs[problem.name])
        if problem.category == "ODE":
            save_1d_plots(result)
        else:
            save_2d_plots(result)
        all_rows.append({
            "name": result.name,
            "category": result.category,
            "rel_l2": result.rel_l2,
            "max_abs": result.max_abs,
            "train_seconds": result.train_seconds,
        })
        print(f"{result.name}: rel_l2={result.rel_l2:.6f}, max_abs={result.max_abs:.6f}, train_seconds={result.train_seconds:.1f}")
    with (OUT / "phase5_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["name", "category", "rel_l2", "max_abs", "train_seconds"])
        writer.writeheader()
        writer.writerows(all_rows)

    lines = ["# Phase 5 Summary", ""]
    for row in all_rows:
        lines.append(f"- {row['name']}: rel L2={row['rel_l2']:.6f}, max abs={row['max_abs']:.6f}, train={row['train_seconds']:.1f}s")
    (OUT / "phase5_summary.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
