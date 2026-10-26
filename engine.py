from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Callable

import numpy as np
import tensorflow as tf

ArrayFn = Callable[[np.ndarray], np.ndarray]
TfFn = Callable[[tf.Tensor], tf.Tensor]
ResidualFn = Callable[[tf.keras.Model, tf.Tensor, tf.Tensor, float], tf.Tensor]


@dataclass(frozen=True)
class PDEProblem:
    name: str
    x_min: float
    x_max: float
    t_min: float
    t_max: float
    alpha: float
    initial_fn: ArrayFn
    boundary_fn: Callable[[np.ndarray, np.ndarray], np.ndarray]
    analytical_fn: Callable[[np.ndarray, np.ndarray], np.ndarray]
    residual_fn: ResidualFn


@dataclass(frozen=True)
class PINNConfig:
    hidden_layers: int = 4
    neurons: int = 64
    learning_rate: float = 8e-4
    epochs: int = 450
    n_collocation: int = 512
    n_initial: int = 96
    n_boundary: int = 96
    batch_size: int = 4096
    pde_weight: float = 1.0
    ic_weight: float = 15.0
    bc_weight: float = 15.0
    seed: int = 7


@dataclass
class TrainingResult:
    model: tf.keras.Model
    losses: list[float]
    pde_losses: list[float]
    ic_losses: list[float]
    bc_losses: list[float]


@dataclass
class SolveResult:
    problem: PDEProblem
    config: PINNConfig
    x: np.ndarray
    t: np.ndarray
    u_fd: np.ndarray
    u_pinn: np.ndarray
    u_true: np.ndarray
    relative_l2_error: float
    stability_number: float
    stable: bool
    training: TrainingResult


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    tf.random.set_seed(seed)


def build_mlp(problem: PDEProblem, hidden_layers: int = 3, neurons: int = 32) -> tf.keras.Model:
    x_scale = problem.x_max - problem.x_min
    t_scale = problem.t_max - problem.t_min

    def normalize_inputs(inputs: tf.Tensor) -> tf.Tensor:
        x = inputs[:, 0:1]
        t = inputs[:, 1:2]
        x_n = 2.0 * (x - problem.x_min) / x_scale - 1.0
        t_n = 2.0 * (t - problem.t_min) / t_scale - 1.0
        return tf.concat([x_n, t_n], axis=1)

    layers: list[tf.keras.layers.Layer] = [tf.keras.layers.Input(shape=(2,)), tf.keras.layers.Lambda(normalize_inputs)]
    for _ in range(hidden_layers):
        layers.append(tf.keras.layers.Dense(neurons, activation="tanh"))
    layers.append(tf.keras.layers.Dense(1))
    return tf.keras.Sequential(layers)


@tf.function
def heat_residual(model: tf.keras.Model, x: tf.Tensor, t: tf.Tensor, alpha: float) -> tf.Tensor:
    with tf.GradientTape(persistent=True) as tape2:
        tape2.watch([x, t])
        with tf.GradientTape(persistent=True) as tape1:
            tape1.watch([x, t])
            u = model(tf.concat([x, t], axis=1))
        u_x = tape1.gradient(u, x)
        u_t = tape1.gradient(u, t)
    u_xx = tape2.gradient(u_x, x)
    return u_t - alpha * u_xx


def heat_equation_problem(alpha: float = 0.1, x_min: float = 0.0, x_max: float = 1.0, t_min: float = 0.0, t_max: float = 1.0) -> PDEProblem:
    def initial_fn(x: np.ndarray) -> np.ndarray:
        return np.sin(np.pi * x)

    def boundary_fn(x: np.ndarray, t: np.ndarray) -> np.ndarray:
        return np.zeros_like(x)

    def analytical_fn(x: np.ndarray, t: np.ndarray) -> np.ndarray:
        return np.sin(np.pi * x) * np.exp(-(np.pi**2) * alpha * t)

    def residual_fn(model: tf.keras.Model, x: tf.Tensor, t: tf.Tensor, alpha_value: float) -> tf.Tensor:
        return heat_residual(model, x, t, alpha_value)

    return PDEProblem(
        name="1D heat equation",
        x_min=x_min,
        x_max=x_max,
        t_min=t_min,
        t_max=t_max,
        alpha=alpha,
        initial_fn=initial_fn,
        boundary_fn=boundary_fn,
        analytical_fn=analytical_fn,
        residual_fn=residual_fn,
    )


def stability_number(alpha: float, dx: float, dt: float) -> float:
    return alpha * dt / (dx**2)


def finite_difference_solution(problem: PDEProblem, dx: float, dt: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, bool]:
    x = np.arange(problem.x_min, problem.x_max + dx * 0.5, dx)
    t = np.arange(problem.t_min, problem.t_max + dt * 0.5, dt)
    nx, nt = len(x), len(t)
    r = stability_number(problem.alpha, dx, dt)
    stable = r <= 0.5
    if not stable:
        raise ValueError(f"Finite-difference stability violated: alpha*dt/dx^2 = {r:.3f} > 0.5")

    u = np.zeros((nx, nt), dtype=np.float64)
    u[:, 0] = problem.initial_fn(x)
    u[0, :] = problem.boundary_fn(np.zeros_like(t), t)
    u[-1, :] = problem.boundary_fn(np.full_like(t, x[-1]), t)
    coeff = problem.alpha * dt / (dx**2)
    for n in range(nt - 1):
        u[1:-1, n + 1] = u[1:-1, n] + coeff * (u[2:, n] - 2.0 * u[1:-1, n] + u[:-2, n])
        u[0, n + 1] = 0.0
        u[-1, n + 1] = 0.0
    return x, t, u, r, stable


def sample_training_points(problem: PDEProblem, config: PINNConfig) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    x0, x1 = problem.x_min, problem.x_max
    t0, t1 = problem.t_min, problem.t_max

    x_colloc = tf.random.uniform((config.n_collocation, 1), x0, x1, dtype=tf.float32)
    t_colloc = tf.random.uniform((config.n_collocation, 1), t0, t1, dtype=tf.float32)

    x_init = tf.random.uniform((config.n_initial, 1), x0, x1, dtype=tf.float32)
    t_init = tf.zeros((config.n_initial, 1), dtype=tf.float32) + t0

    t_left = tf.random.uniform((config.n_boundary, 1), t0, t1, dtype=tf.float32)
    t_right = tf.random.uniform((config.n_boundary, 1), t0, t1, dtype=tf.float32)
    x_left = tf.zeros((config.n_boundary, 1), dtype=tf.float32) + x0
    x_right = tf.zeros((config.n_boundary, 1), dtype=tf.float32) + x1

    return x_colloc, t_colloc, x_init, t_init, x_left, t_left, x_right, t_right


def compute_losses(
    problem: PDEProblem,
    model: tf.keras.Model,
    config: PINNConfig,
    x_colloc: tf.Tensor,
    t_colloc: tf.Tensor,
    x_init: tf.Tensor,
    t_init: tf.Tensor,
    x_left: tf.Tensor,
    t_left: tf.Tensor,
    x_right: tf.Tensor,
    t_right: tf.Tensor,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    residual = problem.residual_fn(model, x_colloc, t_colloc, problem.alpha)
    pde_loss = tf.reduce_mean(tf.square(residual))

    init_pred = model(tf.concat([x_init, t_init], axis=1))
    init_true = tf.convert_to_tensor(problem.initial_fn(x_init.numpy()), dtype=tf.float32)
    ic_loss = tf.reduce_mean(tf.square(init_pred - init_true))

    left_pred = model(tf.concat([x_left, t_left], axis=1))
    right_pred = model(tf.concat([x_right, t_right], axis=1))
    left_true = tf.convert_to_tensor(problem.boundary_fn(x_left.numpy(), t_left.numpy()), dtype=tf.float32)
    right_true = tf.convert_to_tensor(problem.boundary_fn(x_right.numpy(), t_right.numpy()), dtype=tf.float32)
    bc_loss = 0.5 * (tf.reduce_mean(tf.square(left_pred - left_true)) + tf.reduce_mean(tf.square(right_pred - right_true)))

    total = config.pde_weight * pde_loss + config.ic_weight * ic_loss + config.bc_weight * bc_loss
    return total, pde_loss, ic_loss, bc_loss


@lru_cache(maxsize=32)
def train_problem_cached(key: tuple) -> TrainingResult:
    raise RuntimeError("train_problem_cached should not be called directly")


def train_problem(problem: PDEProblem, config: PINNConfig) -> TrainingResult:
    set_seed(config.seed)
    model = build_mlp(problem, hidden_layers=config.hidden_layers, neurons=config.neurons)
    optimizer = tf.keras.optimizers.Adam(learning_rate=config.learning_rate)

    losses: list[float] = []
    pde_losses: list[float] = []
    ic_losses: list[float] = []
    bc_losses: list[float] = []

    for _ in range(config.epochs):
        with tf.GradientTape() as tape:
            points = sample_training_points(problem, config)
            total, pde_loss, ic_loss, bc_loss = compute_losses(problem, model, config, *points)
        grads = tape.gradient(total, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        losses.append(float(total.numpy()))
        pde_losses.append(float(pde_loss.numpy()))
        ic_losses.append(float(ic_loss.numpy()))
        bc_losses.append(float(bc_loss.numpy()))

    return TrainingResult(model=model, losses=losses, pde_losses=pde_losses, ic_losses=ic_losses, bc_losses=bc_losses)


def predict_batched(model: tf.keras.Model, x: np.ndarray, t: np.ndarray, batch_size: int = 4096) -> np.ndarray:
    points = np.column_stack([x.reshape(-1), t.reshape(-1)]).astype(np.float32)
    outputs: list[np.ndarray] = []
    for start in range(0, len(points), batch_size):
        chunk = tf.convert_to_tensor(points[start : start + batch_size])
        pred = model(chunk, training=False).numpy().reshape(-1)
        outputs.append(pred)
    return np.concatenate(outputs, axis=0)


def relative_l2_error(u_pred: np.ndarray, u_true: np.ndarray) -> float:
    denom = np.linalg.norm(u_true.ravel())
    if denom == 0:
        return float(np.linalg.norm(u_pred.ravel()))
    return float(np.linalg.norm((u_pred - u_true).ravel()) / denom)


def solve_heat_problem(problem: PDEProblem, config: PINNConfig, dx: float, dt: float) -> SolveResult:
    x_fd, t_fd, u_fd, r, stable = finite_difference_solution(problem, dx, dt)
    x_grid, t_grid = np.meshgrid(x_fd, t_fd, indexing="ij")

    training = train_problem(problem, config)
    u_pinn_flat = predict_batched(training.model, x_grid.reshape(-1, 1), t_grid.reshape(-1, 1), batch_size=config.batch_size)
    u_pinn = u_pinn_flat.reshape(x_grid.shape)
    u_true = problem.analytical_fn(x_grid, t_grid)
    error = relative_l2_error(u_pinn, u_true)

    return SolveResult(
        problem=problem,
        config=config,
        x=x_fd,
        t=t_fd,
        u_fd=u_fd,
        u_pinn=u_pinn,
        u_true=u_true,
        relative_l2_error=error,
        stability_number=r,
        stable=stable,
        training=training,
    )
