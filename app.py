from __future__ import annotations

from dataclasses import asdict
from functools import lru_cache

import gradio as gr
import matplotlib.pyplot as plt
import numpy as np

from engine import PINNConfig, heat_equation_problem, solve_heat_problem


DEFAULT_PROBLEM = heat_equation_problem(alpha=0.1, x_min=0.0, x_max=1.0, t_min=0.0, t_max=1.0)
DEFAULT_CONFIG = PINNConfig(
    hidden_layers=4,
    neurons=64,
    learning_rate=8e-4,
    epochs=450,
    n_collocation=512,
    n_initial=96,
    n_boundary=96,
    batch_size=4096,
    seed=7,
)


@lru_cache(maxsize=8)
def run_cached(alpha: float, dx: float, dt: float, epochs: int, hidden_layers: int, neurons: int, seed: int):
    problem = heat_equation_problem(alpha=alpha, x_min=0.0, x_max=1.0, t_min=0.0, t_max=1.0)
    config = PINNConfig(
        hidden_layers=hidden_layers,
        neurons=neurons,
        learning_rate=DEFAULT_CONFIG.learning_rate,
        epochs=epochs,
        n_collocation=DEFAULT_CONFIG.n_collocation,
        n_initial=DEFAULT_CONFIG.n_initial,
        n_boundary=DEFAULT_CONFIG.n_boundary,
        batch_size=DEFAULT_CONFIG.batch_size,
        seed=seed,
    )
    return solve_heat_problem(problem, config, dx=dx, dt=dt)


def make_plots(result, time_value: float):
    t_index = int(np.argmin(np.abs(result.t - time_value)))
    actual_time = float(result.t[t_index])

    fig1, ax1 = plt.subplots(figsize=(7, 4))
    ax1.plot(result.x, result.u_fd[:, t_index], label="Finite difference", linewidth=2)
    ax1.plot(result.x, result.u_pinn[:, t_index], label="PINN", linewidth=2, linestyle="--")
    ax1.plot(result.x, result.u_true[:, t_index], label="Analytical", linewidth=2, linestyle=":")
    ax1.set_title(f"Solutions at t = {actual_time:.3f}")
    ax1.set_xlabel("x")
    ax1.set_ylabel("u(x,t)")
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    fig1.tight_layout()

    fig2, ax2 = plt.subplots(figsize=(7, 4))
    err = np.abs(result.u_pinn[:, t_index] - result.u_true[:, t_index])
    ax2.plot(result.x, err, color="crimson", linewidth=2)
    ax2.set_title(f"Absolute error at t = {actual_time:.3f}")
    ax2.set_xlabel("x")
    ax2.set_ylabel("|PINN - analytical|")
    ax2.grid(True, alpha=0.3)
    fig2.tight_layout()

    fig3, ax3 = plt.subplots(figsize=(7, 4))
    ax3.semilogy(result.training.losses, label="Total")
    ax3.semilogy(result.training.pde_losses, label="PDE")
    ax3.semilogy(result.training.ic_losses, label="Initial")
    ax3.semilogy(result.training.bc_losses, label="Boundary")
    ax3.set_title("Training losses")
    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("Loss")
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    fig3.tight_layout()

    return fig1, fig2, fig3


def summarise_result(result, time_value: float) -> str:
    cfl_text = "stable" if result.stable else "unstable"
    t_index = int(np.argmin(np.abs(result.t - time_value)))
    actual_time = float(result.t[t_index])
    return (
        f"Problem: {result.problem.name}\n"
        f"Stability check: alpha*dt/dx² = {result.stability_number:.4f} ({cfl_text})\n"
        f"Relative L2 error vs analytical solution: {result.relative_l2_error:.6f}\n"
        f"Displayed time slice: t = {actual_time:.3f}"
    )


def run_simulation(alpha, dx, dt, epochs, hidden_layers, neurons, time_value):
    result = run_cached(float(alpha), float(dx), float(dt), int(epochs), int(hidden_layers), int(neurons), DEFAULT_CONFIG.seed)
    fig1, fig2, fig3 = make_plots(result, float(time_value))
    status = summarise_result(result, float(time_value))
    state = {
        "alpha": float(alpha),
        "dx": float(dx),
        "dt": float(dt),
        "epochs": int(epochs),
        "hidden_layers": int(hidden_layers),
        "neurons": int(neurons),
    }
    return fig1, fig2, fig3, status, state


def update_time_slice(time_value, state):
    if not state:
        return None, None, None, "Run the simulation first to cache the solution."
    result = run_cached(
        state["alpha"],
        state["dx"],
        state["dt"],
        state["epochs"],
        state["hidden_layers"],
        state["neurons"],
        DEFAULT_CONFIG.seed,
    )
    fig1, fig2, fig3 = make_plots(result, float(time_value))
    return fig1, fig2, fig3, summarise_result(result, float(time_value))


with gr.Blocks(title="Generic PINN Heat Demo") as demo:
    gr.Markdown(
        "# Generic PINN Heat Demo\n"
        "A small reusable PINN engine with a Gradio front end.\n"
        "The default example is the 1D heat equation, but the engine accepts problem-specific residuals and boundary data.\n"
        "The time slider only reuses cached predictions; it does not retrain the model."
    )

    with gr.Row():
        alpha = gr.Number(label="Diffusivity α", value=0.1, precision=3)
        dx = gr.Number(label="Δx", value=0.05, precision=3)
        dt = gr.Number(label="Δt", value=0.01, precision=3)
        epochs = gr.Slider(50, 800, value=DEFAULT_CONFIG.epochs, step=25, label="Training epochs")
    with gr.Row():
        hidden_layers = gr.Slider(2, 5, value=DEFAULT_CONFIG.hidden_layers, step=1, label="Hidden layers")
        neurons = gr.Slider(16, 64, value=DEFAULT_CONFIG.neurons, step=8, label="Neurons per layer")
        time_value = gr.Slider(0.0, 1.0, value=0.0, step=0.05, label="Time slider")
        run_btn = gr.Button("Run simulation", variant="primary")

    state = gr.State(None)
    status = gr.Markdown("Run the simulation to see the stability check and the relative L2 error.")

    with gr.Row():
        solution_plot = gr.Plot(label="Solution slice")
        error_plot = gr.Plot(label="Absolute error slice")
        loss_plot = gr.Plot(label="Training losses")

    run_btn.click(
        fn=run_simulation,
        inputs=[alpha, dx, dt, epochs, hidden_layers, neurons, time_value],
        outputs=[solution_plot, error_plot, loss_plot, status, state],
    )

    time_value.change(
        fn=update_time_slice,
        inputs=[time_value, state],
        outputs=[solution_plot, error_plot, loss_plot, status],
    )

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7861, share=False)
