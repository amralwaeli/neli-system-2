from __future__ import annotations

from pathlib import Path
import csv
import os

import gradio as gr
import matplotlib.pyplot as plt

import phase5

ROOT = Path(__file__).resolve().parent
PHASE5_OUT = ROOT / "outputs" / "phase5"
PHASE6_OUT = ROOT / "outputs" / "phase6"
PHASE6_OUT.mkdir(parents=True, exist_ok=True)

PRESETS = [
    "Exponential decay",
    "Harmonic oscillator",
    "Logistic growth",
    "1D wave equation",
    "Viscous Burgers",
    "2D Poisson",
]

PRESET_FILES = {
    "Exponential decay": {
        "main1": PHASE5_OUT / "exponential_decay_reference_vs_pinn.png",
        "main2": None,
        "error": PHASE5_OUT / "exponential_decay_error.png",
        "loss": PHASE5_OUT / "exponential_decay_loss.png",
    },
    "Harmonic oscillator": {
        "main1": PHASE5_OUT / "harmonic_oscillator_reference_vs_pinn.png",
        "main2": None,
        "error": PHASE5_OUT / "harmonic_oscillator_error.png",
        "loss": PHASE5_OUT / "harmonic_oscillator_loss.png",
    },
    "Logistic growth": {
        "main1": PHASE5_OUT / "logistic_growth_reference_vs_pinn.png",
        "main2": None,
        "error": PHASE5_OUT / "logistic_growth_error.png",
        "loss": PHASE5_OUT / "logistic_growth_loss.png",
    },
    "1D wave equation": {
        "main1": PHASE5_OUT / "1d_wave_equation_reference.png",
        "main2": PHASE5_OUT / "1d_wave_equation_pinn.png",
        "error": PHASE5_OUT / "1d_wave_equation_error.png",
        "loss": PHASE5_OUT / "1d_wave_equation_loss.png",
    },
    "Viscous Burgers": {
        "main1": PHASE5_OUT / "viscous_burgers_reference.png",
        "main2": PHASE5_OUT / "viscous_burgers_pinn.png",
        "error": PHASE5_OUT / "viscous_burgers_error.png",
        "loss": PHASE5_OUT / "viscous_burgers_loss.png",
    },
    "2D Poisson": {
        "main1": PHASE5_OUT / "2d_poisson_reference.png",
        "main2": PHASE5_OUT / "2d_poisson_pinn.png",
        "error": PHASE5_OUT / "2d_poisson_error.png",
        "loss": PHASE5_OUT / "2d_poisson_loss.png",
    },
}


def load_metrics_table():
    rows = []
    csv_path = PHASE5_OUT / "phase5_summary.csv"
    if csv_path.exists():
        with csv_path.open("r", encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
    return rows


METRICS = {row["name"]: row for row in load_metrics_table()}


def preset_view(name: str):
    files = PRESET_FILES[name]
    row = METRICS.get(name, {})
    metrics_md = (
        f"### {name}\n"
        f"- Relative L2: {row.get('rel_l2', 'n/a')}\n"
        f"- Max abs error: {row.get('max_abs', 'n/a')}\n"
        f"- Train time: {row.get('train_seconds', 'n/a')} s"
    )
    if name == "Viscous Burgers":
        metrics_md += "\n- Verification problem: includes a forcing term so the exact answer is known."
    return str(files["main1"]), str(files["main2"]) if files["main2"] else None, str(files["error"]), str(files["loss"]), metrics_md


def _loss_figure(result):
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.semilogy(result.history["total"], label="Total")
    ax.semilogy(result.history["pde"], label="PDE")
    if any(v != 0.0 for v in result.history["ic"]):
        ax.semilogy(result.history["ic"], label="IC")
    if any(v != 0.0 for v in result.history.get("bc", [])):
        ax.semilogy(result.history["bc"], label="Boundary")
    ax.set_title(f"{result.name} loss curves")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig


def failure_experiment(name: str):
    problem_map = {
        "Exponential decay": phase5.make_exponential_decay_problem,
        "Harmonic oscillator": phase5.make_harmonic_oscillator_problem,
        "Logistic growth": phase5.make_logistic_growth_problem,
        "1D wave equation": phase5.make_wave_problem,
        "Viscous Burgers": phase5.make_burgers_problem,
        "2D Poisson": phase5.make_poisson_problem,
    }
    p = problem_map[name]()
    if p.category == "ODE":
        result = phase5.train(p, epochs=40, hidden_layers=2, neurons=16, n_collocation=8, n_initial=8)
        explanation = "Failure mode: the model saw very few collocation points, so it fit the initial condition but missed the physics between samples."
    elif name == "2D Poisson":
        result = phase5.train(p, epochs=40, hidden_layers=2, neurons=16, n_collocation=16, n_boundary=8)
        explanation = "Failure mode: too few interior and boundary samples leave the PDE underconstrained, so the surface bends the wrong way away from the sampled points."
    else:
        result = phase5.train(p, epochs=40, hidden_layers=2, neurons=16, n_collocation=16, n_initial=8, n_boundary=8)
        explanation = "Failure mode: very few samples leave the PDE underconstrained, so the network can match a handful of points while missing the full solution shape."

    figs = result.problem.plot_fn(result)
    loss_fig = _loss_figure(result)
    status = (
        f"### Failure-driven learning on {name}\n"
        f"- Relative L2: {result.rel_l2:.6f}\n"
        f"- Train time: {result.train_seconds:.1f} s\n"
        f"- {explanation}"
    )
    if len(figs) == 3:
        return figs[0], figs[1], figs[2], loss_fig, status

    return figs[0], figs[1], figs[2], loss_fig, status


with gr.Blocks(title="Phase 6 PINN Presets") as demo:
    gr.Markdown(
        "# Phase 6: failure-driven learning and preset browser\n"
        "Pick a problem preset to load the saved reference/PINN/error/loss outputs and metrics.\n"
        "Then run a failure experiment to see how the model breaks when the physics samples are starved."
    )

    with gr.Row():
        preset = gr.Dropdown(PRESETS, value=PRESETS[0], label="Preset")
        failure_btn = gr.Button("Run failure-driven experiment", variant="primary")
    gr.Markdown("Boundary/initial conditions are built into the network (hard constraints); the ODE/PDE residual is the training loss.")

    with gr.Row():
        main1 = gr.Image(label="Main plot 1", type="filepath")
        main2 = gr.Image(label="Main plot 2", type="filepath")
    with gr.Row():
        error_plot = gr.Image(label="Error plot", type="filepath")
        loss_plot = gr.Image(label="Loss curves", type="filepath")

    metrics = gr.Markdown()
    failure_status = gr.Markdown()
    with gr.Row():
        fail_main1 = gr.Plot(label="Failure plot 1")
        fail_main2 = gr.Plot(label="Failure plot 2")
    with gr.Row():
        fail_error = gr.Plot(label="Failure error plot")
        fail_loss = gr.Plot(label="Failure loss plot")

    preset.change(
        fn=preset_view,
        inputs=[preset],
        outputs=[main1, main2, error_plot, loss_plot, metrics],
    )

    failure_btn.click(
        fn=failure_experiment,
        inputs=[preset],
        outputs=[fail_main1, fail_main2, fail_error, fail_loss, failure_status],
    )

    demo.load(
        fn=preset_view,
        inputs=[preset],
        outputs=[main1, main2, error_plot, loss_plot, metrics],
    )

if __name__ == "__main__":
    # NALI_SHARE=1 creates a public link (share=True) for reviewers.
    share = os.environ.get("NALI_SHARE", "0").lower() in {"1", "true", "yes"}
    demo.launch(
        server_name="0.0.0.0" if share else "127.0.0.1",
        server_port=7861,
        share=share,
    )
