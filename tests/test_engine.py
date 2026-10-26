from __future__ import annotations

import numpy as np
import pytest
import tensorflow as tf

from engine import PINNConfig, heat_equation_problem, predict_batched, relative_l2_error, solve_heat_problem, stability_number


class DummyModel(tf.keras.Model):
    def call(self, inputs):
        return tf.reduce_sum(inputs, axis=1, keepdims=True)


def test_stability_number_matches_formula():
    assert stability_number(0.1, 0.05, 0.01) == pytest.approx(0.4)


def test_predict_batched_matches_direct_model_call():
    model = DummyModel()
    x = np.linspace(0, 1, 9).reshape(-1, 1)
    t = np.linspace(0, 1, 9).reshape(-1, 1)
    batched = predict_batched(model, x, t, batch_size=4)
    direct = model(tf.convert_to_tensor(np.column_stack([x.ravel(), t.ravel()]), dtype=tf.float32)).numpy().ravel()
    np.testing.assert_allclose(batched, direct)


@pytest.mark.slow
def test_heat_equation_solution_runs_and_is_reasonable():
    problem = heat_equation_problem(alpha=0.1)
    config = PINNConfig(
        hidden_layers=2,
        neurons=16,
        learning_rate=1e-3,
        epochs=40,
        n_collocation=64,
        n_initial=24,
        n_boundary=24,
        batch_size=1024,
        seed=5,
    )
    result = solve_heat_problem(problem, config, dx=0.1, dt=0.01)
    assert result.stable is True
    assert result.relative_l2_error < 0.5
    assert result.u_pinn.shape == result.u_true.shape
    assert relative_l2_error(result.u_pinn, result.u_true) == pytest.approx(result.relative_l2_error)
