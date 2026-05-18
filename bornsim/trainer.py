"""Training loop wrapper for bornsim circuits, losses, and optimizers."""

from __future__ import annotations

from dataclasses import dataclass

import cupy as cp
import jax.numpy as jnp
import numpy as np
import optax

from bornsim import adjoint, statevec
from bornsim.circuit import Circuit
from bornsim.losses import Loss


@dataclass(frozen=True, slots=True)
class GradientInfo:
    """One gradient evaluation payload."""

    loss_value: float
    grad: np.ndarray
    probs: cp.ndarray
    prob_sum: float
    forward_seconds: float
    backward_seconds: float


class Trainer:
    """Stateless trainer over a bornsim circuit and loss."""

    def __init__(
        self,
        circuit: Circuit,
        loss: Loss,
        optimizer: optax.GradientTransformation,
    ):
        self.circuit = circuit
        self.loss = loss
        self.optimizer = optimizer

    def init_opt_state(self, params: np.ndarray) -> optax.OptState:
        """Initialize optimizer state for a flat parameter vector."""
        return self.optimizer.init(jnp.asarray(params))

    def simulate(self, params: np.ndarray) -> cp.ndarray:
        """Run forward simulation and return the full probability vector."""
        state, _forward_seconds = statevec.simulate_state(
            self.circuit,
            np.asarray(params, dtype=np.float32),
        )
        return statevec.probabilities(state)

    def _gradient_info(self, params: np.ndarray) -> GradientInfo:
        params_np = np.asarray(params, dtype=np.float32)
        state, forward_seconds = statevec.simulate_state(self.circuit, params_np)
        probs = statevec.probabilities(state)
        loss_value, grad_probs = self.loss.value_and_probgrad(probs)
        cotangent = adjoint.cotangent_from_probgrad(state, probs, grad_probs)
        grad, backward_seconds = adjoint.gradient(self.circuit, params_np, state, cotangent)
        prob_sum = float(cp.sum(probs).get())
        return GradientInfo(
            loss_value=loss_value,
            grad=grad,
            probs=probs,
            prob_sum=prob_sum,
            forward_seconds=forward_seconds,
            backward_seconds=backward_seconds,
        )

    def gradient(self, params: np.ndarray) -> tuple[float, np.ndarray]:
        """Return the current loss value and parameter gradient."""
        info = self._gradient_info(params)
        return info.loss_value, info.grad

    def step(
        self,
        params: np.ndarray,
        opt_state: optax.OptState,
    ) -> tuple[np.ndarray, optax.OptState, dict[str, float]]:
        """Take one optimizer step."""
        info = self._gradient_info(params)
        updates, new_opt_state = self.optimizer.update(
            jnp.asarray(info.grad, dtype=jnp.float32),
            opt_state,
            jnp.asarray(params, dtype=jnp.float32),
        )
        new_params = np.asarray(
            optax.apply_updates(jnp.asarray(params, dtype=jnp.float32), updates),
            dtype=np.float32,
        )
        return (
            new_params,
            new_opt_state,
            {
                "loss_value": info.loss_value,
                "grad_norm": float(np.linalg.norm(info.grad)),
                "prob_sum": info.prob_sum,
                "forward_seconds": info.forward_seconds,
                "backward_seconds": info.backward_seconds,
            },
        )
