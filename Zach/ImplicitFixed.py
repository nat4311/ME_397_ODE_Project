"""Batched implicit fixed-step ODE solver (CPU multithreading).

Deliverable mapping:
- Implicit solver with fixed timestep for CPU with Python/multithreading.

Method:
- BDF1 / Backward Euler, solved with Newton iterations.
- Optional analytic Jacobian; finite-difference Jacobian fallback.
- Batched execution via ThreadPoolExecutor (default) or ProcessPoolExecutor.

Parallelism notes:
- "thread" backend: ThreadPoolExecutor. NumPy releases the GIL during linalg
  ops (solve, norm), enabling genuine multi-core execution.
- "process" backend: ProcessPoolExecutor. `rhs` and `jac` must be picklable.
"""

from __future__ import annotations

import os
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Callable, List, Literal, Optional, Sequence

import numpy as np


RHSFn = Callable[[float, np.ndarray, Any], np.ndarray]
JacFn = Callable[[float, np.ndarray, Any], np.ndarray]


@dataclass
class BDF1FixedConfig:
    dt: float = 1e-2
    newton_tol: float = 1e-10
    newton_max_iter: int = 20
    jac_eps: float = 1e-8


@dataclass
class TrajectoryResult:
    times: np.ndarray
    states: np.ndarray
    success: bool
    n_steps: int
    n_newton_total: int
    message: str


@dataclass
class BatchResult:
    trajectories: List[TrajectoryResult]
    wall_time: float = 0.0


# ---------------------------------------------------------------------------
# Core numerics
# ---------------------------------------------------------------------------

def _finite_diff_jacobian(
    rhs: RHSFn, t: float, y: np.ndarray, params: Any, eps: float,
) -> np.ndarray:
    n = y.size
    J = np.zeros((n, n), dtype=float)
    f0 = rhs(t, y, params)
    for j in range(n):
        y_pert = y.copy()
        dy = eps * max(1.0, abs(y[j]))
        y_pert[j] += dy
        J[:, j] = (rhs(t, y_pert, params) - f0) / dy
    return J


def _newton_bdf1_step(
    rhs: RHSFn,
    jac: Optional[JacFn],
    t_next: float,
    y_prev: np.ndarray,
    dt: float,
    params: Any,
    cfg: BDF1FixedConfig,
) -> tuple[np.ndarray, int, bool]:
    """Newton solve for BE residual: y - y_prev - dt*f(t_next, y) = 0."""
    y = y_prev.copy()
    n = y.size
    I = np.eye(n)

    for k in range(cfg.newton_max_iter):
        f_val = rhs(t_next, y, params)
        res = y - y_prev - dt * f_val
        if np.linalg.norm(res, ord=np.inf) < cfg.newton_tol:
            return y, k + 1, True

        Jf = (jac(t_next, y, params)
              if jac is not None
              else _finite_diff_jacobian(rhs, t_next, y, params, cfg.jac_eps))
        try:
            delta = np.linalg.solve(I - dt * Jf, -res)
        except np.linalg.LinAlgError:
            return y, k + 1, False

        y = y + delta
        if np.linalg.norm(delta, ord=np.inf) < cfg.newton_tol:
            return y, k + 1, True

    return y, cfg.newton_max_iter, False


def solve_trajectory_implicit_fixed(
    rhs: RHSFn,
    t_span: tuple[float, float],
    y0: np.ndarray,
    params: Any = None,
    jac: Optional[JacFn] = None,
    config: BDF1FixedConfig = BDF1FixedConfig(),
) -> TrajectoryResult:
    t0, tf = t_span
    if tf <= t0:
        raise ValueError("t_span must satisfy tf > t0.")
    if config.dt <= 0:
        raise ValueError("dt must be positive.")

    y0 = np.asarray(y0, dtype=float)
    n = y0.size
    n_steps = int(np.ceil((tf - t0) / config.dt))
    times = np.empty(n_steps + 1)
    states = np.empty((n_steps + 1, n))
    times[0] = t0
    states[0] = y0

    y = y0.copy()
    t = float(t0)
    n_newton_total = 0

    for step in range(1, n_steps + 1):
        dt = min(config.dt, tf - t)
        t_next = t + dt

        y_next, n_newton, ok = _newton_bdf1_step(
            rhs, jac, t_next, y, dt, params, config,
        )
        n_newton_total += n_newton
        if not ok:
            return TrajectoryResult(
                times=times[:step], states=states[:step],
                success=False, n_steps=step - 1,
                n_newton_total=n_newton_total,
                message=f"Newton failed at step {step}.",
            )

        y = y_next
        t = t_next
        times[step] = t
        states[step] = y

    return TrajectoryResult(
        times=times, states=states, success=True,
        n_steps=n_steps, n_newton_total=n_newton_total, message="OK",
    )


# ---------------------------------------------------------------------------
# Module-level worker (must be top-level for ProcessPoolExecutor pickling)
# ---------------------------------------------------------------------------

def _implicit_fixed_worker(args: tuple) -> tuple[int, TrajectoryResult]:
    idx, rhs, t_span, y0, params, jac, config = args
    return idx, solve_trajectory_implicit_fixed(rhs, t_span, y0, params, jac, config)


# ---------------------------------------------------------------------------
# Batched driver
# ---------------------------------------------------------------------------

def solve_batch_implicit_fixed(
    rhs: RHSFn,
    t_span: tuple[float, float],
    y0_batch: np.ndarray,
    params_batch: Optional[Sequence[Any]] = None,
    jac: Optional[JacFn] = None,
    config: BDF1FixedConfig = BDF1FixedConfig(),
    max_workers: Optional[int] = None,
    backend: Literal["thread", "process"] = "thread",
) -> BatchResult:
    """Solve a batch of IVPs with fixed-step BDF1 in parallel.

    Parameters
    ----------
    backend : "thread" (default) or "process"
        "process" requires `rhs` and `jac` to be picklable (module-level).
    """
    y0_batch = np.asarray(y0_batch, dtype=float)
    if y0_batch.ndim == 1:
        y0_batch = y0_batch[:, None]

    batch_size = y0_batch.shape[0]
    if params_batch is None:
        params_batch = [None] * batch_size
    if len(params_batch) != batch_size:
        raise ValueError("params_batch length must match batch dimension.")

    if max_workers is None:
        max_workers = min(batch_size, os.cpu_count() or 1)

    work = [
        (i, rhs, t_span, y0_batch[i], params_batch[i], jac, config)
        for i in range(batch_size)
    ]

    results: List[Optional[TrajectoryResult]] = [None] * batch_size
    Executor = ProcessPoolExecutor if backend == "process" else ThreadPoolExecutor

    t0 = time.perf_counter()
    with Executor(max_workers=max_workers) as pool:
        futures = {pool.submit(_implicit_fixed_worker, w): w[0] for w in work}
        for fut in as_completed(futures):
            idx, out = fut.result()
            results[idx] = out
    wall = time.perf_counter() - t0

    return BatchResult(
        trajectories=[r for r in results if r is not None],
        wall_time=wall,
    )


# ---------------------------------------------------------------------------
# Demo / validation
# ---------------------------------------------------------------------------

def _demo_rhs(t: float, y: np.ndarray, params: Any) -> np.ndarray:
    """Stiff scalar/vector test: y' = -λ*(y - cos(t)) - sin(t).
    Exact solution: y(t) = (y0 - 1)*e^{-λt} + cos(t).
    """
    lam = params["lam"]
    return -lam * (y - np.cos(t)) - np.sin(t)


def _demo_jac(t: float, y: np.ndarray, params: Any) -> np.ndarray:
    lam = np.asarray(params["lam"], dtype=float)
    return np.diag(-lam)


def _exact_solution(t: float, y0: np.ndarray, lam: np.ndarray) -> np.ndarray:
    """Exact solution: y(t) = (y0 - 1)*e^{-λt} + cos(t)."""
    return (y0 - 1.0) * np.exp(-lam * t) + np.cos(t)


def _run_demo() -> None:
    rng = np.random.default_rng(3)
    batch_size = 32
    dim = 3
    t_span = (0.0, 5.0)

    y0_batch = rng.normal(size=(batch_size, dim))
    lam_batch = [rng.uniform(20.0, 150.0, size=dim) for _ in range(batch_size)]
    params_batch = [{"lam": lam} for lam in lam_batch]

    cfg = BDF1FixedConfig(dt=0.005, newton_tol=1e-11, newton_max_iter=25, jac_eps=1e-8)

    print("=" * 60)
    print("Implicit fixed-step BDF1 — batched CPU benchmark")
    print("=" * 60)

    for nw, label in [(1, "serial (1 worker)"), (None, f"parallel ({os.cpu_count()} workers)")]:
        out = solve_batch_implicit_fixed(
            rhs=_demo_rhs, t_span=t_span,
            y0_batch=y0_batch, params_batch=params_batch,
            jac=_demo_jac, config=cfg,
            max_workers=nw, backend="thread",
        )
        n_ok = sum(1 for tr in out.trajectories if tr.success)
        avg_nit = np.mean([
            tr.n_newton_total / max(1, tr.n_steps)
            for tr in out.trajectories
        ])
        print(f"  {label:30s}  {out.wall_time:.3f}s  "
              f"success={n_ok}/{batch_size}  avg_newton/step={avg_nit:.2f}")

    # --- Correctness vs exact solution ---
    out_ref = solve_batch_implicit_fixed(
        rhs=_demo_rhs, t_span=t_span,
        y0_batch=y0_batch, params_batch=params_batch,
        jac=_demo_jac, config=cfg,
    )
    tf = t_span[1]
    errors = []
    for i, tr in enumerate(out_ref.trajectories):
        if tr.success:
            y_exact = _exact_solution(tf, y0_batch[i], lam_batch[i])
            errors.append(np.max(np.abs(tr.states[-1] - y_exact)))
    print(f"\n  Max pointwise error vs exact: {np.max(errors):.2e}")
    print(f"  Mean pointwise error        : {np.mean(errors):.2e}")
    print("  (Fixed-step BDF1 is O(h); dt=0.005 → O(1e-2) expected)")


if __name__ == "__main__":
    _run_demo()
