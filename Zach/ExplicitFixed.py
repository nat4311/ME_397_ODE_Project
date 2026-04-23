"""Batched explicit variable-step ODE solver (CPU multithreading).

Deliverable mapping:
- Explicit solver with variable timestep for CPU with Python/multithreading.

Method:
- Dormand-Prince RK45 with adaptive timestep control.
- Batched execution via ThreadPoolExecutor (default) or ProcessPoolExecutor.

Parallelism notes:
- "thread" backend (default): ThreadPoolExecutor. NumPy releases the GIL during
  array operations, so multiple threads execute RK45 stages concurrently.
- "process" backend: ProcessPoolExecutor. Bypasses GIL entirely; requires that
  the `rhs` callable be defined at module level (picklable).
"""

from __future__ import annotations

import os
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Callable, List, Literal, Optional, Sequence

import numpy as np


RHSFn = Callable[[float, np.ndarray, Any], np.ndarray]


@dataclass
class RK45Config:
    rtol: float = 1e-6
    atol: float = 1e-9
    h_init: float = 1e-2
    h_min: float = 1e-8
    h_max: float = 0.5
    max_steps: int = 200_000
    max_reject: int = 10_000
    safety: float = 0.9
    min_factor: float = 0.2
    max_factor: float = 5.0


@dataclass
class TrajectoryResult:
    times: np.ndarray
    states: np.ndarray
    n_accept: int
    n_reject: int
    success: bool
    message: str


@dataclass
class BatchResult:
    trajectories: List[TrajectoryResult]
    wall_time: float = 0.0


# ---------------------------------------------------------------------------
# Core numerics
# ---------------------------------------------------------------------------

def _error_norm(
    y_old: np.ndarray, y_new: np.ndarray, err: np.ndarray,
    atol: float, rtol: float,
) -> float:
    scale = atol + rtol * np.maximum(np.abs(y_old), np.abs(y_new))
    ratio = err / scale
    return float(np.sqrt(np.mean(ratio * ratio)))


def _rk45_step(
    rhs: RHSFn, t: float, y: np.ndarray, h: float, params: Any,
) -> tuple[np.ndarray, np.ndarray]:
    """Dormand-Prince 5(4) step; returns (y5, y5-y4)."""
    k1 = rhs(t, y, params)
    k2 = rhs(t + h / 5,       y + h * (1/5 * k1), params)
    k3 = rhs(t + 3*h/10,      y + h * (3/40 * k1 + 9/40 * k2), params)
    k4 = rhs(t + 4*h/5,
             y + h * (44/45 * k1 - 56/15 * k2 + 32/9 * k3), params)
    k5 = rhs(t + 8*h/9,
             y + h * (19372/6561*k1 - 25360/2187*k2
                      + 64448/6561*k3 - 212/729*k4), params)
    k6 = rhs(t + h,
             y + h * (9017/3168*k1 - 355/33*k2 + 46732/5247*k3
                      + 49/176*k4 - 5103/18656*k5), params)

    # 5th-order solution (also used to evaluate k7 for FSAL book-keeping).
    y5 = y + h * (35/384*k1 + 500/1113*k3 + 125/192*k4
                  - 2187/6784*k5 + 11/84*k6)

    # 4th-order solution for error estimate.
    y4 = y + h * (5179/57600*k1 + 7571/16695*k3 + 393/640*k4
                  - 92097/339200*k5 + 187/2100*k6
                  + 1/40 * rhs(t + h, y5, params))

    return y5, (y5 - y4)


def solve_trajectory_explicit_variable(
    rhs: RHSFn,
    t_span: tuple[float, float],
    y0: np.ndarray,
    params: Any = None,
    config: RK45Config = RK45Config(),
) -> TrajectoryResult:
    t0, tf = t_span
    if tf <= t0:
        raise ValueError("t_span must satisfy tf > t0.")

    t = float(t0)
    y = np.asarray(y0, dtype=float).copy()
    h = float(np.clip(config.h_init, config.h_min, config.h_max))

    times = [t]
    states = [y.copy()]
    n_accept = 0
    n_reject = 0

    while t < tf and n_accept + n_reject < config.max_steps:
        h = min(h, tf - t)
        if h < config.h_min:
            return TrajectoryResult(
                np.asarray(times), np.vstack(states),
                n_accept, n_reject, False, "Step size underflow.",
            )

        y_cand, err_vec = _rk45_step(rhs, t, y, h, params)
        err = _error_norm(y, y_cand, err_vec, config.atol, config.rtol)

        if err <= 1.0:
            t += h
            y = y_cand
            times.append(t)
            states.append(y.copy())
            n_accept += 1
            factor = (config.max_factor if err == 0.0
                      else config.safety * err ** (-0.2))
            factor = np.clip(factor, config.min_factor, config.max_factor)
            h = float(np.clip(h * factor, config.h_min, config.h_max))
        else:
            n_reject += 1
            factor = np.clip(config.safety * err ** (-0.2),
                             config.min_factor, 1.0)
            h = float(max(config.h_min, h * factor))
            if n_reject > config.max_reject:
                return TrajectoryResult(
                    np.asarray(times), np.vstack(states),
                    n_accept, n_reject, False, "Too many rejected steps.",
                )

    success = t >= tf
    return TrajectoryResult(
        np.asarray(times), np.vstack(states),
        n_accept, n_reject, success,
        "OK" if success else "Maximum step count reached.",
    )


# ---------------------------------------------------------------------------
# Module-level worker (must be top-level for ProcessPoolExecutor pickling)
# ---------------------------------------------------------------------------

def _explicit_worker(args: tuple) -> tuple[int, TrajectoryResult]:
    idx, rhs, t_span, y0, params, config = args
    return idx, solve_trajectory_explicit_variable(rhs, t_span, y0, params, config)


# ---------------------------------------------------------------------------
# Batched driver
# ---------------------------------------------------------------------------

def solve_batch_explicit_variable(
    rhs: RHSFn,
    t_span: tuple[float, float],
    y0_batch: np.ndarray,
    params_batch: Optional[Sequence[Any]] = None,
    config: RK45Config = RK45Config(),
    max_workers: Optional[int] = None,
    backend: Literal["thread", "process"] = "thread",
) -> BatchResult:
    """Solve a batch of IVPs in parallel.

    Parameters
    ----------
    backend : "thread" (default) or "process"
        "thread"  — ThreadPoolExecutor; GIL released during NumPy ops.
        "process" — ProcessPoolExecutor; `rhs` must be picklable (module-level).
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
        (i, rhs, t_span, y0_batch[i], params_batch[i], config)
        for i in range(batch_size)
    ]

    results: List[Optional[TrajectoryResult]] = [None] * batch_size
    Executor = ProcessPoolExecutor if backend == "process" else ThreadPoolExecutor

    t0 = time.perf_counter()
    with Executor(max_workers=max_workers) as pool:
        futures = {pool.submit(_explicit_worker, w): w[0] for w in work}
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
    """Decoupled mildly stiff linear: y' = -λ*y + sin(t).
    Exact solution: y(t) = e^{-λt}*(y0 - λ/(λ²+1)) + (λ*sin(t)-cos(t))/(λ²+1) + 1/(λ²+1)...
    simplified as y(t) = c*e^{-λt} + (-cos(t)+λ*sin(t))/(λ²+1).
    """
    lam = params["lam"]
    return -lam * y + np.sin(t)


def _exact_solution(t: float, y0: np.ndarray, lam: np.ndarray) -> np.ndarray:
    """Closed-form solution for y' = -λy + sin(t)."""
    c = y0 - (-1.0 / (lam**2 + 1.0))
    return c * np.exp(-lam * t) + (-np.cos(t) + lam * np.sin(t)) / (lam**2 + 1.0)


def _run_demo() -> None:
    rng = np.random.default_rng(7)
    batch_size = 32
    dim = 4
    t_span = (0.0, 10.0)

    y0_batch = rng.normal(size=(batch_size, dim))
    lam_batch = [rng.uniform(0.5, 8.0, size=dim) for _ in range(batch_size)]
    params_batch = [{"lam": lam} for lam in lam_batch]

    cfg = RK45Config(rtol=1e-6, atol=1e-8, h_init=1e-3, h_min=1e-7, h_max=0.2)

    # --- Scaling: 1 worker vs max_workers ---
    print("=" * 60)
    print("Explicit variable-step RK45 — batched CPU benchmark")
    print("=" * 60)

    for nw, label in [(1, "serial (1 worker)"), (None, f"parallel ({os.cpu_count()} workers)")]:
        out = solve_batch_explicit_variable(
            rhs=_demo_rhs,
            t_span=t_span,
            y0_batch=y0_batch,
            params_batch=params_batch,
            config=cfg,
            max_workers=nw,
            backend="thread",
        )
        n_ok = sum(1 for tr in out.trajectories if tr.success)
        avg_accept = np.mean([tr.n_accept for tr in out.trajectories])
        print(f"  {label:30s}  {out.wall_time:.3f}s  "
              f"success={n_ok}/{batch_size}  avg_accept={avg_accept:.0f}")

    # --- Correctness vs exact solution ---
    out_ref = solve_batch_explicit_variable(
        rhs=_demo_rhs, t_span=t_span,
        y0_batch=y0_batch, params_batch=params_batch,
        config=cfg,
    )
    tf = t_span[1]
    errors = []
    for i, tr in enumerate(out_ref.trajectories):
        if tr.success:
            y_exact = _exact_solution(tf, y0_batch[i], lam_batch[i])
            errors.append(np.max(np.abs(tr.states[-1] - y_exact)))
    print(f"\n  Max pointwise error vs exact solution: {np.max(errors):.2e}")
    print(f"  Mean pointwise error                 : {np.mean(errors):.2e}")


if __name__ == "__main__":
    _run_demo()
