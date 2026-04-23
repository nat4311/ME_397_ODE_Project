"""Batched implicit variable-step ODE solver (CPU multithreading).

Deliverable mapping:
- Implicit solver with variable timestep for CPU with Python/multithreading.

Method:
- Startup with Backward Euler (BDF1), then adaptive variable-step BDF2.
- Newton iterations for implicit solves.
- Error estimate from Richardson extrapolation: BDF2 (order 2) vs BE (order 1).
- Batched execution via ThreadPoolExecutor (default) or ProcessPoolExecutor.

Step-size control:
- Error norm uses max(|y_n|, |y_{n+1}|) mixed tolerance (standard VODE style).
- Factor exponent: safety * err^{-1/(p+1)} with p = order of the high-order solution.
  During startup (p=1 BE only, first accepted step always takes one small step),
  then BDF2 (p=2) with factor = err^{-1/3}.

Parallelism notes:
- "thread" backend: ThreadPoolExecutor. NumPy linalg releases the GIL.
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
class VariableBDFConfig:
    rtol: float = 1e-6
    atol: float = 1e-9
    h_init: float = 1e-2
    h_min: float = 1e-8
    h_max: float = 0.5
    max_steps: int = 200_000
    max_reject: int = 10_000
    newton_tol: float = 1e-10
    newton_max_iter: int = 20
    jac_eps: float = 1e-8
    safety: float = 0.9
    min_factor: float = 0.2
    max_factor: float = 3.0


@dataclass
class TrajectoryResult:
    times: np.ndarray
    states: np.ndarray
    n_accept: int
    n_reject: int
    n_newton_total: int
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
    y_ref: np.ndarray, y_new: np.ndarray, err: np.ndarray,
    atol: float, rtol: float,
) -> float:
    scale = atol + rtol * np.maximum(np.abs(y_ref), np.abs(y_new))
    ratio = err / scale
    return float(np.sqrt(np.mean(ratio * ratio)))


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


def _newton_solve(
    residual_and_jacobian: Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]],
    y0_guess: np.ndarray,
    tol: float,
    max_iter: int,
) -> tuple[np.ndarray, int, bool]:
    y = y0_guess.copy()
    for k in range(max_iter):
        res, jac = residual_and_jacobian(y)
        if np.linalg.norm(res, ord=np.inf) < tol:
            return y, k + 1, True
        try:
            delta = np.linalg.solve(jac, -res)
        except np.linalg.LinAlgError:
            return y, k + 1, False
        y = y + delta
        if np.linalg.norm(delta, ord=np.inf) < tol:
            return y, k + 1, True
    return y, max_iter, False


def _backward_euler_step(
    rhs: RHSFn,
    jac: Optional[JacFn],
    t_next: float,
    y_prev: np.ndarray,
    h: float,
    params: Any,
    cfg: VariableBDFConfig,
) -> tuple[np.ndarray, int, bool]:
    """Newton solve for BE residual: y - y_prev - h*f(t_next, y) = 0."""
    n = y_prev.size
    I = np.eye(n)

    def residual_jac(y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        f = rhs(t_next, y, params)
        Jf = (jac(t_next, y, params)
              if jac is not None
              else _finite_diff_jacobian(rhs, t_next, y, params, cfg.jac_eps))
        return y - y_prev - h * f, I - h * Jf

    return _newton_solve(residual_jac, y_prev, cfg.newton_tol, cfg.newton_max_iter)


def _bdf2_variable_step(
    rhs: RHSFn,
    jac: Optional[JacFn],
    t_next: float,
    y_nm1: np.ndarray,
    y_n: np.ndarray,
    h_prev: float,
    h: float,
    params: Any,
    cfg: VariableBDFConfig,
) -> tuple[np.ndarray, int, bool]:
    """Variable-step BDF2.

    Derivative approximation at t_{n+1}:
        a0*y_{n+1} + a1*y_n + a2*y_{n-1}  =  f(t_{n+1}, y_{n+1})

    Coefficients (Krogh 1974 / Shampine):
        r  = h / h_prev
        a0 = (1 + 2r) / (h*(1+r))
        a1 = -(1+r) / h
        a2 = r^2 / (h*(1+r))

    Uniform step (r=1) reduces to the standard 3*y_{n+1} - 4*y_n + y_{n-1} = 2h*f.
    """
    r = h / h_prev
    a0 = (1.0 + 2.0 * r) / (h * (1.0 + r))
    a1 = -(1.0 + r) / h
    a2 = (r * r) / (h * (1.0 + r))
    n = y_n.size
    I = np.eye(n)

    # Linear extrapolation predictor (consistent with BDF2 order).
    y_guess = y_n + (h / h_prev) * (y_n - y_nm1)

    def residual_jac(y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        f = rhs(t_next, y, params)
        Jf = (jac(t_next, y, params)
              if jac is not None
              else _finite_diff_jacobian(rhs, t_next, y, params, cfg.jac_eps))
        return a0 * y + a1 * y_n + a2 * y_nm1 - f, a0 * I - Jf

    return _newton_solve(residual_jac, y_guess, cfg.newton_tol, cfg.newton_max_iter)


def solve_trajectory_implicit_variable(
    rhs: RHSFn,
    t_span: tuple[float, float],
    y0: np.ndarray,
    params: Any = None,
    jac: Optional[JacFn] = None,
    config: VariableBDFConfig = VariableBDFConfig(),
) -> TrajectoryResult:
    t0, tf = t_span
    if tf <= t0:
        raise ValueError("t_span must satisfy tf > t0.")

    y0 = np.asarray(y0, dtype=float)
    t = float(t0)
    y_n = y0.copy()
    y_nm1: Optional[np.ndarray] = None
    h = float(np.clip(config.h_init, config.h_min, config.h_max))
    h_prev: Optional[float] = None

    times = [t]
    states = [y_n.copy()]
    n_accept = n_reject = n_newton_total = 0

    while t < tf and n_accept + n_reject < config.max_steps:
        h = min(h, tf - t)
        if h < config.h_min:
            return TrajectoryResult(
                np.asarray(times), np.vstack(states),
                n_accept, n_reject, n_newton_total, False, "Step size underflow.",
            )

        t_next = t + h

        # Backward Euler step (always computed — needed for error estimate).
        y_be, it_be, ok_be = _backward_euler_step(
            rhs, jac, t_next, y_n, h, params, config,
        )
        n_newton_total += it_be

        if not ok_be:
            n_reject += 1
            h = max(config.h_min, h * 0.5)
            if n_reject > config.max_reject:
                return TrajectoryResult(
                    np.asarray(times), np.vstack(states),
                    n_accept, n_reject, n_newton_total, False,
                    "Too many rejected steps (BE Newton failures).",
                )
            continue

        if y_nm1 is None or h_prev is None:
            # Startup: accept BE step unconditionally with zero error.
            err = 0.0
            y_high = y_be
            p = 1
        else:
            y_bdf2, it_bdf2, ok_bdf2 = _bdf2_variable_step(
                rhs, jac, t_next, y_nm1, y_n, h_prev, h, params, config,
            )
            n_newton_total += it_bdf2

            if not ok_bdf2:
                n_reject += 1
                h = max(config.h_min, h * 0.5)
                if n_reject > config.max_reject:
                    return TrajectoryResult(
                        np.asarray(times), np.vstack(states),
                        n_accept, n_reject, n_newton_total, False,
                        "Too many rejected steps (BDF2 Newton failures).",
                    )
                continue

            y_high = y_bdf2
            # Error estimate: difference between BDF2 (O(h^3)) and BE (O(h^2)).
            err = _error_norm(y_n, y_high, y_high - y_be, config.atol, config.rtol)
            p = 2

        if err <= 1.0:
            y_nm1 = y_n.copy()
            y_n = y_high
            t = t_next
            h_prev = h
            times.append(t)
            states.append(y_n.copy())
            n_accept += 1

            factor = (config.max_factor if err == 0.0
                      else float(np.clip(
                          config.safety * err ** (-1.0 / (p + 1.0)),
                          config.min_factor, config.max_factor,
                      )))
            h = float(np.clip(h * factor, config.h_min, config.h_max))
        else:
            n_reject += 1
            factor = float(np.clip(
                config.safety * err ** (-1.0 / (p + 1.0)),
                config.min_factor, 1.0,
            ))
            h = float(max(config.h_min, h * factor))
            if n_reject > config.max_reject:
                return TrajectoryResult(
                    np.asarray(times), np.vstack(states),
                    n_accept, n_reject, n_newton_total, False,
                    "Too many rejected steps (error control).",
                )

    success = t >= tf
    return TrajectoryResult(
        np.asarray(times), np.vstack(states),
        n_accept, n_reject, n_newton_total, success,
        "OK" if success else "Maximum step count reached.",
    )


# ---------------------------------------------------------------------------
# Module-level worker (must be top-level for ProcessPoolExecutor pickling)
# ---------------------------------------------------------------------------

def _implicit_variable_worker(args: tuple) -> tuple[int, TrajectoryResult]:
    idx, rhs, t_span, y0, params, jac, config = args
    return idx, solve_trajectory_implicit_variable(
        rhs, t_span, y0, params, jac, config,
    )


# ---------------------------------------------------------------------------
# Batched driver
# ---------------------------------------------------------------------------

def solve_batch_implicit_variable(
    rhs: RHSFn,
    t_span: tuple[float, float],
    y0_batch: np.ndarray,
    params_batch: Optional[Sequence[Any]] = None,
    jac: Optional[JacFn] = None,
    config: VariableBDFConfig = VariableBDFConfig(),
    max_workers: Optional[int] = None,
    backend: Literal["thread", "process"] = "thread",
) -> BatchResult:
    """Solve a batch of IVPs with variable-step BDF2 in parallel.

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

    t0_wall = time.perf_counter()
    with Executor(max_workers=max_workers) as pool:
        futures = {pool.submit(_implicit_variable_worker, w): w[0] for w in work}
        for fut in as_completed(futures):
            idx, out = fut.result()
            results[idx] = out
    wall = time.perf_counter() - t0_wall

    return BatchResult(
        trajectories=[r for r in results if r is not None],
        wall_time=wall,
    )


# ---------------------------------------------------------------------------
# Demo / validation
# ---------------------------------------------------------------------------

def _demo_rhs(t: float, y: np.ndarray, params: Any) -> np.ndarray:
    """Stiff test: y' = -λ*(y - cos(t)) - sin(t).
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
    rng = np.random.default_rng(19)
    batch_size = 32
    dim = 3
    t_span = (0.0, 6.0)

    y0_batch = rng.normal(size=(batch_size, dim))
    lam_batch = [rng.uniform(10.0, 120.0, size=dim) for _ in range(batch_size)]
    params_batch = [{"lam": lam} for lam in lam_batch]

    cfg = VariableBDFConfig(
        rtol=1e-5, atol=1e-8,
        h_init=1e-3, h_min=1e-7, h_max=0.25,
        max_steps=150_000, max_reject=5_000,
        newton_tol=1e-11, newton_max_iter=25,
    )

    print("=" * 60)
    print("Implicit variable-step BDF2 — batched CPU benchmark")
    print("=" * 60)

    for nw, label in [(1, "serial (1 worker)"), (None, f"parallel ({os.cpu_count()} workers)")]:
        out = solve_batch_implicit_variable(
            rhs=_demo_rhs, t_span=t_span,
            y0_batch=y0_batch, params_batch=params_batch,
            jac=_demo_jac, config=cfg,
            max_workers=nw, backend="thread",
        )
        n_ok = sum(1 for tr in out.trajectories if tr.success)
        avg_accept = np.mean([tr.n_accept for tr in out.trajectories])
        avg_reject = np.mean([tr.n_reject for tr in out.trajectories])
        avg_nit = np.mean([
            tr.n_newton_total / max(1, tr.n_accept + tr.n_reject)
            for tr in out.trajectories
        ])
        print(f"  {label:30s}  {out.wall_time:.3f}s  "
              f"success={n_ok}/{batch_size}  "
              f"accept={avg_accept:.0f}  reject={avg_reject:.0f}  "
              f"newton/step={avg_nit:.2f}")

    # --- Correctness vs exact solution ---
    out_ref = solve_batch_implicit_variable(
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
    print(f"  (rtol={cfg.rtol:.0e}, atol={cfg.atol:.0e})")


if __name__ == "__main__":
    _run_demo()
