# MIT License
#
# Copyright (c) 2026 THALES

"""Reference decoders of the two final dense-vision candidates (frozen 2026-11-02).

Both decoders read only the network output, the obstacle mask and the
endpoints; neither reads the terrain, a reference path or a reference cost.
Graph: 8 neighbours, diagonals need only a free destination. Every search uses
an indexed binary heap keyed by (priority, tie, cell), compiled with Numba.

* Seg — A* on the raw path-class logit ``z`` (taken before any activation):
  step ``l(u,v) * (1 + w * (max z - z_v))``, heuristic ``eta * octile(v, g)``.
* Reg — Bellman descent on the predicted cost-to-go ``V`` (training scale
  ``REG_TRAIN_SCALE``): among free neighbours with ``V(v) < V(u)`` choose the
  minimiser of ``V(v) + kappa * 30 * l(u,v) / REG_TRAIN_SCALE``. On a local
  minimum, a geometric Dijkstra grows from the goal over reversed edges until it
  settles any visited descent cell; the route is the descent prefix up to that
  cell followed by the Dijkstra branch to the goal.

Parameters were fixed on the calibration split (256, 1 024 scenes) and are
documented in ``docs/benchmark/final-candidates.md``. Numba compilation happens
on first call and belongs to cold-start time, not to request latency.
"""

from __future__ import annotations

import math

import numpy as np
from numba import njit

ROOT2 = math.sqrt(2.0)
SEG_STEP_WEIGHT = 0.25
SEG_HEURISTIC_WEIGHT = 1.5
REG_KAPPA = 8.0
REG_TRAIN_SCALE = 897803.25
STATUS = ("ok", "no_path", "budget_exhausted")


@njit(cache=True, inline="always")
def _less(a1, a2, an, b1, b2, bn):
    return a1 < b1 or (a1 == b1 and (a2 < b2 or (a2 == b2 and an < bn)))


@njit(cache=True)
def _sift_up(k1, k2, kn, pos, i):
    a1, a2, an = k1[i], k2[i], kn[i]
    while i > 0:
        p = (i - 1) >> 1
        if not _less(a1, a2, an, k1[p], k2[p], kn[p]):
            break
        k1[i], k2[i], kn[i] = k1[p], k2[p], kn[p]
        pos[kn[i]] = i
        i = p
    k1[i], k2[i], kn[i] = a1, a2, an
    pos[an] = i


@njit(cache=True)
def _sift_down(k1, k2, kn, pos, i, size):
    a1, a2, an = k1[i], k2[i], kn[i]
    while True:
        c = 2 * i + 1
        if c >= size:
            break
        r = c + 1
        if r < size and _less(k1[r], k2[r], kn[r], k1[c], k2[c], kn[c]):
            c = r
        if not _less(k1[c], k2[c], kn[c], a1, a2, an):
            break
        k1[i], k2[i], kn[i] = k1[c], k2[c], kn[c]
        pos[kn[i]] = i
        i = c
    k1[i], k2[i], kn[i] = a1, a2, an
    pos[an] = i


@njit(cache=True)
def _push(k1, k2, kn, pos, size, node, f, t):
    i = pos[node]
    if i < 0:
        i = size
        size += 1
    k1[i], k2[i], kn[i] = f, t, node
    pos[node] = i
    _sift_up(k1, k2, kn, pos, i)
    return size


@njit(cache=True)
def _pop(k1, k2, kn, pos, size):
    node = kn[0]
    pos[node] = -2
    size -= 1
    if size > 0:
        k1[0], k2[0], kn[0] = k1[size], k2[size], kn[size]
        pos[kn[0]] = 0
        _sift_down(k1, k2, kn, pos, 0, size)
    return node, size


@njit(cache=True, inline="always")
def _octile(r, c, gr, gc):
    dr, dc = abs(r - gr), abs(c - gc)
    return max(dr, dc) + (ROOT2 - 1.0) * min(dr, dc)


@njit(cache=True)
def _trace(parent, start, goal, cols):
    length, cur = 1, goal
    while cur != start:
        cur = parent[cur]
        if cur < 0:
            return np.empty((0, 2), np.int32)
        length += 1
    out = np.empty((length, 2), np.int32)
    cur = goal
    for i in range(length - 1, -1, -1):
        out[i, 0], out[i, 1] = cur // cols, cur % cols
        if i:
            cur = parent[cur]
    return out


@njit(cache=True)
def build_input(elevation, obstacles, sr, sc, gr, gc, low, high):
    """Ordinary NCHW float32 input: calibrated height, markers (-1/0/1), obstacles."""
    rows, cols = elevation.shape
    out = np.empty((1, 3, rows, cols), np.float32)
    span = high - low
    for r in range(rows):
        for c in range(cols):
            out[0, 0, r, c] = np.float32((elevation[r, c] - low) / span)
            out[0, 1, r, c] = -1.0
            out[0, 2, r, c] = 1.0 if obstacles[r, c] else 0.0
    out[0, 1, sr, sc] = 0.0
    out[0, 1, gr, gc] = 1.0
    return out


@njit(cache=True)
def _logit_astar(logit, blocked, sr, sc, gr, gc, weight, eta, budget):
    rows, cols = logit.shape
    n = rows * cols
    start, goal = sr * cols + sc, gr * cols + gc
    high = np.max(logit)
    dist = np.full(n, np.inf)
    parent = np.full(n, -1, np.int32)
    pos = np.full(n, -1, np.int32)
    k1 = np.empty(n, np.float64)
    k2 = np.empty(n, np.float64)
    kn = np.empty(n, np.int32)
    dist[start] = 0.0
    size = _push(k1, k2, kn, pos, 0, start, eta * _octile(sr, sc, gr, gc), 0.0)
    count = 0
    while size:
        cost = k2[0]
        node, size = _pop(k1, k2, kn, pos, size)
        if count >= budget:
            return 2, np.empty((0, 2), np.int32), count
        count += 1
        if node == goal:
            return 0, _trace(parent, start, goal, cols), count
        row, col = node // cols, node % cols
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                nr, nc = row + dr, col + dc
                if nr < 0 or nr >= rows or nc < 0 or nc >= cols or blocked[nr, nc]:
                    continue
                other = nr * cols + nc
                if pos[other] == -2:
                    continue
                new = cost + (ROOT2 if dr and dc else 1.0) * (1.0 + weight * (high - logit[nr, nc]))
                if new < dist[other]:
                    dist[other] = new
                    parent[other] = node
                    size = _push(k1, k2, kn, pos, size, other, new + eta * _octile(nr, nc, gr, gc), new)
    return 1, np.empty((0, 2), np.int32), count


@njit(cache=True)
def _bellman_descent(field, blocked, sr, sc, gr, gc, kappa_step, budget):
    rows, cols = field.shape
    n = rows * cols
    goal = gr * cols + gc
    prefix = np.empty(n, np.int32)
    index = np.full(n, -1, np.int32)
    row, col, length = sr, sc, 1
    prefix[0] = sr * cols + sc
    index[prefix[0]] = 0
    if prefix[0] == goal:
        return 0, _trace(np.full(n, -1, np.int32), goal, goal, cols), 0, 0, False
    steps = 0
    while steps < n:
        current = field[row, col]
        best, br, bc = np.inf, -1, -1
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                nr, nc = row + dr, col + dc
                if nr < 0 or nr >= rows or nc < 0 or nc >= cols or blocked[nr, nc]:
                    continue
                v = 0.0 if nr == gr and nc == gc else field[nr, nc]
                if v >= current:
                    continue
                score = v + kappa_step * (ROOT2 if dr and dc else 1.0)
                if score < best:
                    best, br, bc = score, nr, nc
        steps += 1
        if br < 0:
            break
        row, col = br, bc
        cell = row * cols + col
        prefix[length] = cell
        index[cell] = length
        length += 1
        if cell == goal:
            out = np.empty((length, 2), np.int32)
            for i in range(length):
                out[i, 0], out[i, 1] = prefix[i] // cols, prefix[i] % cols
            return 0, out, steps, 0, False
    dist = np.full(n, np.inf)
    toward = np.full(n, -1, np.int32)
    pos = np.full(n, -1, np.int32)
    k1 = np.empty(n, np.float64)
    k2 = np.zeros(n, np.float64)
    kn = np.empty(n, np.int32)
    dist[goal] = 0.0
    size = _push(k1, k2, kn, pos, 0, goal, 0.0, 0.0)
    count = 0
    while size:
        cost = k1[0]
        node, size = _pop(k1, k2, kn, pos, size)
        if count >= budget:
            return 2, np.empty((0, 2), np.int32), steps, count, True
        count += 1
        if index[node] >= 0:
            cut = index[node]
            tail, cur = 0, node
            while cur != goal:
                cur = toward[cur]
                tail += 1
            out = np.empty((cut + 1 + tail, 2), np.int32)
            for i in range(cut + 1):
                out[i, 0], out[i, 1] = prefix[i] // cols, prefix[i] % cols
            cur, i = node, cut
            while cur != goal:
                cur = toward[cur]
                i += 1
                out[i, 0], out[i, 1] = cur // cols, cur % cols
            return 0, out, steps, count, True
        row, col = node // cols, node % cols
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                pr, pc = row + dr, col + dc
                if pr < 0 or pr >= rows or pc < 0 or pc >= cols or blocked[pr, pc]:
                    continue
                other = pr * cols + pc
                if pos[other] == -2:
                    continue
                new = cost + (ROOT2 if dr and dc else 1.0)
                if new < dist[other]:
                    dist[other] = new
                    toward[other] = node
                    size = _push(k1, k2, kn, pos, size, other, new, 0.0)
    return 1, np.empty((0, 2), np.int32), steps, count, True


def _checked(field, obstacles, start, goal):
    grid = np.ascontiguousarray(field, dtype=np.float64)
    blocked = np.ascontiguousarray(obstacles, dtype=np.bool_)
    if grid.ndim != 2 or grid.shape != blocked.shape or not grid.size or not np.isfinite(grid).all():
        raise ValueError("expected one finite 2D field matching the obstacle mask")
    points = []
    for point in (start, goal):
        r, c = (int(x) for x in point)
        if not (0 <= r < grid.shape[0] and 0 <= c < grid.shape[1]) or blocked[r, c]:
            raise ValueError("endpoint outside the grid or on an obstacle")
        points.append((r, c))
    return grid, blocked, points[0], points[1]


def seg_logit_astar(logit, obstacles, start, goal, *, weight=SEG_STEP_WEIGHT,
                    eta=SEG_HEURISTIC_WEIGHT, max_expansions=None):
    """Final Seg decoder: A* on the raw path-class logit. Returns a result dict."""
    z, blocked, (sr, sc), (gr, gc) = _checked(logit, obstacles, start, goal)
    if weight < 0 or eta <= 0:
        raise ValueError("weight must be >= 0 and eta > 0")
    budget = z.size if max_expansions is None else int(max_expansions)
    status, path, work = _logit_astar(z, blocked, sr, sc, gr, gc, float(weight), float(eta), budget)
    return {"status": STATUS[status], "path": path, "expansions": int(work), "fallback": False}


def reg_bellman_descent(value, obstacles, start, goal, *, kappa=REG_KAPPA,
                        train_scale=REG_TRAIN_SCALE, max_expansions=None):
    """Final Reg decoder: Bellman descent + goal-rooted geometric Dijkstra fallback.

    ``kappa = 0`` gives the plain descent ``argmin V(v)``.
    """
    v, blocked, (sr, sc), (gr, gc) = _checked(value, obstacles, start, goal)
    if (v < 0).any() or kappa < 0 or train_scale <= 0:
        raise ValueError("expected a nonnegative field, kappa >= 0 and a positive training scale")
    budget = v.size if max_expansions is None else int(max_expansions)
    status, path, steps, work, fallback = _bellman_descent(
        v, blocked, sr, sc, gr, gc, float(kappa) * 30.0 / float(train_scale), budget)
    return {"status": STATUS[status], "path": path, "descent_steps": int(steps),
            "fallback_expansions": int(work), "fallback": bool(fallback)}
