#!/usr/bin/env python3
"""Sparse-flow features and an analytic camera estimator.

Farneback dense flow fails on this material. A Minecraft plain is mostly sky
and repeating grass, and a camera turn moves the view about 19 pixels per frame
at the resized width used here, which is far outside what the dense solver
recovers on periodic texture. Pyramidal Lucas-Kanade on a grid of points, with
six levels and a forward-backward check, recovers it. Everything below is built
on that.

``motion_track`` is the entry point. It returns one row per frame pair, holding
two kinds of column.

1. An analytic inverse dynamics: yaw, pitch and roll of the camera, the
   direction of its translation, and the residual that neither explains. These
   have no trained parameter, so they can be scored against the recorded actions
   on real Minecraft video before they are pointed at a generated clip.
   ``calibrate_fov.py`` does exactly that.

   The model is the standard rotating-camera case of multi-view geometry: two
   views of a camera that only rotates are related by the infinite homography
   ``K R K^-1``, whatever the depth of the scene. ``K`` is known here, so ``R``
   is solved for directly instead of fitting a general homography and
   decomposing it. Each step is a textbook algorithm rather than anything new:

   * pyramidal Lucas-Kanade for the correspondences, with the
     forward-backward consistency check of Kalal et al.;
   * the median change in ray azimuth and elevation as a robust initial yaw
     and pitch, which doubles as the inlier test;
   * Wahba's problem solved by the Kabsch SVD to refit ``R`` on the inliers,
     which also recovers roll;
   * for translation, the epipolar constraint with the rotation already known,
     which is linear in the translation direction and solved by its null space.

   Structure from motion and SLAM are deliberately not used, although the
   camera-control literature normally reaches for them (COLMAP for RotErr,
   DROID-SLAM for GameWorld Score's object consistency). These eval clips are
   close to pure rotation, where the closed-form solution is exact and steadier
   than an SfM pipeline on 256 frames of sky and grass.

2. A raw flow summary: a robust global shift, an affine gradient, and the same
   shift split between the upper and lower half of the view. The split separates
   a camera turn from a sideways step. A yaw turn moves every pixel the same
   amount whatever its depth. A sideways step moves near pixels more than far
   ones, and in a first-person Minecraft view the near pixels are the ground
   along the bottom of the frame. The learned key model reads these.

Timing note. The renderer is one frame behind the action log: the image at
ground-truth index ``k`` shows the pose that the action at index ``k`` produced,
so the motion between images ``k`` and ``k+1`` is the action recorded at
``k+1``. ``commanded_camera`` and ``commanded_keys`` apply that shift.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from af_data import focal_px, view_mask  # noqa: E402

FLOW_WIDTH = 320  # every clip is resized to this before tracking
GRID_STEP = 5
LK_PARAMS = dict(
    winSize=(21, 21),
    maxLevel=6,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
)
FB_TOLERANCE = 1.0  # forward-backward reprojection error, pixels

FEATURE_NAMES = [
    "tx", "ty",           # robust global shift, pixels of the resized frame
    "dudx", "dudy", "dvdx", "dvdy",
    "div", "curl",
    "mag",                # median displacement magnitude
    "tx_top", "ty_top",   # upper half: distant geometry and sky
    "tx_bot", "ty_bot",   # lower half: the ground, which carries the parallax
    "tx_bot_minus_top",   # the sideways-step signature
    "ty_bot_minus_top",
    "coverage",           # fraction of grid points that survived the check
]
N_FEATURES = len(FEATURE_NAMES)


def _resize_stack(stack: np.ndarray, width: int = FLOW_WIDTH) -> np.ndarray:
    if stack.shape[0] == 0:
        return stack
    h, w = stack.shape[1:3]
    height = int(round(h * width / w))
    return np.stack([cv2.resize(f, (width, height)) for f in stack])


def _grid(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    ys, xs = np.mgrid[GRID_STEP : h - GRID_STEP : GRID_STEP,
                      GRID_STEP : w - GRID_STEP : GRID_STEP].reshape(2, -1)
    keep = mask[ys, xs]
    return np.stack([xs[keep], ys[keep]], axis=1).astype(np.float32)[:, None, :]


def _track(a: np.ndarray, b: np.ndarray, pts: np.ndarray):
    fwd, st, _ = cv2.calcOpticalFlowPyrLK(a, b, pts, None, **LK_PARAMS)
    if fwd is None:
        return None, None
    back, st2, _ = cv2.calcOpticalFlowPyrLK(b, a, fwd, None, **LK_PARAMS)
    if back is None:
        return None, None
    fb = np.linalg.norm((back - pts).reshape(-1, 2), axis=1)
    ok = (st.ravel() == 1) & (st2.ravel() == 1) & (fb < FB_TOLERANCE)
    return (fwd - pts).reshape(-1, 2), ok


def _robust_shift(d: np.ndarray) -> Tuple[float, float]:
    """Median, then re-median over the points within 2 px of it."""
    if len(d) < 20:
        return 0.0, 0.0
    med = np.median(d, axis=0)
    inl = np.linalg.norm(d - med, axis=1) < 2.0
    if inl.sum() >= 20:
        med = np.median(d[inl], axis=0)
    return float(med[0]), float(med[1])


def resized_height(stack_height: int, stack_width: int) -> int:
    return int(round(stack_height * FLOW_WIDTH / stack_width))


# --- rotation estimation ----------------------------------------------------

def _rays(pts_xy: np.ndarray, f: float, cx: float, cy: float) -> np.ndarray:
    """Pixel coordinates to unit viewing rays."""
    d = np.stack([(pts_xy[:, 0] - cx) / f, (pts_xy[:, 1] - cy) / f,
                  np.ones(len(pts_xy))], axis=1)
    return d / np.linalg.norm(d, axis=1, keepdims=True)


def _kabsch(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Rotation R minimising ||R a - b|| for unit vectors a and b.

    Wahba's problem, in the Kabsch closed form: SVD of the correlation matrix,
    with the determinant correction that keeps the result a rotation rather than
    a reflection.
    """
    h = a.T @ b
    u, _, vt = np.linalg.svd(h)
    d = np.sign(np.linalg.det(vt.T @ u.T))
    return vt.T @ np.diag([1.0, 1.0, d]) @ u.T


INLIER_RAD = 0.02  # about 1.1 degrees


def estimate_rotation(p0: np.ndarray, p1: np.ndarray, f: float,
                      cx: float, cy: float) -> Tuple[float, float, float]:
    """Yaw, pitch and roll of the rotation that best explains a correspondence set.

    Two standard pieces: a robust median initialisation on bearing angles, then
    Wahba's problem solved by the Kabsch SVD on the inliers.

    A plain median of the pixel shift is biased. Under a yaw of ``a`` a pixel at
    horizontal offset ``x`` moves about ``f*a*(1 + x^2/f^2)``, so the edges of
    the frame move further than the centre and the median lands above the true
    rate. Working on viewing rays removes that bias: a yaw turn adds the same
    amount to the *azimuth* of every ray, wherever the ray points, so the median
    per-point azimuth change is an unbiased estimate. Elevation gives pitch the
    same way. A least-squares rotation is then refit on the points those two
    medians agree with, which also recovers roll and rejects the points that
    translation rather than rotation explains.

    Returns radians. Positive yaw matches a positive ``camera[0]`` command.
    """
    if len(p0) < 20:
        return 0.0, 0.0, 0.0
    a = _rays(p0, f, cx, cy)
    b = _rays(p1, f, cx, cy)
    az = np.arctan2(b[:, 0], b[:, 2]) - np.arctan2(a[:, 0], a[:, 2])
    az = (az + np.pi) % (2 * np.pi) - np.pi
    el = np.arcsin(np.clip(b[:, 1], -1, 1)) - np.arcsin(np.clip(a[:, 1], -1, 1))
    yaw, pitch = float(np.median(az)), float(np.median(el))

    keep = (np.abs(az - yaw) < INLIER_RAD) & (np.abs(el - pitch) < INLIER_RAD)
    if keep.sum() >= 20:
        r = _kabsch(a[keep], b[keep])
        yaw = float(np.arctan2(r[0, 2], r[2, 2]))
        # Positive pitch means the same thing here as in the median above and in
        # the recorded ``camera[1]``: the camera looks down. That is the opposite
        # of the sign the raw matrix carries, hence the flip. Checked both ways:
        # the median and this refit correlate at -0.99 without it, and the
        # commanded pitch agrees with the median.
        pitch = float(np.arcsin(np.clip(r[1, 2], -1.0, 1.0)))
        roll = float(np.arctan2(r[1, 0], r[1, 1]))
    else:
        roll = 0.0
    return yaw, pitch, roll


def estimate_translation(p0: np.ndarray, p1: np.ndarray, yaw: float, pitch: float,
                         roll: float, f: float, cx: float, cy: float):
    """Direction of the camera's own translation, once the rotation is removed.

    This is the epipolar constraint with the rotation already known, which makes
    it linear in the translation direction and solvable by a null space rather
    than by the five-point algorithm.

    Depth is unknown, so only the direction is recoverable, not the distance.
    Rotate every ray of the first frame by the estimated rotation. What is left
    between it and the matching ray of the second frame is caused by the step
    the player took. Each residual must lie in the plane through its ray and the
    translation direction ``t``, which gives the linear constraint
    ``t . (b x a') = 0``. The smallest singular vector of the stacked
    constraints is ``t``, and the sign is the one that puts the points in front
    of the camera.

    Returns ``(t, strength)``: a unit 3-vector in camera coordinates, where +z
    is forward and +x is to the right, and the median residual angle in radians,
    which says how much translation there was to measure.
    """
    if len(p0) < 20:
        return np.zeros(3), 0.0
    a = _rays(p0, f, cx, cy)
    b = _rays(p1, f, cx, cy)
    r = _rotation_matrix(yaw, pitch, roll)
    a_rot = a @ r.T
    a_rot /= np.linalg.norm(a_rot, axis=1, keepdims=True)

    resid = np.arccos(np.clip((a_rot * b).sum(1), -1, 1))
    strength = float(np.median(resid))
    keep = resid > np.deg2rad(0.05)
    if keep.sum() < 20:
        return np.zeros(3), strength

    m = np.cross(b[keep], a_rot[keep])
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    m = m / np.maximum(norms, 1e-12)
    _, _, vt = np.linalg.svd(m, full_matrices=False)
    t = vt[-1]
    # Sign: a point that the camera moves towards must flow outwards from the
    # direction of travel. Pick the sign that agrees with the majority.
    proj = ((b[keep] - a_rot[keep]) *
            (t - (a_rot[keep] @ t)[:, None] * a_rot[keep])).sum(1)
    if proj.sum() > 0:
        t = -t
    return t / (np.linalg.norm(t) + 1e-12), strength


def _rotation_matrix(yaw: float, pitch: float, roll: float) -> np.ndarray:
    """Inverse of the extraction in ``estimate_rotation``, same sign convention.

    ``pitch`` is negated on the way in because positive pitch is defined here as
    looking down, which is the opposite of the sign the raw matrix carries.
    """
    pitch = -pitch
    cy, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cr, sr = np.cos(roll), np.sin(roll)
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    rx = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]])
    rz = np.array([[cr, -sr, 0], [sr, cr, 0], [0, 0, 1]])
    return ry @ rx @ rz


MOTION_COLUMNS = (
    ["yaw", "pitch", "roll", "inlier_frac", "tx", "ty", "tz", "resid"]
    + FEATURE_NAMES
)
YAW, PITCH, ROLL, INLIER, TX, TY, TZ, RESID = range(8)


def motion_track(stack: np.ndarray) -> np.ndarray:
    """Per-frame-pair camera motion and flow summary. Columns: ``MOTION_COLUMNS``.

    The first eight columns are the analytic inverse dynamics: a rotation, a
    translation direction and the residual that is left over. They use only the
    known intrinsics and have no trained parameters, so they can be scored on
    ground-truth video before they are applied to a generated clip. The rest are
    the raw flow summary, which the learned key model reads.
    """
    stack = _resize_stack(stack)
    n = stack.shape[0]
    ncol = len(MOTION_COLUMNS)
    if n < 2:
        return np.zeros((0, ncol), dtype=np.float32)
    h, w = stack.shape[1:3]
    mask = view_mask(h, w)
    pts = _grid(mask)
    if len(pts) < 40:
        return np.zeros((n - 1, ncol), dtype=np.float32)
    f = focal_px(h)
    cx, cy = w / 2.0, h / 2.0
    base = pts.reshape(-1, 2)
    nx = (base[:, 0] - cx) / cx
    ny = (base[:, 1] - cy) / cy
    top = base[:, 1] < h * 0.45

    out = np.zeros((n - 1, ncol), dtype=np.float32)
    for i in range(n - 1):
        d, ok = _track(stack[i], stack[i + 1], pts)
        if d is None or ok.sum() < 40:
            continue
        p0 = base[ok]
        p1 = p0 + d[ok]
        yaw, pitch, roll = estimate_rotation(p0, p1, f, cx, cy)
        t, strength = estimate_translation(p0, p1, yaw, pitch, roll, f, cx, cy)

        dd = d[ok]
        tx, ty = _robust_shift(dd)
        X = np.stack([nx[ok], ny[ok], np.ones(ok.sum())], axis=1)
        su, *_ = np.linalg.lstsq(X, dd[:, 0], rcond=None)
        sv, *_ = np.linalg.lstsq(X, dd[:, 1], rcond=None)
        okt, okb = top[ok], ~top[ok]
        tx_t, ty_t = _robust_shift(dd[okt]) if okt.sum() >= 20 else (tx, ty)
        tx_b, ty_b = _robust_shift(dd[okb]) if okb.sum() >= 20 else (tx, ty)
        out[i] = [
            yaw, pitch, roll, ok.mean(), t[0], t[1], t[2], strength,
            tx, ty, su[0], su[1], sv[0], sv[1], su[0] + sv[1], sv[0] - su[1],
            float(np.median(np.linalg.norm(dd, axis=1))),
            tx_t, ty_t, tx_b, ty_b, tx_b - tx_t, ty_b - ty_t, float(ok.mean()),
        ]
    return out


def camera_track(stack: np.ndarray) -> np.ndarray:
    """Rotation columns only: yaw, pitch, roll, inlier fraction."""
    return motion_track(stack)[:, :4]


def commanded_camera(actions: list, gen0_gt_index: int, count: int) -> np.ndarray:
    """Commanded ``(yaw, pitch)`` per generated frame pair, radians per frame."""
    out = np.zeros((count, 2), dtype=np.float32)
    for i in range(count):
        k = gen0_gt_index + i
        if 0 <= k < len(actions):
            cam = actions[k]["action"]["camera"]
            out[i] = [float(cam[0]), float(cam[1])]
    return out


KEY_NAMES = ["forward", "back", "left", "right", "jump", "sneak", "sprint",
             "attack", "use", "place_block", "mine"]


def commanded_keys(actions: list, gen0_gt_index: int, count: int, keys=None) -> dict:
    keys = keys or KEY_NAMES
    out = {k: np.zeros(count, dtype=bool) for k in keys}
    for i in range(count):
        k = gen0_gt_index + i
        if 0 <= k < len(actions):
            act = actions[k]["action"]
            for name in keys:
                out[name][i] = bool(act.get(name, False))
    return out


def commanded_body_velocity(actions: list, gen0_gt_index: int, count: int) -> np.ndarray:
    """Recorded displacement per frame pair, rotated into the player's frame.

    Returns ``(count, 2)`` of (right, forward) blocks per frame.

    The axis convention is not guessed. It was fitted to 843 frames of
    ``translationEval`` on which exactly one movement key was held, by trying the
    candidate conventions and keeping the one where ``forward`` moves forward and
    ``left`` moves left. The winner is

        forward = -(dx sin yaw + dz cos yaw)
        right   =   dx cos yaw - dz sin yaw

    which separates cleanly: forward +0.211, back -0.208, left -0.209 and right
    +0.208 blocks per frame, with the off-axis component under 0.02. Note that
    the two axes do not come from one rotation matrix of the usual handedness,
    because Minecraft's x-east, z-south, y-up frame is left-handed.

    Nothing in the reports reads this yet. It is here because it is the only
    ground-truth translation signal available, and the flow-based translation
    direction is too weak on an open plain to score against it.
    """
    out = np.zeros((count, 2), dtype=np.float32)
    for i in range(count):
        k = gen0_gt_index + i
        if not (0 <= k < len(actions) - 1):
            continue
        a, b = actions[k], actions[k + 1]
        dx, dz = b["x"] - a["x"], b["z"] - a["z"]
        yaw = float(a.get("yaw", 0.0))
        s, c = np.sin(yaw), np.cos(yaw)
        out[i] = [dx * c - dz * s, -(dx * s + dz * c)]
    return out
