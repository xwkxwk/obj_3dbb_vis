#! /usr/bin/env python3
"""Headless 3D scene renderer — reads BoxerNet CSV, saves static 3D view images.

Usage:
  python render_3d_scene.py --input <seq_name>                    # all frames piled
  python render_3d_scene.py --input <seq_name> --per_frame        # one image per frame
  python render_3d_scene.py --input <seq_name> --fused            # prefer fused CSV
  python render_3d_scene.py --input <seq_name> --max_boxes 2000   # subsample if too many
"""

import argparse
import os
import sys
import time
import warnings

import numpy as np

# Headless backends — must be set before importing matplotlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.demo_utils import EVAL_PATH, DEFAULT_SEQ  # noqa: E402
from utils.file_io import read_obb_csv  # noqa: E402

EDGES = [
    (0, 1), (2, 3), (4, 5), (6, 7),
    (0, 2), (1, 3), (4, 6), (5, 7),
    (0, 4), (1, 5), (2, 6), (3, 7),
]

TAB20 = np.array([
    [0.122, 0.467, 0.706], [0.682, 0.780, 0.910],
    [1.000, 0.498, 0.055], [1.000, 0.733, 0.471],
    [0.173, 0.627, 0.173], [0.596, 0.875, 0.541],
    [0.839, 0.153, 0.157], [1.000, 0.596, 0.588],
    [0.580, 0.404, 0.741], [0.773, 0.690, 0.835],
    [0.549, 0.337, 0.294], [0.769, 0.612, 0.580],
    [0.890, 0.467, 0.761], [0.969, 0.714, 0.824],
    [0.498, 0.498, 0.498], [0.780, 0.780, 0.780],
    [0.737, 0.741, 0.133], [0.859, 0.859, 0.553],
    [0.090, 0.745, 0.812], [0.620, 0.855, 0.898],
], dtype=np.float32)


# ---------------------------------------------------------------------------
#  Backend detection
# ---------------------------------------------------------------------------

_HAS_O3D = False
_O3D_VERSION = None

try:
    import open3d as o3d
    _O3D_VERSION = o3d.__version__
    # Quick smoke test: can we create a headless visualizer?
    try:
        vis = o3d.visualization.Visualizer()
        vis.create_window(width=4, height=4, visible=False)
        vis.destroy_window()
        _HAS_O3D = True
    except Exception:
        warnings.warn("Open3D found but headless Visualizer failed, falling back to matplotlib")
except ImportError:
    pass


def _make_lineset(corners_world, colors_rgb, probs):
    """Build an Open3D LineSet from corners + per-box colors."""
    corners = np.asarray(corners_world, dtype=np.float32)
    N = corners.shape[0]

    # Build all edge endpoints: (N*12*2, 3)
    segs = np.empty((N * 12, 2, 3), dtype=np.float32)
    for e, (i, j) in enumerate(EDGES):
        segs[e * N:(e + 1) * N, 0] = corners[:, i]
        segs[e * N:(e + 1) * N, 1] = corners[:, j]
    points = segs.reshape(-1, 3)
    lines = np.arange(N * 24).reshape(-1, 2).astype(np.int32)

    alpha = np.clip(np.asarray(probs), 0.15, 1.0).reshape(-1, 1)  # (N, 1)
    alpha = np.repeat(alpha, 12, axis=0)  # (N*12, 1) — match edge count
    edge_colors = np.repeat(colors_rgb[:N], 12, axis=0) * alpha

    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(points)
    ls.lines = o3d.utility.Vector2iVector(lines)
    ls.colors = o3d.utility.Vector3dVector(np.clip(edge_colors, 0, 1))
    return ls


def _o3d_snapshot(vis, output_path):
    """Capture rendered image from headless visualizer."""
    vis.poll_events()
    vis.update_renderer()
    vis.capture_screen_image(output_path, do_render=True)


def _o3d_set_view(vis, ctr, distance, azimuth_deg, elevation_deg):
    """Set the Open3D Visualizer camera to look at `ctr` from spherical coords."""
    az = np.radians(azimuth_deg)
    el = np.radians(elevation_deg)
    eye = np.array([
        ctr[0] + distance * np.cos(el) * np.sin(az),
        ctr[1] + distance * np.cos(el) * np.cos(az),
        ctr[2] + distance * np.sin(el),
    ])
    up = np.array([0, 0, 1])
    vc = vis.get_view_control()
    vc.set_front((ctr - eye) / np.linalg.norm(ctr - eye))
    vc.set_lookat(ctr)
    vc.set_up(up)
    vc.set_zoom(1.0)


# ---------------------------------------------------------------------------
#  Matplotlib fallback — batched line drawing (fast for moderate N)
# ---------------------------------------------------------------------------

def _mpl_render_view(ax, corners_world, center, distance,
                     azimuth_deg, elevation_deg):
    """Render all boxes into a single matplotlib 3D Axes with one plot call per color."""
    corners = np.asarray(corners_world, dtype=np.float32)
    N = corners.shape[0]

    # Build NaN-separated segments grouped by color to reduce draw calls
    # One plot3D call for ALL edges — color is uniform but lightweight
    segs_all = []
    for i, j in EDGES:
        seg = np.empty((N * 3, 3), dtype=np.float32)
        seg[0::3] = corners[:, i]
        seg[1::3] = corners[:, j]
        seg[2::3] = np.nan
        segs_all.append(seg)
    all_pts = np.concatenate(segs_all, axis=0)

    half = distance
    ax.set_xlim(center[0] - half, center[0] + half)
    ax.set_ylim(center[1] - half, center[1] + half)
    ax.set_zlim(center[2] - half, center[2] + half)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")

    ax.plot3D(all_pts[:, 0], all_pts[:, 1], all_pts[:, 2],
              color=(0.1, 0.1, 0.1), linewidth=0.3, alpha=0.5)

    if elevation_deg >= 89:
        ax.view_init(elev=90, azim=0)
    else:
        ax.view_init(elev=elevation_deg, azim=azimuth_deg)


def _mpl_composite(all_corners_list, all_colors_list, all_probs_list,
                   output_dir, write_name, center, distance, prefix=""):
    """Render 4-view composite using matplotlib (fallback)."""
    views = [
        (-60, 25),
        (0, 90),
        (90, 0),
        (0, 0),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 13),
                             subplot_kw={"projection": "3d"})

    for ax, (az, el) in zip(axes.flat, views):
        for corners_w, *_ in zip(all_corners_list, all_colors_list, all_probs_list):
            _mpl_render_view(ax, corners_w, center, distance, az, el)

    plt.suptitle("BoxerNet 3D Scene Reconstruction", fontsize=13, fontweight="bold")
    plt.tight_layout()
    out = os.path.join(output_dir, f"{write_name}_3d_scene{prefix}.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ---------------------------------------------------------------------------
#  Core rendering logic
# ---------------------------------------------------------------------------

def _prepare_geometry(timed_obbs, max_boxes):
    """Extract corners, colors, probs from timed_obbs. Returns lists (one per frame)."""
    corners_list, colors_list, probs_list = [], [], []
    total = 0

    for obbs in timed_obbs.values():
        if len(obbs) == 0:
            continue
        M = len(obbs)
        cw = obbs.bb3corners_world.numpy()  # (M, 8, 3)
        p = obbs.prob.squeeze(-1).numpy()  # (M,)
        clr = np.array([TAB20[i % len(TAB20)] for i in range(M)], dtype=np.float32)
        corners_list.append(cw)
        colors_list.append(clr)
        probs_list.append(p)
        total += M

    # Subsample if too many
    if total > max_boxes:
        print(f"==> Subsampling {total} -> {max_boxes} boxes (adjust with --max_boxes)")
        rng = np.random.default_rng(42)
        all_idx = np.arange(total)
        keep = set(rng.choice(all_idx, size=max_boxes, replace=False))

        new_c, new_clr, new_p = [], [], []
        offset = 0
        for cw, clr, p in zip(corners_list, colors_list, probs_list):
            M = cw.shape[0]
            idx = [k - offset for k in keep if offset <= k < offset + M]
            if idx:
                new_c.append(cw[idx])
                new_clr.append(clr[idx])
                new_p.append(p[idx])
            offset += M
        corners_list, colors_list, probs_list = new_c, new_clr, new_p
        total = max_boxes

    return corners_list, colors_list, probs_list, total


def _compute_scene_bounds(corners_list):
    all_pts = np.concatenate([c.reshape(-1, 3) for c in corners_list], axis=0)
    ctr = (all_pts.min(axis=0) + all_pts.max(axis=0)) / 2.0
    extent = (all_pts.max(axis=0) - all_pts.min(axis=0)).max()
    dist = max(extent * 1.5, 3.0)
    return ctr, dist


def render_all_frames(timed_obbs, output_dir, write_name, max_boxes):
    print("==> Preparing geometry ...")
    t0 = time.perf_counter()
    corners_list, colors_list, probs_list, total = _prepare_geometry(timed_obbs, max_boxes)
    ctr, dist = _compute_scene_bounds(corners_list)
    print(f"  {total} boxes, center={ctr}, extent={dist:.1f}m "
          f"({(time.perf_counter() - t0):.1f}s)")

    if _HAS_O3D:
        print("==> Rendering with Open3D headless ...")
        _render_all_o3d(corners_list, colors_list, probs_list, output_dir,
                        write_name, ctr, dist)
    else:
        print("==> Rendering with matplotlib (fallback) ...")
        _mpl_composite(timed_obbs, corners_list, colors_list, probs_list,
                       output_dir, write_name, ctr, dist)


def _render_all_o3d(corners_list, colors_list, probs_list, output_dir,
                    write_name, center, distance):
    """Render all boxes piled using Open3D headless visualizer."""
    all_corners = np.concatenate(corners_list, axis=0) if corners_list else np.zeros((0, 8, 3))
    all_colors = np.concatenate(colors_list, axis=0) if colors_list else np.zeros((0, 3))
    all_probs = np.concatenate(probs_list, axis=0) if probs_list else np.zeros(0)

    ls = _make_lineset(all_corners, all_colors, all_probs)

    views = [
        ("perspective", -60, 25, f"{write_name}_3d_perspective.png"),
        ("top_down", 0, 90, f"{write_name}_3d_topdown.png"),
        ("front", 90, 0, f"{write_name}_3d_front.png"),
        ("side", 0, 0, f"{write_name}_3d_side.png"),
    ]

    W, H = 1600, 1200
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=W, height=H, visible=False)
    vis.add_geometry(ls, reset_bounding_box=True)

    # Coordinate frame
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=distance * 0.15, origin=[0, 0, 0]
    )
    vis.add_geometry(frame, reset_bounding_box=False)

    for label, az, el, fname in views:
        out = os.path.join(output_dir, fname)
        _o3d_set_view(vis, center, distance, az, el)
        _o3d_snapshot(vis, out)

    vis.destroy_window()
    print(f"==> All views saved to {output_dir}/")


def render_per_frame(timed_obbs, output_dir, write_name):
    frame_dir = os.path.join(output_dir, f"{write_name}_3d_frames")
    os.makedirs(frame_dir, exist_ok=True)

    ts_list = sorted(timed_obbs.keys())
    if not ts_list:
        print("No frames to render.")
        return

    # Compute global bounds for consistent views
    all_corners = []
    for obbs in timed_obbs.values():
        if len(obbs):
            all_corners.append(obbs.bb3corners_world.numpy().reshape(-1, 3))
    all_pts = np.concatenate(all_corners, axis=0)
    ctr = (all_pts.min(axis=0) + all_pts.max(axis=0)) / 2.0
    dist = max((all_pts.max(axis=0) - all_pts.min(axis=0)).max() * 1.5, 3.0)

    views = [(-60, 25), (0, 90), (90, 0), (0, 0)]

    if _HAS_O3D:
        print(f"==> Rendering {len(ts_list)} frames with Open3D headless ...")
        W, H = 1200, 900
        vis = o3d.visualization.Visualizer()
        vis.create_window(width=W, height=H, visible=False)
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=dist * 0.15, origin=[0, 0, 0]
        )

        saved = 0
        for idx, ts in enumerate(ts_list):
            obbs = timed_obbs[ts]
            if len(obbs) == 0:
                continue
            corners_w = obbs.bb3corners_world.numpy()
            M = corners_w.shape[0]
            probs = obbs.prob.squeeze(-1).numpy()
            colors = np.array([TAB20[i % len(TAB20)] for i in range(M)], dtype=np.float32)
            ls = _make_lineset(corners_w, colors, probs)

            vis.clear_geometries()
            vis.add_geometry(ls, reset_bounding_box=True)
            vis.add_geometry(frame, reset_bounding_box=False)

            view_imgs = []
            for az, el in views:
                _o3d_set_view(vis, ctr, dist, az, el)
                _o3d_snapshot(vis, "")  # capture to buffer
                img = np.asarray(vis.capture_screen_float_buffer(do_render=True)) * 255
                view_imgs.append(img.astype(np.uint8))

            h, w = view_imgs[0].shape[:2]
            composite = np.zeros((h * 2, w * 2, 3), dtype=np.uint8)
            composite[:h, :w] = view_imgs[0]
            composite[:h, w:] = view_imgs[1]
            composite[h:, :w] = view_imgs[2]
            composite[h:, w:] = view_imgs[3]

            out_name = f"frame_{saved:05d}_t{int(ts)}.png"
            o3d.io.write_image(os.path.join(frame_dir, out_name),
                               o3d.geometry.Image(composite))
            saved += 1
            if idx % 50 == 0:
                print(f"  [{idx}/{len(ts_list)}] ...")

        vis.destroy_window()

    else:
        print(f"==> Rendering {len(ts_list)} frames with matplotlib (this may be slow) ...")
        saved = 0
        for idx, ts in enumerate(ts_list):
            obbs = timed_obbs[ts]
            if len(obbs) == 0:
                continue
            corners_w = obbs.bb3corners_world.numpy()
            M = corners_w.shape[0]
            probs = obbs.prob.squeeze(-1).numpy()
            colors = np.array([TAB20[i % len(TAB20)] for i in range(M)], dtype=np.float32)

            fig, axes = plt.subplots(2, 2, figsize=(14, 13),
                                     subplot_kw={"projection": "3d"})
            for ax, (az, el) in zip(axes.flat, views):
                _mpl_render_view(ax, corners_w, colors, probs, ctr, dist, az, el)
            plt.tight_layout()
            out_name = f"frame_{saved:05d}_t{int(ts)}.png"
            fig.savefig(os.path.join(frame_dir, out_name), dpi=120, bbox_inches="tight")
            plt.close(fig)
            saved += 1
            if idx % 20 == 0:
                print(f"  [{idx}/{len(ts_list)}] ...")

    print(f"==> Saved {saved} frames to: {frame_dir}")


def main():
    parser = argparse.ArgumentParser(description="Headless 3D scene renderer for BoxerNet CSV")
    parser.add_argument("--input", type=str, default=DEFAULT_SEQ, help="Sequence name or path")
    parser.add_argument("--output_dir", type=str, default=EVAL_PATH, help="Output directory")
    parser.add_argument("--write_name", default="boxer", type=str, help="CSV prefix")
    parser.add_argument("--fused", action="store_true", help="Prefer fused CSV")
    parser.add_argument("--per_frame", action="store_true", help="Render each frame separately")
    parser.add_argument("--max_boxes", type=int, default=3000, help="Max boxes for all-frames view")
    parser.add_argument("--skip_n", type=int, default=1, help="Subsample frames")
    parser.add_argument("--start_n", type=int, default=0, help="Start from N-th frame")
    parser.add_argument("--max_n", type=int, default=0, help="Max frames (0 = all)")
    args = parser.parse_args()

    output_dir = os.path.expanduser(args.output_dir)
    seq_name = args.input.rstrip("/").split("/")[-1]
    log_dir = os.path.join(output_dir, seq_name)

    # Pick CSV
    fused_csv = os.path.join(log_dir, f"{args.write_name}_3dbbs_fused.csv")
    raw_csv = os.path.join(log_dir, f"{args.write_name}_3dbbs.csv")
    tracked_csv = os.path.join(log_dir, f"{args.write_name}_3dbbs_tracked.csv")

    if args.fused and os.path.exists(fused_csv):
        csv_path = fused_csv
    elif os.path.exists(fused_csv):
        csv_path = fused_csv
    elif os.path.exists(raw_csv):
        csv_path = raw_csv
    elif os.path.exists(tracked_csv):
        csv_path = tracked_csv
    else:
        print(f"ERROR: No CSV found in {log_dir}")
        sys.exit(1)

    print(f"==> Loading: {csv_path}")
    timed_obbs = read_obb_csv(csv_path)

    ts_list = sorted(timed_obbs.keys())
    if args.max_n > 0:
        ts_list = ts_list[:args.max_n]
    ts_list = ts_list[args.start_n::args.skip_n]
    timed_obbs = {ts: timed_obbs[ts] for ts in ts_list if ts in timed_obbs}

    total_dets = sum(len(o) for o in timed_obbs.values())
    print(f"==> {len(timed_obbs)} frames, {total_dets} boxes")

    if _HAS_O3D:
        print(f"==> Backend: Open3D {_O3D_VERSION} (headless)")
    else:
        print("==> Backend: matplotlib (Open3D headless not available)")

    if args.per_frame:
        render_per_frame(timed_obbs, log_dir, args.write_name)
    else:
        render_all_frames(timed_obbs, log_dir, args.write_name, args.max_boxes)


if __name__ == "__main__":
    main()
