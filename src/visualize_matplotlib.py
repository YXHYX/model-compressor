"""
visualize_matplotlib.py -- headless-friendly fallback visualizations
=======================================================================

Mayavi requires a VTK + GUI/OpenGL stack that isn't always available
(notably: it currently fails to build against VTK >= 9.3 on Python 3.12 --
see the README). These matplotlib-based equivalents need nothing beyond
numpy + matplotlib, work on a headless server, and are handy for quickly
sanity-checking results before switching to visualize_mayavi.py for nicer
interactive 3D views.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import core


def _plot_mesh(ax, mesh, color, alpha=1.0, offset=(0, 0, 0)):
    v = np.asarray(mesh.vertices) + np.asarray(offset)
    f = np.asarray(mesh.faces)
    tris = v[f]
    coll = Poly3DCollection(tris, facecolor=color, edgecolor="none", alpha=alpha)
    ax.add_collection3d(coll)
    return v


def compare_original_vs_reconstructions(original_mesh, compressed, N_values,
                                         n_rings=60, n_cols=120, figsize=None,
                                         savepath=None):
    """Grid of 3D subplots: original mesh + one reconstruction per N."""
    n_panels = 1 + len(N_values)
    figsize = figsize or (4 * n_panels, 4.2)
    fig = plt.figure(figsize=figsize)

    all_v = np.asarray(original_mesh.vertices)
    lim = np.abs(all_v).max() * 1.1

    ax = fig.add_subplot(1, n_panels, 1, projection="3d")
    _plot_mesh(ax, original_mesh, color=(0.85, 0.55, 0.25))
    ax.set_title("original")

    for i, N in enumerate(N_values):
        rec = compressed.reconstruct(N=N, n_rings=n_rings, n_cols=n_cols)
        ax = fig.add_subplot(1, n_panels, i + 2, projection="3d")
        _plot_mesh(ax, rec, color=(0.35, 0.55, 0.85))
        n_coeffs = sum(1 for l, m in compressed.lm_list if l <= N)
        ax.set_title(f"N={N}  ({n_coeffs} coeffs)")

    for ax in fig.axes:
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_zlim(-lim, lim)
        ax.set_box_aspect([1, 1, 1])
        ax.set_axis_off()

    fig.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=140, bbox_inches="tight")
    return fig


def plot_radial_heatmap(r_grid, theta, phi, savepath=None, title="radial function r(theta, phi)"):
    """Flat (phi, theta) heatmap of the radial function -- the exact signal
    that gets Fourier/SH-decomposed, shown as an image (like an equirectangular
    projection of the surface distance)."""
    fig, ax = plt.subplots(figsize=(7, 4))
    im = ax.imshow(r_grid, extent=[0, 360, 180, 0], aspect="auto", cmap="viridis")
    ax.set_xlabel("phi (deg, azimuth)")
    ax.set_ylabel("theta (deg, colatitude)")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="radius")
    if savepath:
        fig.savefig(savepath, dpi=140, bbox_inches="tight")
    return fig


def plot_error_vs_degree(rows, savepath=None):
    """rows: list of dicts from core.radial_error(...) at several N, each
    must include 'N' key merged in by the caller. Plots RMSE (%) and
    compression ratio vs N on twin axes."""
    Ns = [r["N"] for r in rows]
    rmse_pct = [r["rmse_pct_of_mean_radius"] for r in rows]
    ratio = [r["ratio"] for r in rows]

    fig, ax1 = plt.subplots(figsize=(6.5, 4.2))
    ax1.plot(Ns, rmse_pct, "o-", color="tab:red")
    ax1.set_xlabel("degree N")
    ax1.set_ylabel("reconstruction RMSE (% of mean radius)", color="tab:red")
    ax1.tick_params(axis="y", labelcolor="tab:red")

    ax2 = ax1.twinx()
    ax2.plot(Ns, ratio, "s--", color="tab:blue")
    ax2.set_ylabel("compression ratio (orig bytes / compressed bytes)", color="tab:blue")
    ax2.set_yscale("log")
    ax2.tick_params(axis="y", labelcolor="tab:blue")

    fig.suptitle("Compression trade-off: quality vs. degree N")
    fig.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=140, bbox_inches="tight")
    return fig
