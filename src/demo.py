"""
demo.py -- end-to-end example / CLI for the spherical-harmonic shape compressor

Usage
-----
    # Run on a built-in synthetic "lumpy asteroid" test shape:
    python demo.py --degrees 2 4 8 16 32 --outdir out

    # Run on your own mesh:
    python demo.py --input my_model.obj --degrees 4 8 16 32 64 --outdir out

Outputs (written to --outdir):
    original.obj                 the (centered) input mesh
    compressed_N{N}.npz          the compressed coefficient file for the
                                  largest requested N (truncate further with
                                  CompressedShape.reconstruct(N=smaller) --
                                  no need to store one file per degree)
    reconstructed_N{N}.obj       decompressed mesh at each requested degree
    report.txt                   compression ratio / error table
    radial_heatmap.png           the r(theta, phi) signal being compressed
    comparison.png               original vs. reconstructions, side by side
    error_vs_degree.png          RMSE and compression ratio vs N
"""
import argparse
import os

import numpy as np
import trimesh

import core


def make_synthetic_test_mesh(seed: int = 42, subdivisions: int = 5):
    """A 'lumpy asteroid': smooth low-frequency shape (degree <= 10, random
    coefficients) so there's a known ground truth to compare against, handy
    when you don't have your own mesh file handy."""
    rng = np.random.default_rng(seed)
    L_TRUE = 10
    lm_list = core.degree_order_list(L_TRUE)
    coeffs = {}
    for (l, m) in lm_list:
        coeffs[(l, m)] = 3.0 if l == 0 else rng.normal(0, 0.35 / (1 + l ** 1.3))

    def r_func(theta, phi):
        r = np.zeros_like(theta)
        for (l, m), c in coeffs.items():
            r = r + c * core.real_sh(l, m, theta, phi)
        return r

    ico = trimesh.creation.icosphere(subdivisions=subdivisions, radius=1.0)
    v = np.asarray(ico.vertices)
    theta = np.arccos(np.clip(v[:, 2], -1, 1))
    phi = np.arctan2(v[:, 1], v[:, 0]) % (2 * np.pi)
    verts = v * r_func(theta, phi)[:, None]
    return trimesh.Trimesh(vertices=verts, faces=ico.faces, process=False)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", type=str, default=None,
                     help="Path to a mesh file (OBJ/STL/PLY/...). If omitted, a "
                          "synthetic test shape is generated.")
    ap.add_argument("--degrees", type=int, nargs="+", default=[2, 4, 8, 16, 32],
                     help="Spherical-harmonic degrees N to report/reconstruct.")
    ap.add_argument("--outdir", type=str, default="../sh_compressor_out")
    ap.add_argument("--grid-oversample", type=int, default=4,
                     help="Extra quadrature resolution beyond the minimum needed "
                          "for the largest requested degree (reduces aliasing).")
    ap.add_argument("--rings", type=int, default=100, help="Output mesh latitude rings.")
    ap.add_argument("--cols", type=int, default=200, help="Output mesh longitude columns.")
    ap.add_argument("--engine", choices=["trimesh", "numpy"], default="trimesh",
                     help="Ray-casting backend. 'trimesh' (AABB-accelerated) is "
                          "much faster; 'numpy' is a dependency-free fallback.")
    ap.add_argument("--no-plots", action="store_true", help="Skip matplotlib figures.")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    N_max = max(args.degrees)

    # ---- 1. Load or generate the mesh ----
    if args.input:
        mesh = core.load_mesh(args.input)
        print(f"Loaded {args.input}: {len(mesh.vertices)} verts, {len(mesh.faces)} faces, "
              f"watertight={mesh.is_watertight}")
    else:
        mesh = make_synthetic_test_mesh()
        print("No --input given: using a generated synthetic 'lumpy asteroid' test "
              f"shape ({len(mesh.vertices)} verts, {len(mesh.faces)} faces).")

    centered, center = core.center_mesh(mesh)
    centered.export(os.path.join(args.outdir, "original.obj"))

    # ---- 2. Fit once at the highest requested degree; lower degrees are just truncations ----
    print(f"\nFitting spherical harmonics up to degree N={N_max} "
          f"({(N_max + 1) ** 2} coefficients) ...")
    compressed = core.compress_mesh(centered, N=N_max, grid_oversample=args.grid_oversample,
                                     center=np.zeros(3), ray_engine=args.engine)
    compressed.save(os.path.join(args.outdir, f"compressed_N{N_max}.npz"))

    # ---- 3. Report ----
    rows = core.compression_report(centered, compressed, N_values=sorted(args.degrees))
    n_theta, n_phi = core.recommended_grid(N_max, oversample=args.grid_oversample)
    theta, phi, w_theta, w_phi = core.quadrature_grid(n_theta, n_phi)
    r_grid, n_missed = core.cast_radial_function(centered, theta, phi, engine=args.engine)

    lines = []
    header = f"{'N':>4} | {'#coeffs':>8} | {'compressed KB':>14} | {'ratio':>10} | {'RMSE (% radius)':>16}"
    lines.append(header)
    lines.append("-" * len(header))
    for row in rows:
        err = core.radial_error(r_grid, compressed.coeffs, compressed.lm_list, theta, phi, row["N"])
        row["rmse_pct_of_mean_radius"] = err["rmse_pct_of_mean_radius"]
        lines.append(f"{row['N']:>4} | {row['n_coeffs']:>8} | "
                      f"{row['compressed_bytes']/1024:>14.2f} | {row['ratio']:>10.1f} | "
                      f"{err['rmse_pct_of_mean_radius']:>16.3f}")
    report_text = "\n".join(lines)
    print("\n" + report_text)
    with open(os.path.join(args.outdir, "report.txt"), "w") as f:
        f.write(report_text + "\n")

    # ---- 4. Reconstruct + export a mesh at every requested degree ----
    for N in sorted(args.degrees):
        rec = compressed.reconstruct(N=N, n_rings=args.rings, n_cols=args.cols)
        path = os.path.join(args.outdir, f"reconstructed_N{N}.obj")
        rec.export(path)
        print(f"  saved {path}  (watertight={rec.is_watertight}, volume={rec.volume:.4f})")

    # ---- 5. Plots (matplotlib fallback; swap for visualize_mayavi.py for interactive 3D) ----
    if not args.no_plots:
        import visualize_matplotlib as vplt

        ## TODO
        ## WORK ON VISUALIZATION FOR DEMO
        vplt.plot_radial_heatmap(r_grid, theta, phi,
                                  savepath=os.path.join(args.outdir, "radial_heatmap.png"))
        vplt.compare_original_vs_reconstructions(
            centered, compressed, N_values=sorted(args.degrees)[:4],
            n_rings=60, n_cols=120,
            savepath=os.path.join(args.outdir, "comparison.png"))
        vplt.plot_error_vs_degree(rows, savepath=os.path.join(args.outdir, "error_vs_degree.png"))
        print(f"\nSaved figures to {args.outdir}/")

    print(f"\nDone. All output in: {os.path.abspath(args.outdir)}")


if __name__ == "__main__":
    main()
