"""
core.py -- Spherical-Harmonic 3D Shape Compressor
====================================================

ALGORITHM
---------
Given a closed ("watertight"), convex surface mesh (every point on the
surface must be visible from the center -- i.e. a ray from the center hits
the surface exactly once in every direction), the shape can be written as a
single scalar function on the unit sphere:

    r(theta, phi) = distance from the center to the surface in direction (theta, phi)

where theta is the colatitude (angle from +z, 0..pi) and phi is the azimuth
(angle around z, 0..2*pi). This turns a 3D compression problem into a 2D
signal-compression problem on the sphere.

r(theta, phi) is real-valued (it's a distance), so we expand it in the REAL
orthonormal spherical harmonic basis:

    r(theta, phi) ~= sum_{l=0}^{N} sum_{m=-l}^{l}  c_lm * R_l^m(theta, phi)

R_l^m is built from an associated Legendre polynomial in theta (the
"spherical" part) times cos(m*phi) or sin(m*phi) in phi (the "Fourier
series" part) -- this is literally a Fourier series in phi whose
coefficients are functions of theta, re-expanded in Legendre polynomials.
Everything here is real; no complex numbers are needed because the signal
being compressed (a distance) is real. (Complex spherical harmonics Y_l^m
are the more common textbook form, and they *do* show up if you decompose
with e^{i*m*phi} instead of cos/sin(m*phi) -- but since r is real, the
resulting complex coefficients are redundant, c_{l,-m} = (-1)^m * conj(c_lm),
so you'd be storing exactly 2x more numbers than necessary. The real basis
used here stores the minimum: (N+1)^2 real numbers for degree N.)

COMPRESSION
-----------
- The mesh (V vertices, F faces) is replaced by (N+1)^2 real coefficients
  plus a center point and a mean radius -- typically many times smaller.
- Truncating the sum at a smaller N discards high-frequency (fine) detail
  while keeping the low-frequency (overall) shape -- exactly analogous to
  truncating a Fourier series or a JPEG's DCT coefficients.
- Decompression = evaluate the truncated series on any grid of directions
  you like and rebuild a mesh from the resulting radial function.

LIMITATION
----------
This only works for *star-shaped* objects (with respect to the chosen
center). Deep concavities, handles/holes (genus > 0, e.g. a torus or a
mug), or self-occluding parts (e.g. a hollow shell, arms tucked behind a
body) cannot be represented exactly -- rays that hit the surface more than
once are collapsed to their closest hit, and rays that hit the surface
zero times leave a gap that gets filled in by interpolation. The code below
detects and warns about both situations.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
import trimesh
from scipy import special



def load_mesh(path: str) -> trimesh.Trimesh:
    """Load a mesh file (OBJ, STL, PLY, GLB, ...) as a single Trimesh."""
    mesh = trimesh.load(path, force="mesh", process=False)
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError(f"Could not load a single triangle mesh from {path!r}")
    return mesh


def center_mesh(mesh: trimesh.Trimesh, center: Optional[np.ndarray] = None):
    """
    Translate a copy of `mesh` so `center` sits at the origin.
    Defaults to the vertex centroid.
    Returns (centered_mesh, center_point).
    """
    if center is None:
        center = np.asarray(mesh.vertices).mean(axis=0)
    else:
        center = np.asarray(center, dtype=float)

    centered = mesh.copy()
    centered.vertices = np.asarray(centered.vertices) - center

    if not centered.is_watertight:
        warnings.warn(
            "Mesh is not watertight. Ray casting (and hence the radial "
            "function) may have gaps or be ill-defined. Consider repairing "
            "the mesh first (e.g. trimesh.repair.fill_holes)."
        )
    return centered, center

def quadrature_grid(n_theta: int, n_phi: int):
    """
    Grid for *decomposition* (numerically integrating against the basis).
    Gauss-Legendre nodes in cos(theta) (exact for polynomials in cos(theta)
    up to degree 2*n_theta-1) crossed with a uniform grid in phi (exact for
    Fourier modes up to order n_phi-1). With n_theta = N+2 and
    n_phi = 2N+3 this integrates products of degree-N spherical harmonics
    to machine precision.

    Returns
    -------
    theta : (n_theta,)   colatitude nodes, in (0, pi)
    phi   : (n_phi,)     azimuth nodes, in [0, 2*pi)
    w_theta : (n_theta,) quadrature weights for the theta integral
    w_phi : float        (constant) quadrature weight for the phi integral
    """
    x, w_theta = np.polynomial.legendre.leggauss(n_theta)
    theta = np.arccos(np.clip(x, -1.0, 1.0))
    phi = np.linspace(0.0, 2 * np.pi, n_phi, endpoint=False)
    w_phi = 2 * np.pi / n_phi
    return theta, phi, w_theta, w_phi


def directions_from_angles(theta: np.ndarray, phi: np.ndarray):
    """theta:(n_theta,), phi:(n_phi,) -> unit vectors (n_theta, n_phi, 3), plus meshgrids."""
    TH, PH = np.meshgrid(theta, phi, indexing="ij")
    x = np.sin(TH) * np.cos(PH)
    y = np.sin(TH) * np.sin(PH)
    z = np.cos(TH)
    return np.stack([x, y, z], axis=-1), TH, PH


def min_degree_for_grid(n_theta: int, n_phi: int) -> int:
    """Largest N a (n_theta, n_phi) quadrature grid can decompose exactly."""
    return min(n_theta - 2, (n_phi - 3) // 2)


def recommended_grid(N: int, oversample: int = 0):
    """Smallest quadrature grid that exactly decomposes degree N (+ headroom)."""
    n_theta = N + 2 + oversample
    n_phi = 2 * N + 3 + 2 * oversample
    return n_theta, n_phi

def cast_radial_function(mesh: trimesh.Trimesh, theta: np.ndarray, phi: np.ndarray,
                          engine: str = "trimesh"):
    """
    For every direction (theta_i, phi_j), cast a ray from the origin and
    return the distance to the *closest* surface crossing (this is exactly
    "the distance to the point on the model closest to the line through the
    center and the sphere point", for a star-shaped mesh centered at the
    origin). Rays that miss the surface are filled by averaging their
    valid grid neighbours (with a warning if this happens often -- it's a
    sign the object isn't star-shaped from this center).

    engine: "trimesh" (fast, uses an AABB/rtree-accelerated intersector,
            default) or "numpy" (pure numpy Moller-Trumbore fallback, no
            extra native dependencies, slower on dense meshes).

    Returns r: (n_theta, n_phi) array of radial distances, and n_missed (int).
    """
    dirs, TH, PH = directions_from_angles(theta, phi)
    flat_dirs = dirs.reshape(-1, 3)
    n_rays = flat_dirs.shape[0]
    origins = np.zeros_like(flat_dirs)

    if engine == "trimesh":
        locations, index_ray, _ = mesh.ray.intersects_location(
            origins, flat_dirs, multiple_hits=True
        )
        r_flat = np.full(n_rays, np.inf)
        if len(index_ray) > 0:
            d = np.linalg.norm(locations - origins[index_ray], axis=1)
            np.minimum.at(r_flat, index_ray, d)
    elif engine == "numpy":
        r_flat = _moller_trumbore_closest_hit(
            np.zeros(3), flat_dirs, np.asarray(mesh.vertices), np.asarray(mesh.faces)
        )
        r_flat = np.where(np.isnan(r_flat), np.inf, r_flat)
    else:
        raise ValueError("engine must be 'trimesh' or 'numpy'")

    missed = ~np.isfinite(r_flat)
    n_missed = int(missed.sum())
    r_grid = r_flat.reshape(TH.shape)
    if n_missed:
        r_grid = _fill_missing_on_grid(r_grid, missed.reshape(TH.shape))
        frac = n_missed / n_rays
        if frac > 0.02:
            warnings.warn(
                f"{n_missed}/{n_rays} rays ({frac:.1%}) never hit the mesh -- "
                "the object is probably not fully star-shaped from this "
                "center. Missing samples were filled from grid neighbours, "
                "which will locally smooth out that part of the shape."
            )
    return r_grid, n_missed


def _fill_missing_on_grid(r_grid, missing_mask, max_iters: int = 50):
    """Iteratively fill NaN/inf entries with the mean of their valid grid
    neighbours (phi wraps around; theta does not). Structured-grid
    equivalent of a simple inpainting pass."""
    r = r_grid.copy()
    r[missing_mask] = np.nan
    with warnings.catch_warnings():
        # "Mean of empty slice" is expected/handled here: some grid points
        # may have zero valid neighbours in early iterations and simply get
        # filled on a later pass instead.
        warnings.simplefilter("ignore", category=RuntimeWarning)
        for _ in range(max_iters):
            nan_mask = np.isnan(r)
            if not nan_mask.any():
                break
            up = np.roll(r, 1, axis=0); up[0, :] = np.nan
            down = np.roll(r, -1, axis=0); down[-1, :] = np.nan
            left = np.roll(r, 1, axis=1)
            right = np.roll(r, -1, axis=1)
            stack = np.stack([up, down, left, right])
            neighbor_mean = np.nanmean(stack, axis=0)
            fillable = nan_mask & ~np.isnan(neighbor_mean)
            r[fillable] = neighbor_mean[fillable]
        if np.isnan(r).any():
            r[np.isnan(r)] = np.nanmean(r)
    return r


def _moller_trumbore_closest_hit(origin, directions, vertices, faces,
                                  tri_chunk: Optional[int] = None, eps: float = 1e-9):
    """Pure-numpy vectorized ray/triangle intersection (Moller-Trumbore),
    returns the closest forward hit distance per ray, NaN if none.
    Provided as a dependency-free fallback to the trimesh ray engine."""
    R = directions.shape[0]
    F = faces.shape[0]
    tri = vertices[faces]
    v0, v1, v2 = tri[:, 0], tri[:, 1], tri[:, 2]
    e1 = v1 - v0
    e2 = v2 - v0

    if tri_chunk is None:
        tri_chunk = max(1, int(5e7 / max(R, 1)))

    best_t = np.full(R, np.inf)
    o = np.broadcast_to(origin, (R, 3))

    for start in range(0, F, tri_chunk):
        end = min(start + tri_chunk, F)
        e1c, e2c, v0c = e1[start:end], e2[start:end], v0[start:end]

        d = directions[:, None, :]
        pvec = np.cross(d, e2c[None, :, :])
        det = np.einsum("rcj,cj->rc", pvec, e1c)
        invalid = np.abs(det) < eps
        det_safe = np.where(invalid, 1.0, det)

        tvec = o[:, None, :] - v0c[None, :, :]
        u = np.einsum("rcj,rcj->rc", tvec, pvec) / det_safe

        qvec = np.cross(tvec, e1c[None, :, :])
        v = np.einsum("rj,rcj->rc", directions, qvec) / det_safe
        t = np.einsum("rcj,cj->rc", qvec, e2c) / det_safe

        hit = (~invalid) & (u >= -1e-7) & (u <= 1 + 1e-7) & (v >= -1e-7) \
              & (u + v <= 1 + 1e-7) & (t > eps)
        t_masked = np.where(hit, t, np.inf)
        best_t = np.minimum(best_t, t_masked.min(axis=1))

    return np.where(np.isinf(best_t), np.nan, best_t)

def real_sh(l: int, m: int, theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
    """
    Real orthonormal spherical harmonic R_l^m(theta, phi), normalized so that
    integral over the sphere of (R_l^m)^2 dOmega = 1.

        m == 0:  R = Y_l^0                          (already real)
        m  > 0:  R = sqrt(2) * (-1)^m * Re(Y_l^m)  = sqrt(2)*(-1)^m*P_l^m(cos theta)*cos(m*phi)
        m  < 0:  R = sqrt(2) * (-1)^m * Im(Y_l^|m|) = sqrt(2)*(-1)^m*P_l^|m|(cos theta)*sin(|m|*phi)

    theta, phi must be broadcastable numpy arrays (colatitude, azimuth).
    """
    # scipy.special.sph_legendre_p(l, |m|, theta) == complex Y_l^|m|(theta, phi=0),
    # already fully normalized and including the Condon-Shortley phase.
    # It returns shape (1,) + theta.shape (a leading "derivative order" axis); we
    # only ever want the 0th derivative.
    P = special.sph_legendre_p(l, abs(m), theta)[0]
    if m == 0:
        return P
    elif m > 0:
        return np.sqrt(2.0) * ((-1) ** m) * P * np.cos(m * phi)
    else:
        mm = -m
        return np.sqrt(2.0) * ((-1) ** mm) * P * np.sin(mm * phi)


def degree_order_list(N: int):
    """All (l, m) pairs for l = 0..N, m = -l..l, in a fixed canonical order."""
    return [(l, m) for l in range(N + 1) for m in range(-l, l + 1)]


def sh_basis_matrix(lm_list, theta_flat: np.ndarray, phi_flat: np.ndarray) -> np.ndarray:
    """Design matrix of shape (n_points, n_coeffs); column j = R_{l_j}^{m_j} evaluated
    at every (theta_flat[i], phi_flat[i])."""
    B = np.empty((theta_flat.size, len(lm_list)))
    for j, (l, m) in enumerate(lm_list):
        B[:, j] = real_sh(l, m, theta_flat, phi_flat)
    return B


def sh_decompose(r_grid: np.ndarray, theta: np.ndarray, phi: np.ndarray,
                  w_theta: np.ndarray, w_phi: float, N: int):
    """
    Project r(theta, phi) (sampled on the quadrature grid returned by
    `quadrature_grid`) onto real spherical harmonics up to degree N.

    Returns coeffs (length (N+1)^2) and the corresponding lm_list.
    """
    n_theta, n_phi = r_grid.shape
    assert theta.shape == (n_theta,) and phi.shape == (n_phi,)

    TH, PH = np.meshgrid(theta, phi, indexing="ij")
    theta_flat, phi_flat, r_flat = TH.ravel(), PH.ravel(), r_grid.ravel()
    weights = np.repeat(w_theta, n_phi) * w_phi

    lm_list = degree_order_list(N)
    B = sh_basis_matrix(lm_list, theta_flat, phi_flat)
    # orthonormal basis -> the projection integral IS the coefficient:
    # c_lm = integral r * R_lm dOmega  ~=  sum_i r_i * R_lm(i) * weight_i
    coeffs = (B * weights[:, None]).T @ r_flat
    return coeffs, lm_list


def sh_reconstruct(coeffs: np.ndarray, lm_list, theta: np.ndarray, phi: np.ndarray,
                    N: Optional[int] = None) -> np.ndarray:
    """
    Evaluate sum_{l<=N} c_lm * R_lm(theta, phi) at arbitrary (theta, phi)
    (need not be a quadrature grid -- this is just a function evaluation).
    theta, phi may be any broadcastable shape (e.g. a fine mesh grid for
    rendering). N defaults to the full degree stored in lm_list.
    """
    if N is None:
        N = max(l for l, _ in lm_list)
    out = np.zeros(np.broadcast(theta, phi).shape)
    for c, (l, m) in zip(coeffs, lm_list):
        if l <= N and abs(c) > 0:
            out = out + c * real_sh(l, m, theta, phi)
    return out


def truncate(coeffs: np.ndarray, lm_list, N: int):
    """Return the coefficient sub-array and lm_list containing only l <= N
    (i.e. the actual compressed representation at degree N)."""
    keep = [j for j, (l, m) in enumerate(lm_list) if l <= N]
    return coeffs[keep], [lm_list[j] for j in keep]


def build_mesh_from_radial_function(r_func: Callable[[np.ndarray, np.ndarray], np.ndarray],
                                     n_rings: int = 80, n_cols: int = 160,
                                     center: np.ndarray = np.zeros(3)) -> trimesh.Trimesh:
    """
    Sample r_func on a uniform latitude/longitude grid (INCLUDING the poles)
    and build a closed, watertight triangle mesh (UV-sphere topology).
    r_func(theta_array, phi_array) -> radius_array, any shape.
    """
    theta_rings = np.linspace(0, np.pi, n_rings + 2)[1:-1]   # exclude exact poles
    phi_cols = np.linspace(0, 2 * np.pi, n_cols, endpoint=False)
    TH, PH = np.meshgrid(theta_rings, phi_cols, indexing="ij")
    R = np.asarray(r_func(TH.ravel(), PH.ravel())).reshape(TH.shape)
    R = np.clip(R, 1e-9, None)  # guard against a degenerate/negative radius

    dirs = np.stack([np.sin(TH) * np.cos(PH), np.sin(TH) * np.sin(PH), np.cos(TH)], axis=-1)
    ring_verts = dirs * R[..., None]

    north_r = float(np.asarray(r_func(np.array([0.0]), np.array([0.0])))[0])
    south_r = float(np.asarray(r_func(np.array([np.pi]), np.array([0.0])))[0])
    north = np.array([0.0, 0.0, max(north_r, 1e-9)])
    south = np.array([0.0, 0.0, -max(south_r, 1e-9)])

    verts = np.vstack([north[None], south[None], ring_verts.reshape(-1, 3)])
    ring_start = 2

    def ring_idx(i, j):
        return ring_start + i * n_cols + (j % n_cols)

    faces = []
    for j in range(n_cols):
        faces.append([0, ring_idx(0, j), ring_idx(0, j + 1)])
    for i in range(n_rings - 1):
        for j in range(n_cols):
            a, b = ring_idx(i, j), ring_idx(i, j + 1)
            c, d = ring_idx(i + 1, j), ring_idx(i + 1, j + 1)
            faces.append([a, c, d])
            faces.append([a, d, b])
    for j in range(n_cols):
        faces.append([1, ring_idx(n_rings - 1, j + 1), ring_idx(n_rings - 1, j)])

    mesh = trimesh.Trimesh(vertices=verts + center, faces=np.array(faces), process=True)
    return mesh


# =============================================================================
# 6. Error metrics
# =============================================================================

def radial_error(r_true_grid: np.ndarray, coeffs, lm_list, theta, phi, N: int):
    """RMSE and max-abs error, in the same units as the mesh, between the
    sampled ground-truth radial function and its degree-N reconstruction on
    the same grid."""
    TH, PH = np.meshgrid(theta, phi, indexing="ij")
    r_rec = sh_reconstruct(coeffs, lm_list, TH, PH, N=N)
    diff = r_rec - r_true_grid
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    max_err = float(np.max(np.abs(diff)))
    mean_r = float(np.mean(r_true_grid))
    return {
        "rmse": rmse,
        "max_abs_error": max_err,
        "rmse_pct_of_mean_radius": 100 * rmse / mean_r,
        "n_coeffs": sum(1 for l, m in lm_list if l <= N),
    }


# =============================================================================
# 7. Top-level convenience API + compressed-file I/O
# =============================================================================

@dataclass
class CompressedShape:
    coeffs: np.ndarray            # (n_coeffs,) real numbers, canonical (l,m) order
    lm_list: list = field(repr=False)
    N: int                        # degree the coefficients were fit at
    center: np.ndarray = field(default_factory=lambda: np.zeros(3))

    def reconstruct(self, N: Optional[int] = None, n_rings: int = 80, n_cols: int = 160):
        """Decompress into a closed triangle mesh at degree N (<= self.N)."""
        N = self.N if N is None else min(N, self.N)
        c, lms = truncate(self.coeffs, self.lm_list, N)

        def r_func(theta, phi):
            return sh_reconstruct(c, lms, theta, phi, N=N)

        return build_mesh_from_radial_function(r_func, n_rings=n_rings, n_cols=n_cols,
                                                 center=self.center)

    def save(self, path: str):
        """Save the compressed representation (this IS the compressed file)."""
        np.savez_compressed(
            path,
            coeffs=self.coeffs,
            l=np.array([l for l, m in self.lm_list]),
            m=np.array([m for l, m in self.lm_list]),
            N=self.N,
            center=self.center,
        )

    @staticmethod
    def load(path: str) -> "CompressedShape":
        data = np.load(path)
        lm_list = list(zip(data["l"].tolist(), data["m"].tolist()))
        return CompressedShape(coeffs=data["coeffs"], lm_list=lm_list,
                                N=int(data["N"]), center=data["center"])

    def nbytes(self) -> int:
        # coefficients (float64) + center (3 float64) + degree (int); ignore
        # the tiny fixed npz header overhead.
        return self.coeffs.nbytes + self.center.nbytes + 8


def compress_mesh(mesh_or_path, N: int, grid_oversample: int = 4,
                   center: Optional[np.ndarray] = None,
                   ray_engine: str = "trimesh") -> CompressedShape:
    """
    End-to-end: load (if given a path) -> center -> ray-cast a
    quadrature grid -> decompose to degree N. `grid_oversample` adds extra
    quadrature resolution beyond the minimum needed for degree N, which
    costs more ray casts but reduces aliasing from high-frequency detail in
    the input mesh that's beyond degree N anyway.
    """
    mesh = load_mesh(mesh_or_path) if isinstance(mesh_or_path, str) else mesh_or_path
    centered, c = center_mesh(mesh, center=center)

    n_theta, n_phi = recommended_grid(N, oversample=grid_oversample)
    theta, phi, w_theta, w_phi = quadrature_grid(n_theta, n_phi)
    r_grid, n_missed = cast_radial_function(centered, theta, phi, engine=ray_engine)

    coeffs, lm_list = sh_decompose(r_grid, theta, phi, w_theta, w_phi, N)
    return CompressedShape(coeffs=coeffs, lm_list=lm_list, N=N, center=c)


def compression_report(original_mesh: trimesh.Trimesh, compressed: CompressedShape,
                        N_values=None):
    """Human-readable table: bytes, compression ratio and (if N_values given)
    reconstruction error at each degree, reusing the same fitted coefficients
    (no re-fitting needed since lower-degree reconstructions are just a
    truncation of the same series)."""
    v_bytes = np.asarray(original_mesh.vertices).nbytes
    f_bytes = np.asarray(original_mesh.faces).nbytes
    orig_bytes = v_bytes + f_bytes

    rows = []
    Ns = N_values if N_values is not None else [compressed.N]
    for N in Ns:
        c, lms = truncate(compressed.coeffs, compressed.lm_list, N)
        comp_bytes = c.nbytes + compressed.center.nbytes + 8
        rows.append({
            "N": N,
            "n_coeffs": len(c),
            "compressed_bytes": comp_bytes,
            "original_bytes": orig_bytes,
            "ratio": orig_bytes / comp_bytes,
        })
    return rows
