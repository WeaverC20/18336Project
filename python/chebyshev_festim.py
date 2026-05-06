"""
Chebyshev pseudospectral solver for 1D McNabb-Foster hydrogen transport.

Mirrors the public API of FESTIM v2 so that user code reads almost identically
to a finite-element FESTIM run. The state vector is
    y = [c_m(x_0..x_N), c_t1(x_0..x_N), c_t2(x_0..x_N), ...]
on Chebyshev-Gauss-Lobatto nodes that cluster near both boundaries -- the
exact regions where boundary layers form when D(T) jumps.

Time integration is implicit backward Euler with Newton iteration and an
adaptive step-size controller modeled on FESTIM's `Stepsize`. Boundary
conditions enter as algebraic constraints that overwrite the diffusion-equation
rows for the mobile species at the boundary nodes.

Usage:
    import chebyshev_festim as CF
    my_model = CF.HydrogenTransportProblem()
    my_model.mesh = CF.ChebyshevMesh1D(N=64, x_max=1e-3)
    ...
    my_model.initialise()
    my_model.run()
"""

from __future__ import annotations

import math
import time as _time
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple, Union

import numpy as np
from scipy import linalg as splinalg

K_B_EV = 8.617333262145e-5  # eV / K


# =========================================================================
# Chebyshev infrastructure
# =========================================================================
def cheb_diff_matrix(N: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    First-derivative Chebyshev differentiation matrix on the canonical
    interval [-1, 1] for N+1 Gauss-Lobatto nodes (Trefethen, SMM p. 54).

    Returns
    -------
    D : (N+1, N+1) ndarray
        Differentiation matrix, ordered with xi descending from 1 to -1.
    xi : (N+1,) ndarray
        xi_j = cos(pi j / N), j = 0..N (descending).
    """
    if N == 0:
        return np.array([[0.0]]), np.array([1.0])
    j = np.arange(N + 1)
    xi = np.cos(np.pi * j / N)
    c = np.ones(N + 1)
    c[0] = 2.0
    c[-1] = 2.0
    c *= (-1.0) ** j
    X = np.tile(xi.reshape(-1, 1), (1, N + 1))
    dX = X - X.T
    D = np.outer(c, 1.0 / c) / (dX + np.eye(N + 1))
    D -= np.diag(D.sum(axis=1))
    return D, xi


# =========================================================================
# Lightweight FESTIM-compatible parameter classes
# =========================================================================
@dataclass
class Material:
    D_0: float
    E_D: float
    K_S_0: Optional[float] = None
    E_K_S: Optional[float] = None
    name: str = ""

    def D(self, T: float) -> float:
        return self.D_0 * math.exp(-self.E_D / (K_B_EV * T))

    def K_S(self, T: float) -> float:
        if self.K_S_0 is None:
            raise ValueError("Material has no Sieverts solubility defined.")
        return self.K_S_0 * math.exp(-self.E_K_S / (K_B_EV * T))


@dataclass
class Species:
    name: str
    mobile: bool = True


@dataclass
class ImplicitSpecies:
    """Empty-trap density. `others` lists the trapped Species sharing the site."""
    n: float
    others: List[Species]
    name: str = ""


@dataclass
class VolumeSubdomain1D:
    id: int
    borders: Tuple[float, float]
    material: Material


@dataclass
class SurfaceSubdomain1D:
    id: int
    x: float


# Boundary conditions ------------------------------------------------------
@dataclass
class FixedConcentrationBC:
    subdomain: SurfaceSubdomain1D
    value: Union[float, Callable]
    species: Species

    def evaluate(self, t: float, T: float) -> float:
        v = self.value
        if callable(v):
            try:
                return float(v(t))
            except TypeError:
                return float(v(t, T))
        return float(v)


@dataclass
class SievertsBC:
    subdomain: SurfaceSubdomain1D
    S_0: float
    E_S: float
    pressure: Union[float, Callable]
    species: Species

    def evaluate(self, t: float, T: float) -> float:
        p = self.pressure
        if callable(p):
            try:
                p_val = float(p(t))
            except TypeError:
                p_val = float(p(t, T))
        else:
            p_val = float(p)
        p_val = max(p_val, 0.0)
        S = self.S_0 * math.exp(-self.E_S / (K_B_EV * T))
        return S * math.sqrt(p_val)


# Reaction (McNabb-Foster trapping) ---------------------------------------
@dataclass
class Reaction:
    """
    Trapping reaction (mobile + empty trap -> trapped):
        c_t' = nu_m * c_m * (n_i - c_t) - nu_r * c_t
    with nu_m = k_0 exp(-E_k/kT) and nu_r = p_0 exp(-E_p/kT).
    """
    reactant: List
    product: List[Species]
    k_0: float
    E_k: float
    p_0: float
    E_p: float
    volume: VolumeSubdomain1D

    def nu_m(self, T: float) -> float:
        return self.k_0 * math.exp(-self.E_k / (K_B_EV * T))

    def nu_r(self, T: float) -> float:
        return self.p_0 * math.exp(-self.E_p / (K_B_EV * T))

    @property
    def trapped_species(self) -> Species:
        return self.product[0]

    @property
    def empty_trap(self) -> ImplicitSpecies:
        for r in self.reactant:
            if isinstance(r, ImplicitSpecies):
                return r
        raise ValueError("Reaction has no ImplicitSpecies reactant.")

    @property
    def mobile_species(self) -> Species:
        for r in self.reactant:
            if isinstance(r, Species) and r.mobile:
                return r
        raise ValueError("Reaction has no mobile Species reactant.")


# Sources ------------------------------------------------------------------
@dataclass
class ParticleSource:
    """value(x, t) -> H/m^3/s. x is a numpy array."""
    value: Callable
    volume: VolumeSubdomain1D
    species: Species


# Settings / Stepsize ------------------------------------------------------
@dataclass
class Settings:
    atol: float = 1e-10
    rtol: float = 1e-8
    final_time: float = 1.0
    max_iterations: int = 30
    transient: bool = True
    stepsize: Optional["Stepsize"] = None


@dataclass
class Stepsize:
    initial_value: float
    growth_factor: float = 1.1
    cutback_factor: float = 0.9
    target_nb_iterations: int = 4
    max_stepsize: Union[float, Callable, None] = None
    milestones: List[float] = field(default_factory=list)
    pre_milestone_duration: Optional[float] = None
    post_milestone_duration: Optional[float] = None

    def cap(self, t: float) -> float:
        m = self.max_stepsize
        if m is None:
            return float("inf")
        if callable(m):
            v = m(t)
            return float(v) if v is not None else float("inf")
        return float(m)

    def modify(self, dt: float, n_iter: int, t: float) -> float:
        if n_iter < self.target_nb_iterations:
            dt *= self.growth_factor
        elif n_iter > self.target_nb_iterations:
            dt *= self.cutback_factor

        # Pre/post milestone shaping (Colin's pattern from PTTEP)
        for m_ in self.milestones:
            if (
                self.pre_milestone_duration
                and 0 < (m_ - t) < self.pre_milestone_duration
            ):
                frac = (m_ - t) / self.pre_milestone_duration
                cap = self.initial_value + frac * (dt - self.initial_value)
                dt = min(dt, cap)
                break
            if (
                self.post_milestone_duration
                and 0 < (t - m_) < self.post_milestone_duration
            ):
                dt = self.initial_value
                break

        dt = min(dt, self.cap(t))
        return dt


# =========================================================================
# Mesh
# =========================================================================
class ChebyshevMesh1D:
    """
    1D Chebyshev-Gauss-Lobatto mesh on [x_min, x_max] with N+1 nodes.

    Nodes are returned ascending (x[0] = x_min, x[N] = x_max). The first-
    and second-derivative operators are stored ready for use on this ordering.

    Parameters
    ----------
    N : int
        Polynomial degree -- gives N+1 collocation nodes.
    x_max, x_min : float
        Endpoints of the physical interval.
    left_stretch : float, optional (default 0.0)
        Sinh-stretch parameter that clusters nodes toward ``x_min``. With
        s in [0,1] the natural Lobatto parameter, the physical mapping is
            x(s) = x_min + (x_max - x_min) * sinh(alpha * s) / sinh(alpha)
        and ``alpha = left_stretch``. Setting ``left_stretch = 0`` recovers
        a plain Chebyshev grid (smooth limit via L'Hopital). Larger
        positive values push the cluster more aggressively toward x_min,
        which helps resolve narrow boundary-layer sources without creating
        a singular Jacobian at x = x_min (unlike a polynomial stretch).
    """

    def __init__(self, N: int, x_max: float, x_min: float = 0.0,
                 left_stretch: float = 0.0):
        self.N = N
        self.x_min = x_min
        self.x_max = x_max
        self.left_stretch = float(left_stretch)
        D_xi, xi = cheb_diff_matrix(N)
        L = x_max - x_min
        # s in [0,1], descending in j (s=0 -> xi=1, s=1 -> xi=-1)
        s = (1.0 - xi) / 2.0
        a = self.left_stretch
        if a == 0.0:
            x_desc = x_min + L * s
            order = np.argsort(x_desc)
            self.x = x_desc[order]
            D_phys_desc = (-2.0 / L) * D_xi
            self.D = D_phys_desc[np.ix_(order, order)]
            self.D2 = self.D @ self.D
        else:
            # x = x_min + L * sinh(a s) / sinh(a)
            # dx/ds = L a cosh(a s) / sinh(a)
            # d2x/ds2 = L a^2 sinh(a s) / sinh(a)
            sinh_a = math.sinh(a)
            x_desc = x_min + L * np.sinh(a * s) / sinh_a
            dx_ds = L * a * np.cosh(a * s) / sinh_a
            d2x_ds2 = L * a * a * np.sinh(a * s) / sinh_a
            # ds/dxi = -1/2, d2s/dxi2 = 0
            ds_dxi = -0.5
            dx_dxi = dx_ds * ds_dxi  # = -0.5 * dx_ds
            d2x_dxi2 = d2x_ds2 * ds_dxi**2  # = 0.25 * d2x_ds2
            inv_dx = 1.0 / dx_dxi
            order = np.argsort(x_desc)
            self.x = x_desc[order]
            D_phys_desc = (inv_dx[:, None]) * D_xi
            D2_xi = D_xi @ D_xi
            D2_phys_desc = (inv_dx[:, None] ** 2) * D2_xi - (
                d2x_dxi2[:, None] / dx_dxi[:, None] ** 3
            ) * D_xi
            self.D = D_phys_desc[np.ix_(order, order)]
            self.D2 = D2_phys_desc[np.ix_(order, order)]
        self.vertices = self.x  # FESTIM-compatible alias

    @property
    def n_nodes(self) -> int:
        return len(self.x)

    def integrate(self, f: np.ndarray) -> float:
        """
        Clenshaw-Curtis quadrature on Lobatto nodes.

        ``f`` is given in ascending-x order. The CC weights are for nodes
        ξ_j = cos(πj/N) descending in j, which under our mapping corresponds
        to ascending physical x (j=0 -> x=x_min, j=N -> x=x_max). So no
        reversal is needed.

        For a non-linear mapping x = x(xi) we use
            int f(x) dx = int f(x(xi)) (dx/dxi) dxi.
        """
        N = self.N
        if N == 0:
            return 0.0
        w = _clenshaw_curtis_weights(N)
        L = self.x_max - self.x_min
        f_arr = np.asarray(f)
        if self.left_stretch == 0.0:
            return 0.5 * L * float(np.dot(w, f_arr))
        # Sinh stretch: |dx/dxi| = 0.5 * L * a * cosh(a s) / sinh(a)
        a = self.left_stretch
        sinh_a = math.sinh(a)
        j = np.arange(N + 1)
        xi_desc = np.cos(np.pi * j / N)
        s = (1.0 - xi_desc) / 2.0
        abs_dx_dxi = 0.5 * L * a * np.cosh(a * s) / sinh_a
        return float(np.dot(w, f_arr * abs_dx_dxi))


class MultiDomainChebyshevMesh1D:
    """
    Multi-domain (spectral-element-style) Chebyshev-Gauss-Lobatto mesh on
    [breaks[0], breaks[-1]] composed of ``len(Ns)`` subdomains stitched at
    shared interface nodes.

    Each block ``k`` carries its own Chebyshev-Lobatto grid with ``Ns[k]+1``
    nodes on ``[breaks[k], breaks[k+1]]`` (no per-block stretching). Adjacent
    blocks share their interface node, so the global mesh has
    ``sum(Ns[k]+1) - (n_blocks - 1) = sum(Ns) + 1`` unique nodes -- the
    polynomial degree across blocks behaves like a single global ``N`` such
    that ``n_nodes = N + 1``.

    The first- and second-derivative matrices are assembled block-by-block
    so that block-interior rows carry that block's local Chebyshev
    derivatives directly. Interface rows are special: they will be replaced
    by a *flux-continuity* equation when the solver assembles the residual
    (this is the standard strong-form spectral-element treatment of an
    interface for a smooth-coefficient diffusion problem). The mesh exposes
    the flux operator via ``iface_indices`` and ``iface_flux_op`` so the
    solver can do the override without knowing the block layout.

    The same C^0-continuity-with-shared-nodes pattern is what FESTIM's
    graded P1 mesh enforces between vertex blocks; this mesh is the
    pseudospectral analogue.

    Parameters
    ----------
    breaks : sequence of float
        Strictly increasing block boundaries. ``len(breaks) == len(Ns)+1``.
    Ns : sequence of int
        Per-block polynomial degrees; block ``k`` has ``Ns[k]+1`` nodes.
    """

    def __init__(self, breaks, Ns):
        breaks = [float(b) for b in breaks]
        Ns = [int(n) for n in Ns]
        if len(breaks) != len(Ns) + 1:
            raise ValueError("breaks must have len(Ns) + 1 entries.")
        if any(breaks[i] >= breaks[i + 1] for i in range(len(Ns))):
            raise ValueError("breaks must be strictly increasing.")
        if any(n < 1 for n in Ns):
            raise ValueError("each block degree Ns[k] must be >= 1.")

        self.breaks = breaks
        self.Ns = Ns
        self.n_blocks = len(Ns)
        self.x_min = breaks[0]
        self.x_max = breaks[-1]
        # API parity with ChebyshevMesh1D (single-domain unstretched).
        self.left_stretch = 0.0
        self.N = sum(Ns)  # global degree: n_nodes = N + 1

        # Per-block local nodes and operators on [breaks[k], breaks[k+1]].
        block_xs = []
        block_Ds = []
        block_D2s = []
        for k, Nk in enumerate(Ns):
            a, b = breaks[k], breaks[k + 1]
            D_xi, xi = cheb_diff_matrix(Nk)
            # xi descends from 1 (j=0) to -1 (j=Nk); s = (1-xi)/2 ascends 0..1.
            # x = a + (b-a) s ascends a..b. dx/dxi = -(b-a)/2.
            s = (1.0 - xi) / 2.0
            x_block = a + (b - a) * s
            D_phys = (-2.0 / (b - a)) * D_xi
            D2_phys = D_phys @ D_phys
            block_xs.append(x_block)
            block_Ds.append(D_phys)
            block_D2s.append(D2_phys)

        # Global indexing: block k spans global indices offsets[k]..offsets[k]+Nk.
        # The right-end node of block k is the same global DOF as the left-end
        # node of block k+1, so offsets[k+1] = offsets[k] + Ns[k].
        offsets = [0]
        for k in range(self.n_blocks):
            offsets.append(offsets[-1] + Ns[k])
        self._offsets = offsets
        n_total = offsets[-1] + 1

        # Concatenate node positions, dropping the duplicated first entry of
        # every block past the first.
        x_global = [block_xs[0]]
        for k in range(1, self.n_blocks):
            x_global.append(block_xs[k][1:])
        self.x = np.concatenate(x_global)
        self.vertices = self.x  # FESTIM-compatible alias

        # Assemble global D and D2. Block-interior rows take the block's own
        # Chebyshev derivative rows; shared interface rows are zeroed here
        # because the solver will overwrite them with the flux-continuity
        # equation built below in ``iface_flux_op``. (Mixing the two sides'
        # second-derivative rows by averaging is mathematically tempting but
        # blows up the Jacobian conditioning when adjacent blocks have very
        # different widths -- the smaller block's row scales as 1/h^2 and
        # swamps the larger block's contributions.)
        D_global = np.zeros((n_total, n_total))
        D2_global = np.zeros((n_total, n_total))
        for k in range(self.n_blocks):
            i0 = offsets[k]
            Nk = Ns[k]
            for i_loc in range(Nk + 1):
                g_row = i0 + i_loc
                is_left_iface = (i_loc == 0) and (k > 0)
                is_right_iface = (i_loc == Nk) and (k < self.n_blocks - 1)
                if is_left_iface or is_right_iface:
                    continue
                D_global[g_row, i0:i0 + Nk + 1] = block_Ds[k][i_loc, :]
                D2_global[g_row, i0:i0 + Nk + 1] = block_D2s[k][i_loc, :]

        self.D = D_global
        self.D2 = D2_global

        # Interface flux-continuity rows. For the shared node g between
        # block k-1 and block k, the strong-form interface equation is
        #     (D u/dx)|_{x_g, left}  =  (D u/dx)|_{x_g, right},
        # which, with the same diffusivity on both sides, reduces to
        #     D_block[k-1][end, :] @ u_left  =  D_block[k][0, :] @ u_right.
        # We pack this as a single global row in iface_flux_op so the solver
        # can overwrite the mobile-row residual at index g without needing
        # to know the block decomposition. ``iface_indices`` is the list of
        # global node indices that are interface nodes.
        self.iface_indices = []
        self.iface_flux_op = np.zeros((n_total, n_total))
        for k in range(1, self.n_blocks):
            g = offsets[k]
            Nkm1 = Ns[k - 1]
            Nk = Ns[k]
            cols_left = slice(offsets[k - 1], offsets[k] + 1)
            cols_right = slice(offsets[k], offsets[k] + Nk + 1)
            self.iface_flux_op[g, cols_left] += block_Ds[k - 1][Nkm1, :]
            self.iface_flux_op[g, cols_right] -= block_Ds[k][0, :]
            self.iface_indices.append(g)

    @property
    def n_nodes(self) -> int:
        return len(self.x)

    def integrate(self, f: np.ndarray) -> float:
        """
        Sum of Clenshaw-Curtis quadrature over each block. Shared interface
        nodes contribute once per side with each side's local CC weight and
        Jacobian -- the two contributions correctly cover the small
        neighbourhood of the interface from both directions.
        """
        f = np.asarray(f)
        total = 0.0
        for k, Nk in enumerate(self.Ns):
            i0 = self._offsets[k]
            i1 = i0 + Nk + 1
            f_block = f[i0:i1]
            w = _clenshaw_curtis_weights(Nk)
            a, b = self.breaks[k], self.breaks[k + 1]
            total += 0.5 * (b - a) * float(np.dot(w, f_block))
        return total


def _clenshaw_curtis_weights(N: int) -> np.ndarray:
    """Clenshaw-Curtis quadrature weights on [-1,1] for N+1 Lobatto nodes."""
    if N == 0:
        return np.array([2.0])
    if N == 1:
        return np.array([1.0, 1.0])
    c = np.zeros(N + 1)
    c[0] = 1.0
    c[N] = 1.0 / (1.0 - N * N) if (N % 2 == 0) else 0.0
    # Use FFT-based formula (Trefethen, "Is Gauss quadrature better than CC?").
    n = np.arange(2, N, 2) if N >= 2 else np.array([], dtype=int)
    # Direct formula from Trefethen:
    w = np.zeros(N + 1)
    theta = np.pi * np.arange(N + 1) / N
    for k in range(N + 1):
        s = 0.0
        for jj in range(1, N // 2 + 1):
            b = 2.0 if 2 * jj < N else 1.0
            s += b / (4.0 * jj * jj - 1.0) * math.cos(2.0 * jj * theta[k])
        w[k] = (1.0 - s) * 2.0 / N
    w[0] *= 0.5
    w[-1] *= 0.5
    return w


# =========================================================================
# Exports
# =========================================================================
class _ExportBase:
    def __init__(self):
        self.t: List[float] = []
        self.data = []


class SurfaceFlux(_ExportBase):
    """-D dC/dx at the surface (FESTIM convention: outward-pointing flux)."""

    def __init__(self, field: Species, surface: SurfaceSubdomain1D):
        super().__init__()
        self.field = field
        self.surface = surface


class TotalVolume(_ExportBase):
    def __init__(self, field: Species, volume: VolumeSubdomain1D):
        super().__init__()
        self.field = field
        self.volume = volume


class Profile1DExport(_ExportBase):
    """Spatial snapshots at requested times (or every step if times=None)."""

    def __init__(self, field: Species, times: Optional[List[float]] = None):
        super().__init__()
        self.field = field
        self.times = times
        self.x: Optional[np.ndarray] = None


# =========================================================================
# The solver
# =========================================================================
class HydrogenTransportProblem:
    """Cheby pseudospectral analogue of festim.HydrogenTransportProblem."""

    def __init__(self):
        self.mesh: Optional[ChebyshevMesh1D] = None
        self.subdomains: List = []
        self.species: List[Species] = []
        self.reactions: List[Reaction] = []
        self.boundary_conditions: List = []
        self.sources: List[ParticleSource] = []
        self.temperature: Union[float, Callable] = 300.0
        self.settings: Settings = Settings()
        self.exports: List = []
        self.initial_conditions: dict = {}  # {species_name: float or callable}
        # Diagnostics filled during run():
        self.solve_stats: dict = {}

    # ---- helpers ----
    def _T(self, t: float) -> float:
        T = self.temperature
        if callable(T):
            return float(T(t))
        return float(T)

    def _mobile(self) -> Species:
        for s in self.species:
            if s.mobile:
                return s
        raise ValueError("No mobile species defined.")

    def _trap_species(self) -> List[Species]:
        return [s for s in self.species if not s.mobile]

    def _bc_for_surface(
        self, surf: SurfaceSubdomain1D
    ) -> Optional[Union[FixedConcentrationBC, SievertsBC]]:
        for bc in self.boundary_conditions:
            if bc.subdomain is surf or bc.subdomain.id == surf.id:
                return bc
        return None

    def _surface_subdomains(self) -> Tuple[SurfaceSubdomain1D, SurfaceSubdomain1D]:
        surfs = [s for s in self.subdomains if isinstance(s, SurfaceSubdomain1D)]
        if len(surfs) != 2:
            raise ValueError("Expected exactly two SurfaceSubdomain1D entries.")
        # Order by x: left first.
        surfs.sort(key=lambda s: s.x)
        return surfs[0], surfs[1]

    def _volume(self) -> VolumeSubdomain1D:
        for s in self.subdomains:
            if isinstance(s, VolumeSubdomain1D):
                return s
        raise ValueError("No VolumeSubdomain1D defined.")

    # ---- initialisation ----
    def initialise(self):
        if self.mesh is None:
            raise RuntimeError("Set my_model.mesh before initialise().")
        self.x = self.mesh.x
        self.D1 = self.mesh.D
        self.D2 = self.mesh.D2
        self.n_nodes = self.mesh.n_nodes

        self.mobile = self._mobile()
        self.traps = self._trap_species()
        self.n_traps = len(self.traps)
        self.n_state = self.n_nodes * (1 + self.n_traps)

        # Map trap species -> reaction governing it
        self._trap_to_rxn = {}
        for r in self.reactions:
            self._trap_to_rxn[r.trapped_species.name] = r

        # Surfaces (left/right).
        self.left_surf, self.right_surf = self._surface_subdomains()
        self.left_bc = self._bc_for_surface(self.left_surf)
        self.right_bc = self._bc_for_surface(self.right_surf)
        self.volume = self._volume()
        self.material = self.volume.material

        # Initial conditions: mobile and traps default to 0
        y0 = np.zeros(self.n_state)
        for spe_name, val in self.initial_conditions.items():
            idx = self._species_block(spe_name)
            if callable(val):
                y0[idx] = val(self.x)
            else:
                y0[idx] = float(val)

        # Apply BC values to t=0 mobile boundary nodes (consistency).
        T0 = self._T(0.0)
        if self.left_bc is not None:
            y0[0] = self.left_bc.evaluate(0.0, T0)
        if self.right_bc is not None:
            y0[self.n_nodes - 1] = self.right_bc.evaluate(0.0, T0)

        self.y = y0
        self.t = 0.0

        # Prepare exports
        for exp in self.exports:
            exp.t = []
            exp.data = []
            if isinstance(exp, Profile1DExport):
                exp.x = self.x.copy()

        # Pre-record state at t=0 so plots include the initial condition
        self._record_exports(0.0)

    def _species_block(self, name: str) -> slice:
        if name == self.mobile.name:
            return slice(0, self.n_nodes)
        for i, sp in enumerate(self.traps):
            if sp.name == name:
                start = (i + 1) * self.n_nodes
                return slice(start, start + self.n_nodes)
        raise KeyError(f"Species {name} not found.")

    # ---- residual + jacobian ----
    def _residual_and_jacobian(
        self, y_new: np.ndarray, y_old: np.ndarray, dt: float, t_new: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        N = self.n_nodes
        n_t = self.n_traps
        T = self._T(t_new)
        D_T = self.material.D(T)

        # Source term
        S = np.zeros(N)
        for src in self.sources:
            if src.species is self.mobile or src.species.name == self.mobile.name:
                S = S + np.asarray(src.value(self.x, t_new), dtype=float).reshape(N)

        # Slice helpers
        c_m = y_new[0:N]
        c_t = [y_new[(i + 1) * N : (i + 2) * N] for i in range(n_t)]

        # Reaction params per trap
        nu_m = []
        nu_r = []
        n_density = []
        for sp in self.traps:
            r = self._trap_to_rxn[sp.name]
            nu_m.append(r.nu_m(T))
            nu_r.append(r.nu_r(T))
            n_density.append(r.empty_trap.n)

        # ---- residual ----
        F = np.zeros_like(y_new)

        # Mobile residual (interior + boundary -- we'll overwrite boundary rows)
        diff_term = D_T * (self.D2 @ c_m)
        # Sum of trap source/sink reactions felt by mobile
        rxn_sum = np.zeros(N)
        for i in range(n_t):
            rate = nu_m[i] * c_m * (n_density[i] - c_t[i]) - nu_r[i] * c_t[i]
            rxn_sum += rate

        F[0:N] = (c_m - y_old[0:N]) / dt - diff_term - S + rxn_sum

        # Trap residual at every node
        for i in range(n_t):
            sl = slice((i + 1) * N, (i + 2) * N)
            rate = nu_m[i] * c_m * (n_density[i] - c_t[i]) - nu_r[i] * c_t[i]
            F[sl] = (c_t[i] - y_old[sl]) / dt - rate

        # ---- Jacobian ----
        J = np.zeros((self.n_state, self.n_state))

        # Mobile-mobile block: d(F_m)/dc_m
        # = I/dt - D_T * D2 + sum_i nu_m_i * (n_i - c_t_i)*I  (diagonal contribution)
        diag_mm = np.full(N, 1.0 / dt)
        for i in range(n_t):
            diag_mm = diag_mm + nu_m[i] * (n_density[i] - c_t[i])
        J[0:N, 0:N] = -D_T * self.D2 + np.diag(diag_mm)

        # Mobile-trap blocks: d(F_m)/dc_t_i = diag( -nu_m_i * c_m - nu_r_i )
        # (because rxn_sum has -nu_m c_m c_t - nu_r c_t -- note signs in rate expression)
        # rate = nu_m c_m (n - c_t) - nu_r c_t = nu_m c_m n - nu_m c_m c_t - nu_r c_t
        # d rate / d c_t = -nu_m c_m - nu_r
        # d(F_m)/dc_t = +d rxn_sum / d c_t = -nu_m c_m - nu_r  (diagonal)
        for i in range(n_t):
            sl = slice((i + 1) * N, (i + 2) * N)
            J[0:N, sl] = np.diag(-nu_m[i] * c_m - nu_r[i])

        # Trap-mobile blocks: d(F_t_i)/dc_m = diag( -nu_m_i (n_i - c_t_i) )
        for i in range(n_t):
            sl = slice((i + 1) * N, (i + 2) * N)
            J[sl, 0:N] = np.diag(-nu_m[i] * (n_density[i] - c_t[i]))

        # Trap-trap blocks: d(F_t_i)/dc_t_i = diag( 1/dt + nu_m_i c_m + nu_r_i )
        for i in range(n_t):
            sl = slice((i + 1) * N, (i + 2) * N)
            diag_tt = 1.0 / dt + nu_m[i] * c_m + nu_r[i]
            J[sl, sl] = np.diag(diag_tt)

        # ---- multi-domain interface flux-continuity overwrite ----
        # Replace the mobile-row residual/Jacobian at every shared interface
        # node with the strong-form flux-balance equation
        # ``D_block_left[end, :] @ u  -  D_block_right[0, :] @ u = 0`` so the
        # solution stays C^1 across blocks. The trap rows at interface nodes
        # are untouched: traps are local ODEs, the standard backward-Euler
        # update at row N+g still applies. Single-domain meshes set
        # ``iface_indices = []`` (or don't define it), so this is a no-op.
        iface_indices = getattr(self.mesh, "iface_indices", None)
        if iface_indices:
            iface_op = self.mesh.iface_flux_op
            for g in iface_indices:
                F[g] = float(iface_op[g, :] @ c_m)
                J[g, :] = 0.0
                J[g, 0:N] = iface_op[g, :]

        # ---- BC overwrite (mobile only, Dirichlet) ----
        if self.left_bc is not None:
            c_left = self.left_bc.evaluate(t_new, T)
            F[0] = c_m[0] - c_left
            J[0, :] = 0.0
            J[0, 0] = 1.0
        if self.right_bc is not None:
            c_right = self.right_bc.evaluate(t_new, T)
            F[N - 1] = c_m[N - 1] - c_right
            J[N - 1, :] = 0.0
            J[N - 1, N - 1] = 1.0

        return F, J

    # ---- a single backward-Euler step with damped Newton iteration ----
    def _step(
        self, dt: float, t_new: float
    ) -> Tuple[bool, int, np.ndarray]:
        atol = self.settings.atol
        rtol = self.settings.rtol
        max_it = self.settings.max_iterations
        N = self.n_nodes
        y = self.y.copy()
        # Better initial guess across BC jumps: snap boundary mobile values to
        # the new BC so Newton starts close to the answer.
        T_new_val = self._T(t_new)
        if self.left_bc is not None:
            y[0] = self.left_bc.evaluate(t_new, T_new_val)
        if self.right_bc is not None:
            y[N - 1] = self.right_bc.evaluate(t_new, T_new_val)

        res0 = None
        for it in range(max_it):
            F, J = self._residual_and_jacobian(y, self.y, dt, t_new)
            res_norm = float(np.linalg.norm(F, ord=np.inf))
            sol_norm = max(float(np.linalg.norm(y, ord=np.inf)), 1.0)
            if res0 is None:
                res0 = max(res_norm, 1.0)
            # Relative convergence (residual reduced by 1/rtol from initial)
            # OR absolute (||F|| < atol + rtol * ||y||)
            converged = (
                res_norm < atol + rtol * sol_norm
                or res_norm / res0 < rtol
            )
            if converged and it > 0:
                return True, it, y
            try:
                dy = splinalg.solve(J, -F)
            except splinalg.LinAlgError:
                return False, it, y
            if not np.all(np.isfinite(dy)):
                return False, it, y

            # Backtracking line search: accept smaller and smaller fractions
            # until residual decreases (or we give up after a few halvings).
            alpha = 1.0
            accepted = False
            for _ in range(6):
                y_try = y + alpha * dy
                if not np.all(np.isfinite(y_try)):
                    alpha *= 0.5
                    continue
                F_try, _ = self._residual_and_jacobian(y_try, self.y, dt, t_new)
                res_try = float(np.linalg.norm(F_try, ord=np.inf))
                if not np.isfinite(res_try):
                    alpha *= 0.5
                    continue
                if getattr(self, "_debug_step", False) and len(ls_log) == 0:
                    imax = int(np.argmax(np.abs(F_try)))
                    print(f'           y_try[0..3]={y_try[0:4]} y_try[N-2..]={y_try[N-2:N+1]}')
                    print(f'           imax_F_try={imax} (block {imax//N}, node {imax%N}), F_try.max={F_try[imax]:.3e}')
                    # check trap values
                    for ii in range(self.n_traps):
                        sl = slice((ii+1)*N, (ii+2)*N)
                        print(f'           trap{ii} max={y_try[sl].max():.3e} min={y_try[sl].min():.3e}')
                # Accept either if residual drops or first iteration (initial guess uncertain)
                if res_try < res_norm or it == 0 or alpha <= 0.0625:
                    y = y_try
                    accepted = True
                    break
                alpha *= 0.5
            if not accepted:
                # No progress -- bail out so caller can cut dt
                return False, it, y

        # Did not converge in max_it
        F, _ = self._residual_and_jacobian(y, self.y, dt, t_new)
        res_norm = float(np.linalg.norm(F, ord=np.inf))
        sol_norm = max(float(np.linalg.norm(y, ord=np.inf)), 1.0)
        ok = res_norm < atol + rtol * sol_norm
        return ok, max_it, y

    # ---- exports ----
    def _record_exports(self, t: float):
        N = self.n_nodes
        T = self._T(t)
        D_T = self.material.D(T)
        c_m = self.y[0:N]
        for exp in self.exports:
            if isinstance(exp, SurfaceFlux):
                # Outward flux at surface = -D dc/dx . n_out
                gradc = self.D1 @ c_m
                if exp.surface is self.left_surf or exp.surface.id == self.left_surf.id:
                    # Outward normal at left points -x => flux_out = +D * dc/dx[0]
                    val = D_T * gradc[0]
                else:
                    # Outward normal at right points +x => flux_out = -D * dc/dx[-1]
                    val = -D_T * gradc[-1]
                exp.t.append(t)
                exp.data.append(val)
            elif isinstance(exp, TotalVolume):
                if exp.field is self.mobile or exp.field.name == self.mobile.name:
                    f = c_m
                else:
                    sl = self._species_block(exp.field.name)
                    f = self.y[sl]
                exp.t.append(t)
                exp.data.append(self.mesh.integrate(f))
            elif isinstance(exp, Profile1DExport):
                if exp.field is self.mobile or exp.field.name == self.mobile.name:
                    f = c_m.copy()
                else:
                    sl = self._species_block(exp.field.name)
                    f = self.y[sl].copy()
                if exp.times is None:
                    exp.t.append(t)
                    exp.data.append(f)
                else:
                    # Snapshot only when crossing a requested time
                    last_t = exp.t[-1] if exp.t else -np.inf
                    for tt in exp.times:
                        if last_t < tt <= t:
                            exp.t.append(t)
                            exp.data.append(f)
                            break

    # ---- main run ----
    def run(self, verbose: bool = True):
        if self.settings.stepsize is None:
            raise RuntimeError("Set my_model.settings.stepsize before run().")
        ss = self.settings.stepsize
        t_final = self.settings.final_time
        dt = ss.initial_value
        n_steps = 0
        n_failed = 0
        n_iters_total = 0
        t0_wall = _time.time()

        # Build a sorted list of milestones for milestone clamping
        milestones = sorted([m for m in (ss.milestones or []) if 0 < m <= t_final])
        next_ms_idx = 0
        # advance past milestones already crossed (e.g. t=0 init)
        while next_ms_idx < len(milestones) and milestones[next_ms_idx] <= self.t + 1e-12:
            next_ms_idx += 1

        while self.t < t_final - 1e-12:
            # Cap dt to not overshoot final_time or next milestone
            dt = min(dt, ss.cap(self.t))
            if next_ms_idx < len(milestones):
                dt = min(dt, milestones[next_ms_idx] - self.t)
            dt = min(dt, t_final - self.t)
            # Treat FP-tiny remainders as done. Without this, accumulated
            # floating-point drift in self.t after >~1000 fixed-cap steps
            # leaves the loop alive with dt in the 1e-13 range, where
            # Newton on the (1/dt)-scaled mass diagonal cannot converge.
            if dt <= 1e-12 * max(t_final, 1.0):
                break

            t_new = self.t + dt
            ok, n_iter, y_new = self._step(dt, t_new)

            if ok:
                self.y = y_new
                self.t = t_new
                n_steps += 1
                n_iters_total += n_iter
                self._record_exports(self.t)
                # Cross milestone
                if (
                    next_ms_idx < len(milestones)
                    and abs(self.t - milestones[next_ms_idx]) < 1e-9
                ):
                    next_ms_idx += 1
                # Adjust dt for next step
                dt = ss.modify(dt, n_iter, self.t)
                if verbose and n_steps % 25 == 0:
                    print(
                        f"  step {n_steps}: t={self.t:.3e}/{t_final:.2e} "
                        f"dt={dt:.2e} newton_it={n_iter}"
                    )
            else:
                # Cut back and retry
                n_failed += 1
                dt *= ss.cutback_factor
                if dt < 1e-14 * t_final:
                    raise RuntimeError(
                        f"Time stepping failed: dt collapsed at t={self.t:.3e}"
                    )

        wall = _time.time() - t0_wall
        self.solve_stats = {
            "n_steps": n_steps,
            "n_failed": n_failed,
            "n_iter_total": n_iters_total,
            "wall_time": wall,
        }
        if verbose:
            print(
                f"Cheby run done: {n_steps} steps ({n_failed} retries), "
                f"{n_iters_total} Newton iters, {wall:.2f} s wall."
            )
