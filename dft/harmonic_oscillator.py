from scipy.special import eval_genlaguerre, gammaln
import numpy as np
from .constants import hbar_c, pi


def compute_omega(mass, b):
    R"""Returns omega in units of MeV

    Parameters
    ----------
    mass
    b
    """
    return hbar_c ** 2 / (mass * b ** 2)


def ho_radial_wf(r, n, ell, b):
    r"""The radial wave function u_{nl} for the 3d isotropic harmonic oscillator.

    These are normalized such that \int |u_nl(r)|**2 dr = 1

    Parameters
    ----------
    r :
        The distance in fm
    n :
        The n quantum number
    ell :
        The angular momentum quantum number
    b :
        The oscillator parameter
    # mass :
    #     Mass in MeV
    # omega :
    #     The harmonic oscillator angular frequency in MeV

    Returns
    -------
    u_nl
    """
    # b = 1 / np.sqrt(mass * omega / hbar_c)
    # N_{nl} = 2 Gamma(n) / [b * Gamma(n + l + 1/2)]
    norm = np.sqrt(2 * np.exp(gammaln(n) - np.log(b) - gammaln(n + ell + 0.5)))
    y = r / b
    y2 = y ** 2
    laguerre = eval_genlaguerre(n - 1, ell + 0.5, y2)
    return norm * y ** (ell + 1) * np.exp(-y2 / 2) * laguerre


def ho_energy(n, ell, omega):
    R"""The energy of the harmonic oscillator

    Note that N = 2 (n - 1) + ell.

    Parameters
    ----------
    n
    ell
    omega

    Returns
    -------

    """
    return omega * (2 * (n - 1) + ell + 3 / 2)


def ho_density(r, n_max, g, b):
    rho = np.zeros(len(r))
    for ell in range(n_max+1):
        max_idx = maximum_wf_index(n_max=n_max, ell=ell)
        for i in range(max_idx+1):
            n = i + 1  # n starts at 1, not 0
            u_nl = ho_radial_wf(r=r, n=n, ell=ell, b=b)
            rho += (2 * ell + 1) * u_nl ** 2
    rho *= g / (4 * pi * r ** 2)
    return rho


def total_fermions(n_max, g):
    return g * (n_max + 1) * (n_max + 2) * (n_max + 3) / 6


def total_fermions_one_shell(N, g):
    return g * (N + 1) * (N + 2) / 2


def maximum_wf_index(n_max, ell):
    return int(np.floor((n_max - ell) / 2))
