from __future__ import division

import numpy as np
from .constants import pi


def harmonic_oscillator_potential(r, mass, omega):
    return mass * (omega * r) ** 2 / 2.0


# def harmonic_oscillator_hamiltonian_ho_basis(omega, ):
#     return mass * (omega * r) ** 2 / 2.0


def kohn_sham_lo(rho, mass, g, a_s):
    r"""The leading order kohn sham potential (-J_0)

    Parameters
    ----------
    rho
    mass
    g
    a_s

    Returns
    -------

    """
    return 4 * pi * a_s / mass * (g - 1) / g * rho


def kohn_sham_exchange_lo(r, rho, mass, g, a_s):
    return 4 * pi * r ** 2 * (-4 * pi * a_s) / (2 * mass) * (g - 1) / g * rho ** 2


def kohn_sham_energy_lo(r, dr, rho, mass, g, a_s):
    rho_int = 4 * pi * dr * r ** 2 * rho ** 2
    pre = -0.5 * (g - 1) / g * 4 * pi * a_s / mass
    return pre * np.sum(rho_int)


def b1_lda(g):
    return (
        4
        / (35 * pi ** 2)
        * (g - 1)
        * (6 * pi ** 2 / g) ** (4 / 3)
        * (11 - 2 * np.log(2))
    )


def b2_lda(g):
    return (g - 1) / (10 * pi) * (6 * pi ** 2 / g) ** (5 / 3)


def b3_lda(g):
    return (g + 1) / (5 * pi) * (6 * pi ** 2 / g) ** (5 / 3)


def b4_lda(g):
    return (6 * pi ** 2 / g) ** (5 / 3) * (
        0.0755 * (g - 1) + 0.0574 * (g - 1) * (g - 3)
    )


def kohn_sham_nlo_lda(rho, mass, g, a_s):
    lo = kohn_sham_lo(rho=rho, mass=mass, g=g, a_s=a_s)
    b1 = b1_lda(g)
    nlo = (7 / 3) * b1 * a_s ** 2 / (2 * mass) * rho ** (4 / 3)
    return lo + nlo


def kohn_sham_exchange_nlo(r, rho, mass, g, a_s):
    lo = kohn_sham_exchange_lo(r=r, rho=rho, mass=mass, g=g, a_s=a_s)
    b1 = b1_lda(g)
    return lo + b1 * a_s ** 2 / (2 * mass) * rho ** (7 / 3)


def kohn_sham_energy_nlo(r, dr, rho, mass, g, a_s):
    lo = kohn_sham_energy_lo(r=r, dr=dr, rho=rho, mass=mass, g=g, a_s=a_s)
    rho_int = 4 * pi * dr * r ** 2 * rho ** (7 / 3)
    b1 = b1_lda(g)
    pre = -4 / 3 * b1 * a_s ** 2 / (2 * mass)
    return lo + pre * np.sum(rho_int)


def kohn_sham_nnlo_lda(rho, mass, g, a_s, a_p, r_s):
    nlo = kohn_sham_nlo_lda(rho=rho, mass=mass, g=g, a_s=a_s)
    b2 = b2_lda(g)
    b3 = b3_lda(g)
    b4 = b4_lda(g)
    pre = (b2 * a_s ** 2 * r_s) + (b3 * a_p ** 3) + (b4 * a_s ** 3)
    return nlo + 8 / 3 * pre / (2 * mass) * rho ** (5 / 3)


def kohn_sham_exchange_nnlo(r, rho, mass, g, a_s, a_p, r_s):
    nlo = kohn_sham_exchange_nlo(r=r, rho=rho, mass=mass, g=g, a_s=a_s)
    b2 = b2_lda(g)
    b3 = b3_lda(g)
    b4 = b4_lda(g)
    pre = (b2 * a_s ** 2 * r_s) + (b3 * a_p ** 3) + (b4 * a_s ** 3)
    return nlo + pre / (2 * mass) * rho ** (8 / 3)


def kohn_sham_energy_nnlo(r, dr, rho, mass, g, a_s, a_p, r_s):
    nlo = kohn_sham_energy_nlo(r=r, dr=dr, rho=rho, mass=mass, g=g, a_s=a_s)
    rho_int = 4 * pi * dr * r ** 2 * rho ** (8 / 3)
    b2 = b2_lda(g)
    b3 = b3_lda(g)
    b4 = b4_lda(g)
    pre = (b2 * a_s ** 2 * r_s) + (b3 * a_p ** 3) + (b4 * a_s ** 3) / (2 * mass)
    return nlo - 5 / 3 * pre * np.sum(rho_int)
