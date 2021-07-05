import numpy as np
import scipy as sp
from scipy.linalg import eigh

from .constants import *
from .potentials import (
    kohn_sham_lo,
    kohn_sham_nlo_lda,
    kohn_sham_nnlo_lda,
    kohn_sham_energy_lo,
    kohn_sham_energy_nlo,
    kohn_sham_energy_nnlo,
)
from .harmonic_oscillator import (
    ho_radial_wf,
    maximum_wf_index,
    ho_energy,
    compute_omega,
    ho_density,
)


class Emulator:
    def __init__(self, H0, H1, n_max_vec=1):
        self.H0 = H0
        self.H1 = H1
        self.H0_sub = None
        self.H1_sub = None
        self.X_sub = None
        self.N_sub = None
        self.p_train = None
        self.n_max_vec = n_max_vec

    def compute_full_hamiltonian(self, p):
        return self.H0 + self.H1 @ p

    def compute_subspace_hamiltonian(self, p):
        return self.H0_sub + self.H1_sub @ p

    def solve_schrodinger_full(self, p):
        E, psi = eigh(self.compute_full_hamiltonian(p))
        return E[self.n_max_vec], psi[:, : self.n_max_vec]

    def solve_schrodinger_subspace(self, p):
        E, beta = eigh(self.compute_subspace_hamiltonian(p), b=self.N_sub, type=1)
        return E[self.n_max_vec], beta[:, : self.n_max_vec]

    def fit(self, p_train, max_vecs=2):
        X = []
        for p in p_train:
            _, psi = self.solve_schrodinger_full(p)
            # print(psi.shape)
            psi = psi[:, :max_vecs]
            # print(psi.shape)
            X.append(psi)
        X = np.concatenate(X, axis=1)
        X_sub = X

        N_sub = X.T @ X
        self.X_sub = X
        self.N_sub = N_sub
        self.p_train = p_train

        self.H0_sub = X.T @ self.H0 @ X
        H1_sub_reshaped = X_sub.T @ (np.transpose(self.H1, (2, 0, 1)) @ X_sub)
        self.H1_sub = np.transpose(H1_sub_reshaped, (1, 2, 0))
        return self

    def predict(self, p, use_emulator=True):
        if use_emulator:
            E, beta = self.solve_schrodinger_subspace(p)
            return E, self.X_sub @ beta
        else:
            return self.solve_schrodinger_full(p)


class HarmonicTrap:
    def __init__(
        self, r, dr, mass, b, g, n_max_shell, n_max_osc, kind="nnlo", damping_factor=0.8
    ):
        self.r = r
        self.dr = dr
        self.mass = mass
        self.b = b
        self.g = g
        # self.a_s = a_s
        # self.a_p = a_p
        # self.r_s = r_s
        self.n_max_shell = n_max_shell
        self.n_max_osc = n_max_osc
        # self.omega = compute_omega(mass=mass, b=b)
        self.omega = 1
        self.rho_init = ho_density(r=r, n_max=n_max_shell, g=g, b=b)
        self.kind = kind
        self.damping_factor = damping_factor

        if kind == "harmonic":

            def j0_func(rho, params):
                C0 = params[0]
                return 0.5 * C0 * rho ** 2

            def energy_func(r, dr, rho, params):
                C0 = params[0]
                rho_int = np.sum(dr * r ** 2 * rho ** 3)
                return -0.5 * C0 * rho_int / 3

        elif kind == "lo":

            def j0_func(rho, params):
                a_s = params[0]
                return kohn_sham_lo(rho, mass=mass, g=g, a_s=a_s)

            def energy_func(r, dr, rho, params):
                a_s = params[0]
                return kohn_sham_energy_lo(r=r, dr=dr, rho=rho, mass=mass, g=g, a_s=a_s)

        elif kind == "nlo":

            def j0_func(rho, params):
                a_s = params[0]
                return kohn_sham_nlo_lda(rho, mass=mass, g=g, a_s=a_s)

            def energy_func(r, dr, rho, params):
                a_s = params[0]
                return kohn_sham_energy_nlo(
                    r=r, dr=dr, rho=rho, mass=mass, g=g, a_s=a_s
                )

        elif kind == "nnlo":

            def j0_func(rho, params):
                a_s, a_p, r_s = params
                return kohn_sham_nnlo_lda(
                    rho, mass=mass, g=g, a_s=a_s, a_p=a_p, r_s=r_s
                )

            def energy_func(r, dr, rho, params):
                a_s, a_p, r_s = params
                return kohn_sham_energy_nnlo(
                    r=r, dr=dr, rho=rho, mass=mass, g=g, a_s=a_s, a_p=a_p, r_s=r_s
                )

        else:
            raise ValueError("Order must be one of 'harmonic', 'lo', 'nlo', or 'nnlo'.")

        self.j0_func = j0_func
        self.energy_func = energy_func
        self.ell_max = n_max_shell

        self.H_ho_osc = np.zeros((self.ell_max + 1, n_max_osc + 1, n_max_osc + 1))
        # self.V_ks_osc = np.zeros((self.ell_max + 1, n_max_osc + 1, n_max_osc + 1))
        self.wfs_ho = np.zeros((self.ell_max + 1, n_max_osc + 1, r.shape[0]))
        for ell in range(self.ell_max + 1):
            for i in range(n_max_osc + 1):
                n = i + 1
                self.H_ho_osc[ell, i, i] = ho_energy(n, ell, self.omega)
                self.wfs_ho[ell, i] = ho_radial_wf(r, n=n, ell=ell, b=b)

        self.N_sub = None
        self.X_sub = None
        self.X_r_sub = None

    def convert_to_ho_basis(self, v, ell, out=None):
        n_max = self.n_max_osc
        if out is None:
            out = np.zeros((n_max + 1, n_max + 1))
        for n in range(n_max + 1):
            u_nl = self.wfs_ho[ell, n]
            for m in range(n, n_max + 1):
                u_ml = self.wfs_ho[ell, m]
                out[n, m] = out[m, n] = np.sum(v * u_nl * u_ml * self.dr)
        return out

    @staticmethod
    def solve_schrodinger_equation_full(H):
        E_nl, u_nl = eigh(H)
        return E_nl, u_nl

    def solve_schrodinger_equation_subspace(self, H, ell):
        H = self.X_sub[ell].T @ H @ self.X_sub[ell]
        # print(H.shape, self.N_sub[ell].shape, self.X_sub[ell].shape)
        E_nl, beta = eigh(H, b=self.N_sub[ell], type=1)
        u_nl = self.X_r_sub[ell] @ beta
        return E_nl, u_nl

    def kinetic_energy_ks(self, params, rho):
        V_ks_osc = np.zeros((self.ell_max + 1, self.n_max_osc + 1, self.n_max_osc + 1))
        J0 = self.j0_func(rho=rho, params=params)
        for ell in range(self.ell_max + 1):
            self.convert_to_ho_basis(v=J0, ell=ell, out=V_ks_osc[ell])
        H = self.H_ho_osc + V_ks_osc
        T = 0
        for ell in range(self.ell_max + 1):
            max_idx = maximum_wf_index(n_max=self.n_max_shell, ell=ell)
            E_nl = np.linalg.eigvalsh(H[ell])
            T += (2 * ell + 1) * np.sum(E_nl[: max_idx + 1])
        return self.g * T

    def update_rho(self, params, rho, out=None):
        J0 = self.j0_func(rho=rho, params=params)
        V_ks_osc = np.zeros((self.ell_max + 1, self.n_max_osc + 1, self.n_max_osc + 1))
        for ell in range(self.ell_max + 1):
            self.convert_to_ho_basis(v=J0, ell=ell, out=V_ks_osc[ell])
        H = self.H_ho_osc + V_ks_osc

        if out is None:
            out = np.zeros(rho.shape)

        out[:] = 0.0
        r = self.r
        g = self.g
        for ell in range(self.ell_max + 1):
            max_idx = maximum_wf_index(n_max=self.n_max_shell, ell=ell)
            E_nl, u_nl = np.linalg.eigh(H[ell])
            u_nl = self.wfs_ho[ell].T @ u_nl

            for i in range(max_idx + 1):
                out += (2 * ell + 1) * u_nl[:, i] ** 2
        out *= g / (4 * pi * r ** 2)
        return out

    def consistent_density_and_orbitals(
        self, params, tolerance=1e-2, use_emulator=False, verbose=False
    ):
        rho_current = self.rho_init.copy()
        rho_new = rho_current.copy()
        J0 = self.j0_func(rho=rho_current, params=params)
        V_ks_osc = np.zeros((self.ell_max + 1, self.n_max_osc + 1, self.n_max_osc + 1))
        for ell in range(self.ell_max + 1):
            self.convert_to_ho_basis(v=J0, ell=ell, out=V_ks_osc[ell])

        r = self.r
        g = self.g
        iteration = 0
        error = np.inf
        max_iter = 1000
        u_nl_best = {}
        while (error > tolerance or iteration < 5) and (iteration < max_iter):
            # print(iteration, error)
            H = self.H_ho_osc + V_ks_osc
            rho_new[:] = 0.0
            for ell in range(self.ell_max + 1):
                max_idx = maximum_wf_index(n_max=self.n_max_shell, ell=ell)
                if use_emulator:
                    E_nl, u_nl = self.solve_schrodinger_equation_subspace(H[ell], ell)
                else:
                    E_nl, u_nl = np.linalg.eigh(H[ell])
                    u_nl_best[ell] = u_nl[:, : max_idx + 1]
                    u_nl = self.wfs_ho[ell].T @ u_nl

                for i in range(max_idx + 1):
                    rho_new += (2 * ell + 1) * u_nl[:, i] ** 2
            rho_new *= g / (4 * pi * r ** 2)
            J0 = self.j0_func(rho=rho_new, params=params)
            for ell in range(self.ell_max + 1):
                self.convert_to_ho_basis(v=J0, ell=ell, out=V_ks_osc[ell])
            error = np.max(np.abs(rho_current - rho_new))
            # if verbose:
            #     print(iteration, error)
            rho_current[:] = (
                self.damping_factor * rho_current + (1 - self.damping_factor) * rho_new
            )
            iteration += 1
        if verbose:
            print(f"It took {iteration} iterations to converge")
            print(f"The error is {error}")
        if iteration == max_iter:
            print(f"Warning: Did not converge in {iteration} iterations")
        return rho_new, u_nl_best

    def fit(self, p_train, tolerance=1e-2):
        X_sub = {ell: [] for ell in range(self.ell_max + 1)}
        X_r_sub = {ell: [] for ell in range(self.ell_max + 1)}
        rho_train = []
        for params in p_train:
            rho_i, wfs = self.consistent_density_and_orbitals(
                params=params, tolerance=tolerance, use_emulator=False, verbose=False
            )
            rho_train.append(rho_i)
            for ell in wfs:
                X_sub[ell].append(wfs[ell])
                wf_r = self.wfs_ho[ell].T @ wfs[ell]
                X_r_sub[ell].append(wf_r)

        rho_train = np.stack(rho_train, axis=1)
        # print(rho_train)
        N_sub = {}
        for ell in X_sub:
            X_sub[ell] = np.concatenate(X_sub[ell], axis=1)
            X_r_sub[ell] = np.concatenate(X_r_sub[ell], axis=1)
            N_sub[ell] = X_sub[ell].T @ X_sub[ell]

        self.rho_train = rho_train
        self.X_sub = X_sub
        self.X_r_sub = X_r_sub
        self.N_sub = N_sub
        return self

    def _energy_subspace(self, beta, params):
        beta = np.concatenate([beta, [1-np.sum(beta)]])
        rho = np.abs(self.rho_train @ beta)
        return self.energy(params=params, rho=rho)

    def energy(self, params, rho=None, **kwargs):
        if rho is None:
            rho = self.predict(params, **kwargs)
        T = self.kinetic_energy_ks(params=params, rho=rho)
        E = self.energy_func(r=self.r, dr=self.dr, rho=rho, params=params)
        return T + E

    def predict(self, params, tolerance=1e-2, use_emulator=False, verbose=False):
        if use_emulator:
            from scipy.optimize import minimize

            beta0 = np.zeros(self.rho_train.shape[-1]-1)
            beta0[0] = 0
            bounds = [(-10, 10) for _ in beta0]
            # method = "Nelder-Mead"
            method = "L-BFGS-B"
            res = minimize(
                self._energy_subspace,
                x0=beta0,
                args=(params,),
                bounds=bounds,
                method=method,
                tol=tolerance,
            )
            beta_opt = res.x
            beta_opt = np.concatenate([beta_opt, [1-np.sum(beta_opt)]])
            rho = np.abs(self.rho_train @ beta_opt)
            rho = self.update_rho(params=params, rho=rho)
            return rho, res
        else:
            return self.consistent_density_and_orbitals(
                params=params,
                tolerance=tolerance,
                use_emulator=use_emulator,
                verbose=verbose,
            )[0]
