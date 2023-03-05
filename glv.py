import numpy as np
from scipy.integrate import solve_ivp

class Glv:
    """
    This class is responsible to solve the GLV model with verification of reaching the steady state
    for a given parameters.
    """
    def __init__(self, n_samples, n_species, delta, r, s,
                 interaction_matrix, initial_cond, final_time, max_step):
        """
        :param n_samples: The number of samples you are need to compute.
        :param n_species: The number of species at each sample.
        :param delta: This parameter is responsible for the stop condition at the steady state.
        :param r: growth rate vector of shape (,n_species).
        :param s: logistic growth term vector of size (,n_species).
        :param A: interaction matrix of shape (n_species, n_species).
        :param Y: set of initial conditions for each sample. If n_samples=1, the shape is (,n_species).
        If n_samples=m for m!=1 so the shape is (n_species, n_samples)
        :param final_time: the final time of the integration.
        :param max_step: maximal allowed step size.
        """
        self.smp = n_samples
        self.n = n_species
        self.delta = delta
        self.r = r
        self.s = s
        self.A = interaction_matrix
        self.Y = initial_cond
        self.final_time = final_time
        self.max_step = max_step
        # Initiation.
        self.Final_abundances = np.zeros((self.n, self.smp))
        self.Final_abundances_single_sample = np.zeros(self.n)

    def solve(self):
        """
        This function updates the final abundances, rows are the species and columns represent the samples.
        """
        def f(t, x):
            """
            GLV formula.
            """
            return np.array([self.r[i] * x[i] - self.s[i] * x[i] ** 2 + sum([self.A[i, p] * x[
                i] * x[p] for p in range(self.n) if p != i]) for i in range(self.n)])

        def event(t, x):
            """
            Event function that triggers when dxdt is close to steady state.
            """
            return max(abs(f(t, x))) - self.delta

        if self.smp > 1:  # Solution for cohort.
            for m in range(self.smp):
                # solve GLV up to time span.
                sol = solve_ivp(f, (0, self.final_time), self.Y[:][m], max_step=self.max_step, events=event)

                # Get the index at which the event occurred.
                event_idx = int(sol.t_events[0][0]) if len(sol.t_events[0]) > 0 else None

                # Save the solution up to the event time.
                self.Final_abundances[:, m] = sol.y[:, event_idx] if event_idx is not None else sol.y[:, -1]
            final_abundances = self.Final_abundances
            return final_abundances

        else:  # Solution for single sample.
            sol = solve_ivp(f, (0, self.final_time),
                            self.Y[:], max_step=self.max_step, events=event)

            # Get the index at which the event occurred
            event_idx = int(sol.t_events[0][0]) if len(sol.t_events[0]) > 0 else None

            # Save the solution up to the event time
            self.Final_abundances_single_sample[:] = sol.y[:, event_idx] if event_idx is not None else sol.y[:, -1]
        final_abundances = self.Final_abundances_single_sample
        return final_abundances.T

    def normalize_results(self):
        """
        Normalization of the final abundances.
        """
        if self.smp > 1:  # Normalization for cohort
            norm_factors = np.sum(self.Final_abundances, axis=0)
            print(norm_factors)
            norm_Final_abundances = np.array([self.Final_abundances[:, i] / norm_factors[i] for i in range(
                0, np.size(norm_factors))])
            return norm_Final_abundances.T
        else:  # Normalization for single sample
            norm_factor = np.sum(self.Final_abundances_single_sample)
            norm_Final_abundances_single_sample = self.Final_abundances_single_sample/norm_factor
            return norm_Final_abundances_single_sample.T
