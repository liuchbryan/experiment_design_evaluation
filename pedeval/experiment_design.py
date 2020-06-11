from abc import ABC, abstractmethod
from pedeval.sampler import BinarySampler, NormalSampler

import numpy as np
from scipy.stats import norm, percentileofscore
import noisyopt
import warnings


def _actual_effect_intersection_only(mu_Iphi: float, mu_Ipsi: float, **kwargs):
    return mu_Ipsi - mu_Iphi


def _mde_size_intersection_only(sigma_sq_Iphi: float, sigma_sq_Ipsi: float, n_3: int, z: float, **kwargs) -> float:
    return(
        z *
        np.sqrt(2 * (sigma_sq_Iphi + sigma_sq_Ipsi) / n_3)
    )


def _actual_effect_all_sample(mu_C1: float, mu_I1: float , mu_C2: float, mu_I2: float,
                              mu_Iphi: float, mu_Ipsi: float,
                              n_0: int, n_1: int, n_2: int, n_3: int, **kwargs) -> float:
    return(
        (n_1 * (mu_C1 - mu_I1) + n_2 * (mu_I2 - mu_C2) + n_3 * (mu_Ipsi - mu_Iphi)) /
        (n_0 + n_1 + n_2 + n_3)
    )


def _mde_size_all_sample(sigma_sq_C0: float, sigma_sq_C1: float, sigma_sq_I1: float,
                         sigma_sq_C2: float, sigma_sq_I2: float,
                         sigma_sq_Iphi: float, sigma_sq_Ipsi: float,
                         n_0: int, n_1: int, n_2: int, n_3:int , z: float, **kwargs) -> float:
    return(
        z *
        np.sqrt(2 * (n_0 * 2 * sigma_sq_C0 + n_1 * (sigma_sq_C1 + sigma_sq_I1) +
                     n_2 * (sigma_sq_C2 + sigma_sq_I2) + n_3 * (sigma_sq_Iphi + sigma_sq_Ipsi))) /
        (n_0 + n_1 + n_2 + n_3)
    )


def _actual_effect_qualified_only(mu_C1: float, mu_I1: float , mu_C2: float, mu_I2: float,
                                  mu_Iphi: float, mu_Ipsi: float, n_1: int, n_2: int, n_3: int, **kwargs) -> float:
    return (
        (n_1 * (mu_C1 - mu_I1) + n_2 * (mu_I2 - mu_C2) + n_3 * (mu_Ipsi - mu_Iphi)) /
        (n_1 + n_2 + n_3)
    )


def _mde_size_qualified_only(sigma_sq_C1: float, sigma_sq_I1: float,
                             sigma_sq_C2: float, sigma_sq_I2: float,
                             sigma_sq_Iphi: float, sigma_sq_Ipsi: float,
                             n_1: int, n_2: int, n_3:int , z: float, **kwargs) -> float:
    return (
        z *
        np.sqrt(2 * (n_1 * (sigma_sq_C1 + sigma_sq_I1) +
                     n_2 * (sigma_sq_C2 + sigma_sq_I2) + n_3 * (sigma_sq_Iphi + sigma_sq_Ipsi))) /
        (n_1 + n_2 + n_3)
    )


def _actual_effect_dual_control(mu_C1: float, mu_I1: float , mu_C2: float, mu_I2: float,
                                mu_C3: float, mu_Iphi: float, mu_Ipsi: float,
                                n_1: int, n_2: int, n_3: int, **kwargs) -> float:
    return (
        (n_2 * (mu_I2 - mu_C2) + n_3 * (mu_Ipsi - mu_C3)) / (n_2 + n_3) -
        (n_1 * (mu_I1 - mu_C1) + n_3 * (mu_Iphi - mu_C3)) / (n_1 + n_3)
    )


def _mde_size_dual_control(sigma_sq_C1: float, sigma_sq_I1: float,
                           sigma_sq_C2: float, sigma_sq_I2: float,
                           sigma_sq_C3: float, sigma_sq_Iphi: float, sigma_sq_Ipsi: float,
                           n_1: int, n_2: int, n_3:int , z: float, **kwargs) -> float:
    return(
        2 * z *
        np.sqrt(
            (n_1 * (sigma_sq_C1 + sigma_sq_I1) + n_3 * (sigma_sq_C3 + sigma_sq_Iphi)) / (n_1 + n_3) ** 2 +
            (n_2 * (sigma_sq_C2 + sigma_sq_I2) + n_3 * (sigma_sq_C3 + sigma_sq_Ipsi)) / (n_2 + n_3) ** 2
        )
    )


class ExperimentDesign(ABC):
    def __init__(self, n_0, n_1, n_2, n_3, alpha, pi_min):
        if not(all([(0 <= n) for n in [n_0, n_1, n_2, n_3]])):
            raise ValueError("Sample counts (n) must be greater or equal to 0")

        if not(all([(0 <= p <= 1) for p in [alpha, pi_min]])):
            raise ValueError("Probabilities (alpha/pi_min) must be between 0 and 1")

        self.n_0 = n_0
        self.n_1 = n_1
        self.n_2 = n_2
        self.n_3 = n_3
        self.alpha = alpha
        self.pi_min = pi_min
        self.z = norm.ppf(1 - alpha/2.0) - norm.ppf(1 - pi_min)

    @abstractmethod
    def theoretical_actual_effect(self):
        pass

    @abstractmethod
    def theoretical_mde_size(self):
        pass

    @abstractmethod
    def get_group_a_samples(self) -> np.array:
        """
        Obtain sample responses from analysis group A
        :return:
        """
        pass

    @abstractmethod
    def get_group_b_samples_by_spec(self) -> np.array:
        """
        Obtain sample responses from analysis group B, if they are distributed
        with parameters specified by the experiment design.
        :return:
        """
        pass

    @abstractmethod
    def get_group_b_samples_by_effect(self, effect: float) -> np.array:
        """
        Obtain sample responses from analysis group B, if the group differ in
        metric to analysis group A by th effect size specified in `effect`.
        :param effect: effect size
        :return:
        """
        pass

    @abstractmethod
    def get_actual_effect_sample(self):
        pass

    @abstractmethod
    def get_mde_size_sample(self):
        pass

    def _simulated_power(self, effect: float,
                         null_critical_value: float,
                         n_alt_metric_samples: int, **kwargs) -> float:
        """
        Provide a statistical power sample under a specific effect size. Assumes one-sided statistical tests.
        Only works well for two-sided statistical tests if `effect` > `null_critical_value`.
        :param effect: Size of the effect
        :param null_critical_value: The critical value under H_0
        :param n_alt_metric_samples: Number of samples to be drawn to determine the metric distribution under H_1
        :param kwargs:
        :return: Simulated statistical power
        """
        if effect < null_critical_value:
            warnings.warn("This function generates statistical power samples assuming a one-sided test. "
                          "Power samples for two-sided test from this function are only representative when "
                          "`effect` > `null_critical_value`.", UserWarning)

        alt_metric_samples = []

        for sample in range(0, n_alt_metric_samples):
            group_a_responses = self.get_group_a_samples()
            group_b_responses_alt = self.get_group_b_samples_by_effect(effect=effect)
            alt_metric_samples.append(
                np.mean(group_b_responses_alt) - np.mean(group_a_responses))

        power = 1 - (percentileofscore(alt_metric_samples, null_critical_value) / 100)

        return power

    def _mde_size_sample(self, n_null_metric_samples: int = 2000, n_alt_metric_samples: int = 400):
        null_metric_samples = []

        # Metric distribution under null hypothesis
        for sample in range(0, n_null_metric_samples):
            group_a_responses = self.get_group_a_samples()
            group_b_responses_null = self.get_group_b_samples_by_effect(effect=0)

            null_metric_samples.append(np.mean(group_b_responses_null) - np.mean(group_a_responses))

        null_critical_value = np.percentile(null_metric_samples, (1 - self.alpha / 2) * 100)

        # Speeding up the search by bounding the search space more efficiently
        indicative_mde = (
                null_critical_value / norm.ppf(1 - (self.alpha / 2)) *
                (norm.ppf(1 - (self.alpha / 2)) - norm.ppf(1 - self.pi_min)))
        search_lbound = indicative_mde * 0.7
        search_ubound = indicative_mde * 1.5

        # Let the root finding tolerance to be around 0.05% of the group A mean (in absolute value)
        # With enough bootstrap samples this error would be further reduced
        mde_search_tol = np.abs(np.mean(self.get_group_a_samples())) * 0.0005

        # Extrapolation sometimes give extremely large (both +ve/-ve) values
        # due to small denominator, the while loop guards against that
        mde_sample = 2 * search_ubound
        while np.abs(mde_sample) > search_ubound:
            mde_sample = noisyopt.bisect(
                noisyopt.AveragedFunction(
                    lambda x: self._simulated_power(x, null_critical_value, n_alt_metric_samples) - self.pi_min,
                    N=20),
                search_lbound, search_ubound,
                xtol=mde_search_tol, errorcontrol=True,
                testkwargs={'alpha': 0.01, 'force': True, 'eps': 0.001, 'maxN': 160},
                outside='extrapolate', ascending=True, disp=False)

        return mde_sample


class NormalResponseED(ExperimentDesign):
    def __init__(self, mu_C0: float, mu_C1: float, mu_I1: float, mu_C2: float, mu_I2: float,
                 mu_C3: float, mu_Iphi: float, mu_Ipsi: float,
                 sigma_sq_C0: float, sigma_sq_C1: float, sigma_sq_I1: float, sigma_sq_C2: float,
                 sigma_sq_I2: float, sigma_sq_C3: float, sigma_sq_Iphi: float, sigma_sq_Ipsi: float,
                 n_0: int, n_1: int, n_2: int, n_3: int, alpha: float, pi_min: float, **kwargs):
        if not(all([(0 < sigma_sq) for sigma_sq in [sigma_sq_C0, sigma_sq_C1, sigma_sq_I1, sigma_sq_C2,
                                                    sigma_sq_I2, sigma_sq_C3, sigma_sq_Iphi, sigma_sq_Ipsi]])):
            raise ValueError("Variances (sigma_sq) must be greater than 0")

        super().__init__(n_0, n_1, n_2, n_3, alpha, pi_min)
        self.mu_C0 = mu_C0
        self.mu_C1 = mu_C1
        self.mu_I1 = mu_I1
        self.mu_C2 = mu_C2
        self.mu_I2 = mu_I2
        self.mu_C3 = mu_C3
        self.mu_Iphi = mu_Iphi
        self.mu_Ipsi = mu_Ipsi
        self.sigma_sq_C0 = sigma_sq_C0
        self.sigma_sq_C1 = sigma_sq_C1
        self.sigma_sq_I1 = sigma_sq_I1
        self.sigma_sq_C2 = sigma_sq_C2
        self.sigma_sq_I2 = sigma_sq_I2
        self.sigma_sq_C3 = sigma_sq_C3
        self.sigma_sq_Iphi = sigma_sq_Iphi
        self.sigma_sq_Ipsi = sigma_sq_Ipsi


class IntersectionOnlyNRED(NormalResponseED):
    def __init__(self, mu_C0: float, mu_C1: float, mu_I1: float, mu_C2: float, mu_I2: float,
                 mu_C3: float, mu_Iphi: float, mu_Ipsi: float,
                 sigma_sq_C0: float, sigma_sq_C1: float, sigma_sq_I1: float, sigma_sq_C2: float,
                 sigma_sq_I2: float, sigma_sq_C3: float, sigma_sq_Iphi: float, sigma_sq_Ipsi: float,
                 n_0: int, n_1: int, n_2: int, n_3: int, alpha: float, pi_min: float, **kwargs):
        super().__init__(mu_C0, mu_C1, mu_I1, mu_C2, mu_I2, mu_C3, mu_Iphi, mu_Ipsi,
                         sigma_sq_C0, sigma_sq_C1, sigma_sq_I1, sigma_sq_C2,
                         sigma_sq_I2, sigma_sq_C3, sigma_sq_Iphi, sigma_sq_Ipsi,
                         n_0, n_1, n_2, n_3, alpha, pi_min, **kwargs)

    def theoretical_actual_effect(self):
        return (
            _actual_effect_intersection_only(mu_Iphi=self.mu_Iphi, mu_Ipsi=self.mu_Ipsi)
        )

    def theoretical_mde_size(self):
        return (
            _mde_size_intersection_only(
                sigma_sq_Iphi=self.sigma_sq_Iphi, sigma_sq_Ipsi=self.sigma_sq_Ipsi,
                n_3=self.n_3, z=self.z)
        )

    def get_group_a_samples(self) -> np.array:
        return (
            np.concatenate([
                NormalSampler(n=int(self.n_3 / 2), mu=self.mu_Iphi, sigma_sq=self.sigma_sq_Iphi).get_samples()])
        )

    def get_group_b_samples_by_spec(self) -> np.array:
        return (
            np.concatenate([
                NormalSampler(n=int(self.n_3 / 2), mu=self.mu_Ipsi, sigma_sq=self.sigma_sq_Ipsi).get_samples()])
        )

    def get_group_b_samples_by_effect(self, effect: float) -> np.array:
        # Here we only vary the effect, but keep the variance of analysis group B as the same
        return (
            np.concatenate([
                NormalSampler(n=int(self.n_3 / 2), mu=self.mu_Iphi + effect,
                              sigma_sq=self.sigma_sq_Ipsi).get_samples()])
        )

    def get_actual_effect_sample(self):
        # Get simulated responses from analysis group A and B
        group_a_responses = self.get_group_a_samples()
        group_b_responses = self.get_group_b_samples_by_spec()

        # The actual effect is the mean of group B minus mean of group A
        return np.mean(group_b_responses) - np.mean(group_a_responses)

    def get_mde_size_sample(self):
        return self._mde_size_sample()


class AllSampleNRED(NormalResponseED):
    def __init__(self, mu_C0: float, mu_C1: float, mu_I1: float, mu_C2: float, mu_I2: float,
                 mu_C3: float, mu_Iphi: float, mu_Ipsi: float,
                 sigma_sq_C0: float, sigma_sq_C1: float, sigma_sq_I1: float, sigma_sq_C2: float,
                 sigma_sq_I2: float, sigma_sq_C3: float, sigma_sq_Iphi: float, sigma_sq_Ipsi: float,
                 n_0: int, n_1: int, n_2: int, n_3: int, alpha: float, pi_min: float, **kwargs):
        super().__init__(mu_C0, mu_C1, mu_I1, mu_C2, mu_I2, mu_C3, mu_Iphi, mu_Ipsi,
                         sigma_sq_C0, sigma_sq_C1, sigma_sq_I1, sigma_sq_C2,
                         sigma_sq_I2, sigma_sq_C3, sigma_sq_Iphi, sigma_sq_Ipsi,
                         n_0, n_1, n_2, n_3, alpha, pi_min, **kwargs)

    def theoretical_actual_effect(self):
        return (
            _actual_effect_all_sample(mu_C1=self.mu_C1, mu_I1=self.mu_I1, mu_C2=self.mu_C2, mu_I2=self.mu_I2,
                                      mu_Iphi=self.mu_Iphi, mu_Ipsi=self.mu_Ipsi,
                                      n_0=self.n_0, n_1=self.n_1, n_2=self.n_2, n_3=self.n_3)
        )

    def theoretical_mde_size(self):
        return (
            _mde_size_all_sample(
                sigma_sq_C0=self.sigma_sq_C0, sigma_sq_C1=self.sigma_sq_C1, sigma_sq_I1=self.sigma_sq_I1,
                sigma_sq_C2=self.sigma_sq_C2, sigma_sq_I2=self.sigma_sq_I2,
                sigma_sq_Iphi=self.sigma_sq_Iphi, sigma_sq_Ipsi=self.sigma_sq_Ipsi,
                n_0=self.n_0, n_1=self.n_1, n_2=self.n_2, n_3=self.n_3, z=self.z)
        )

    def get_group_a_samples(self) -> np.array:
        return (
            np.concatenate([
                NormalSampler(n=int(self.n_0 / 2), mu=self.mu_C0, sigma_sq=self.sigma_sq_C0).get_samples(),
                NormalSampler(n=int(self.n_1 / 2), mu=self.mu_I1, sigma_sq=self.sigma_sq_I1).get_samples(),
                NormalSampler(n=int(self.n_2 / 2), mu=self.mu_C2, sigma_sq=self.sigma_sq_C2).get_samples(),
                NormalSampler(n=int(self.n_3 / 2), mu=self.mu_Iphi, sigma_sq=self.sigma_sq_Iphi).get_samples()])
        )

    def get_group_b_samples_by_spec(self) -> np.array:
        return (
            np.concatenate([
                NormalSampler(n=int(self.n_0 / 2), mu=self.mu_C0, sigma_sq=self.sigma_sq_C0).get_samples(),
                NormalSampler(n=int(self.n_1 / 2), mu=self.mu_C1, sigma_sq=self.sigma_sq_C1).get_samples(),
                NormalSampler(n=int(self.n_2 / 2), mu=self.mu_I2, sigma_sq=self.sigma_sq_I2).get_samples(),
                NormalSampler(n=int(self.n_3 / 2), mu=self.mu_Ipsi, sigma_sq=self.sigma_sq_Ipsi).get_samples()])
        )

    def get_group_b_samples_by_effect(self, effect: float) -> np.array:
        # Here we only vary the effect, but keep the variance of analysis group B as the same
        return (
            np.concatenate([
                NormalSampler(n=int(self.n_0 / 2), mu=self.mu_C0 + effect, sigma_sq=self.sigma_sq_C0).get_samples(),
                NormalSampler(n=int(self.n_1 / 2), mu=self.mu_I1 + effect, sigma_sq=self.sigma_sq_C1).get_samples(),
                NormalSampler(n=int(self.n_2 / 2), mu=self.mu_C2 + effect, sigma_sq=self.sigma_sq_I2).get_samples(),
                NormalSampler(n=int(self.n_3 / 2), mu=self.mu_Iphi + effect,
                              sigma_sq=self.sigma_sq_Ipsi).get_samples()])
        )

    def get_actual_effect_sample(self):
        # Get simulated responses from analysis group A and B
        group_a_responses = self.get_group_a_samples()
        group_b_responses = self.get_group_b_samples_by_spec()

        # The actual effect is the mean of group B minus mean of group A
        return np.mean(group_b_responses) - np.mean(group_a_responses)

    def get_mde_size_sample(self):
        return self._mde_size_sample()


class QualifiedOnlyNRED(NormalResponseED):
    def __init__(self, mu_C0: float, mu_C1: float, mu_I1: float, mu_C2: float, mu_I2: float,
                 mu_C3: float, mu_Iphi: float, mu_Ipsi: float,
                 sigma_sq_C0: float, sigma_sq_C1: float, sigma_sq_I1: float, sigma_sq_C2: float,
                 sigma_sq_I2: float, sigma_sq_C3: float, sigma_sq_Iphi: float, sigma_sq_Ipsi: float,
                 n_0: int, n_1: int, n_2: int, n_3: int, alpha: float, pi_min: float, **kwargs):
        super().__init__(mu_C0, mu_C1, mu_I1, mu_C2, mu_I2, mu_C3, mu_Iphi, mu_Ipsi,
                         sigma_sq_C0, sigma_sq_C1, sigma_sq_I1, sigma_sq_C2,
                         sigma_sq_I2, sigma_sq_C3, sigma_sq_Iphi, sigma_sq_Ipsi,
                         n_0, n_1, n_2, n_3, alpha, pi_min, **kwargs)

    def theoretical_actual_effect(self):
        return (
            _actual_effect_qualified_only(mu_C1=self.mu_C1, mu_I1=self.mu_I1, mu_C2=self.mu_C2, mu_I2=self.mu_I2,
                                          mu_Iphi=self.mu_Iphi, mu_Ipsi=self.mu_Ipsi,
                                          n_1=self.n_1, n_2=self.n_2, n_3=self.n_3)
        )

    def theoretical_mde_size(self):
        return (
            _mde_size_qualified_only(
                sigma_sq_C1=self.sigma_sq_C1, sigma_sq_I1=self.sigma_sq_I1,
                sigma_sq_C2=self.sigma_sq_C2, sigma_sq_I2=self.sigma_sq_I2,
                sigma_sq_Iphi=self.sigma_sq_Iphi, sigma_sq_Ipsi=self.sigma_sq_Ipsi,
                n_1=self.n_1, n_2=self.n_2, n_3=self.n_3, z=self.z)
        )

    def get_group_a_samples(self) -> np.array:
        return (
            np.concatenate([
                NormalSampler(n=int(self.n_1 / 2), mu=self.mu_I1, sigma_sq=self.sigma_sq_I1).get_samples(),
                NormalSampler(n=int(self.n_2 / 2), mu=self.mu_C2, sigma_sq=self.sigma_sq_C2).get_samples(),
                NormalSampler(n=int(self.n_3 / 2), mu=self.mu_Iphi, sigma_sq=self.sigma_sq_Iphi).get_samples()])
        )

    def get_group_b_samples_by_spec(self) -> np.array:
        return (
            np.concatenate([
                NormalSampler(n=int(self.n_1 / 2), mu=self.mu_C1, sigma_sq=self.sigma_sq_C1).get_samples(),
                NormalSampler(n=int(self.n_2 / 2), mu=self.mu_I2, sigma_sq=self.sigma_sq_I2).get_samples(),
                NormalSampler(n=int(self.n_3 / 2), mu=self.mu_Ipsi, sigma_sq=self.sigma_sq_Ipsi).get_samples()])
        )

    def get_group_b_samples_by_effect(self, effect: float) -> np.array:
        # Here we only vary the effect, but keep the variance of analysis group B as the same
        return (
            np.concatenate([
                NormalSampler(n=int(self.n_1 / 2), mu=self.mu_I1 + effect, sigma_sq=self.sigma_sq_C1).get_samples(),
                NormalSampler(n=int(self.n_2 / 2), mu=self.mu_C2 + effect, sigma_sq=self.sigma_sq_I2).get_samples(),
                NormalSampler(n=int(self.n_3 / 2), mu=self.mu_Iphi + effect,
                              sigma_sq=self.sigma_sq_Ipsi).get_samples()])
        )

    def get_actual_effect_sample(self):
        # Get simulated responses from analysis group A and B
        group_a_responses = self.get_group_a_samples()
        group_b_responses = self.get_group_b_samples_by_spec()

        # The actual effect is the mean of group B minus mean of group A
        return np.mean(group_b_responses) - np.mean(group_a_responses)

    def get_mde_size_sample(self):
        return self._mde_size_sample()


class DualControlNRED(NormalResponseED):
    def __init__(self, mu_C0: float, mu_C1: float, mu_I1: float, mu_C2: float, mu_I2: float,
                 mu_C3: float, mu_Iphi: float, mu_Ipsi: float,
                 sigma_sq_C0: float, sigma_sq_C1: float, sigma_sq_I1: float, sigma_sq_C2: float,
                 sigma_sq_I2: float, sigma_sq_C3: float, sigma_sq_Iphi: float, sigma_sq_Ipsi: float,
                 n_0: int, n_1: int, n_2: int, n_3: int, alpha: float, pi_min: float, **kwargs):
        super().__init__(mu_C0, mu_C1, mu_I1, mu_C2, mu_I2, mu_C3, mu_Iphi, mu_Ipsi,
                         sigma_sq_C0, sigma_sq_C1, sigma_sq_I1, sigma_sq_C2,
                         sigma_sq_I2, sigma_sq_C3, sigma_sq_Iphi, sigma_sq_Ipsi,
                         n_0, n_1, n_2, n_3, alpha, pi_min, **kwargs)

    def theoretical_actual_effect(self):
        return (
            _actual_effect_dual_control(mu_C1=self.mu_C1, mu_I1=self.mu_I1, mu_C2=self.mu_C2, mu_I2=self.mu_I2,
                                        mu_C3=self.mu_C3, mu_Iphi=self.mu_Iphi, mu_Ipsi=self.mu_Ipsi,
                                        n_1=self.n_1, n_2=self.n_2, n_3=self.n_3)
        )

    def theoretical_mde_size(self):
        return (
            _mde_size_dual_control(
                sigma_sq_C1=self.sigma_sq_C1, sigma_sq_I1=self.sigma_sq_I1,
                sigma_sq_C2=self.sigma_sq_C2, sigma_sq_I2=self.sigma_sq_I2,
                sigma_sq_C3=self.sigma_sq_C3, sigma_sq_Iphi=self.sigma_sq_Iphi, sigma_sq_Ipsi=self.sigma_sq_Ipsi,
                n_1=self.n_1, n_2=self.n_2, n_3=self.n_3, z=self.z)
        )

    def get_group_a1_samples(self) -> np.array:
        return (
            np.concatenate([
                NormalSampler(n=int(self.n_1 / 4), mu=self.mu_C1, sigma_sq=self.sigma_sq_C1).get_samples(),
                NormalSampler(n=int(self.n_3 / 4), mu=self.mu_C3, sigma_sq=self.sigma_sq_C3).get_samples()])
        )

    def get_group_a2_samples(self) -> np.array:
        return (
            np.concatenate([
                NormalSampler(n=int(self.n_1 / 4), mu=self.mu_I1, sigma_sq=self.sigma_sq_I1).get_samples(),
                NormalSampler(n=int(self.n_3 / 4), mu=self.mu_Iphi, sigma_sq=self.sigma_sq_Iphi).get_samples()])
        )

    def get_group_b1_samples(self) -> np.array:
        return (
            np.concatenate([
                NormalSampler(n=int(self.n_2 / 4), mu=self.mu_C2, sigma_sq=self.sigma_sq_C2).get_samples(),
                NormalSampler(n=int(self.n_3 / 4), mu=self.mu_C3, sigma_sq=self.sigma_sq_C3).get_samples()])
        )

    def get_group_b2_samples(self) -> np.array:
        return (
            np.concatenate([
                NormalSampler(n=int(self.n_2 / 4), mu=self.mu_I2, sigma_sq=self.sigma_sq_I2).get_samples(),
                NormalSampler(n=int(self.n_3 / 4), mu=self.mu_Ipsi, sigma_sq=self.sigma_sq_Ipsi).get_samples()])
        )

    def get_group_b2_samples_by_effect(self, effect: float) -> np.array:
        group_a_effect = (
            (self.n_1 * (self.mu_I1 - self.mu_C1) + self.n_3 * (self.mu_Iphi - self.mu_C3)) / (self.n_1 + self.n_3)
        )

        # For the experiment to see an effect, the effect has to be in excess of what analysis group A achieved
        # Keep the variance of analysis group B2 as the same as that specified
        return (
            np.concatenate([
                NormalSampler(n=int(self.n_2 / 4), mu=self.mu_C2 + group_a_effect + effect,
                              sigma_sq=self.sigma_sq_I2).get_samples(),
                NormalSampler(n=int(self.n_3 / 4), mu=self.mu_C3 + group_a_effect + effect,
                              sigma_sq=self.sigma_sq_Ipsi).get_samples()])
        )

    def get_group_a_samples(self) -> np.array:
        raise NotImplementedError("Dual control experiments are implemented differently.")

    def get_group_b_samples_by_spec(self) -> np.array:
        raise NotImplementedError("Dual control experiments are implemented differently.")

    def get_group_b_samples_by_effect(self, effect: float) -> np.array:
        raise NotImplementedError("Dual control experiments are implemented differently.")

    def get_actual_effect_sample(self):
        # Get simulated responses from all four groups
        group_a1_responses = self.get_group_a1_samples()
        group_a2_responses = self.get_group_a2_samples()
        group_b1_responses = self.get_group_b1_samples()
        group_b2_responses = self.get_group_b2_samples()

        # The actual effect is the difference between (mean of group B2 minus mean of group B1),
        # and (mean of group A2 minus mean of group A1)
        return (
            (np.mean(group_b2_responses) - np.mean(group_b1_responses)) -
            (np.mean(group_a2_responses) - np.mean(group_a1_responses))
        )

    def _simulated_power_dual_control(self, effect: float,
                                      null_critical_value: float, n_alt_metric_samples: int):
        if effect < null_critical_value:
            warnings.warn("This function generates statistical power samples assuming a one-sided test. "
                          "Power samples for two-sided test from this function are only representative when "
                          "`effect` > `null_critical_value`.", UserWarning)

        alt_metric_samples = []
        for sample in range(0, n_alt_metric_samples):
            group_a1_responses = self.get_group_a1_samples()
            group_a2_responses = self.get_group_a2_samples()
            group_b1_responses = self.get_group_b1_samples()
            group_b2_responses_alt = self.get_group_b2_samples_by_effect(effect=effect)

            alt_metric_samples.append(
                (np.mean(group_b2_responses_alt) - np.mean(group_b1_responses)) -
                (np.mean(group_a2_responses) - np.mean(group_a1_responses))
            )

        power = 1 - (percentileofscore(alt_metric_samples, null_critical_value) / 100)

        return power

    def get_mde_size_sample(self):
        n_null_metric_samples = 2000
        n_alt_metric_samples = 400
        null_metric_samples = []

        # Metric distribution under null hypothesis
        for sample in range(0, n_null_metric_samples):
            group_a1_responses = self.get_group_a1_samples()
            group_a2_responses = self.get_group_a2_samples()
            group_b1_responses = self.get_group_b1_samples()
            group_b2_responses_null = self.get_group_b2_samples_by_effect(effect=0)

            null_metric_samples.append(
                (np.mean(group_b2_responses_null) - np.mean(group_b1_responses)) -
                (np.mean(group_a2_responses) - np.mean(group_a1_responses))
            )

        null_critical_value = np.percentile(null_metric_samples, (1 - self.alpha / 2) * 100)

        # Speeding up the search by bounding the search space more efficiently
        indicative_mde = (
                null_critical_value / norm.ppf(1 - (self.alpha / 2)) *
                (norm.ppf(1 - (self.alpha / 2)) - norm.ppf(1 - self.pi_min)))
        search_lbound = indicative_mde * 0.7
        search_ubound = indicative_mde * 1.5

        # Let the root finding tolerance to be around 0.05% of the difference between
        # the group A means (in absolute value)
        # With enough bootstrap samples this error would be further reduced
        mde_search_tol = np.abs(np.mean(self.get_group_a2_samples()) - np.mean(self.get_group_a1_samples())) * 0.0005

        # Extrapolation sometimes give extremely large (both +ve/-ve) values
        # due to small denominator, the while loop guards against that
        mde_sample = 2 * search_ubound
        while np.abs(mde_sample) > search_ubound:
            mde_sample = noisyopt.bisect(
                noisyopt.AveragedFunction(
                    lambda x: self._simulated_power_dual_control(x, null_critical_value, n_alt_metric_samples) -
                              self.pi_min,
                    N=20),
                search_lbound, search_ubound,
                xtol=mde_search_tol, errorcontrol=True,
                testkwargs={'alpha': 0.01, 'force': True, 'eps': 0.001, 'maxN': 160},
                outside='extrapolate', ascending=True, disp=False)

        return mde_sample


class BinaryResponseRateED(ExperimentDesign):
    def __init__(self, p_C0, p_C1, p_I1, p_C2, p_I2, p_C3, p_Iphi, p_Ipsi,
                 n_0, n_1, n_2, n_3, alpha, pi_min):
        if not(all([(0 <= p <= 1) for p in [p_C0, p_C1, p_I1, p_C2, p_I2, p_C3, p_Iphi, p_Ipsi]])):
            raise ValueError("Probabilities (p) must be between 0 and 1")

        super().__init__(n_0, n_1, n_2, n_3, alpha, pi_min)
        self.p_C0 = p_C0
        self.p_C1 = p_C1
        self.p_I1 = p_I1
        self.p_C2 = p_C2
        self.p_I2 = p_I2
        self.p_C3 = p_C3
        self.p_Iphi = p_Iphi
        self.p_Ipsi = p_Ipsi


class AllSampleBRRED(BinaryResponseRateED):
    """
    Experiment design 2 (all-sample) for binary response metrics (e.g. conversion rate)
    """
    def __init__(self, p_C0, p_C1, p_I1, p_C2, p_I2, p_C3, p_Iphi, p_Ipsi,
                 n_0, n_1, n_2, n_3, alpha, pi_min, **kwargs):
        super().__init__(p_C0, p_C1, p_I1, p_C2, p_I2, p_C3, p_Iphi, p_Ipsi,
                         n_0, n_1, n_2, n_3, alpha, pi_min)

    def theoretical_actual_effect(self):
        return(
            _actual_effect_all_sample(mu_C1=self.p_C1, mu_I1=self.p_I1, mu_C2=self.p_C2, mu_I2=self.p_I2,
                                      mu_Iphi=self.p_Iphi, mu_Ipsi=self.p_Ipsi,
                                      n_0=self.n_0, n_1=self.n_1, n_2=self.n_2, n_3=self.n_3)
        )

    def theoretical_mde_size(self):
        sigma_sq_C0 = BinarySampler(n=1, p=self.p_C0).theoretical_variance()
        sigma_sq_C1 = BinarySampler(n=1, p=self.p_C1).theoretical_variance()
        sigma_sq_I1 = BinarySampler(n=1, p=self.p_I1).theoretical_variance()
        sigma_sq_C2 = BinarySampler(n=1, p=self.p_C2).theoretical_variance()
        sigma_sq_I2 = BinarySampler(n=1, p=self.p_I2).theoretical_variance()
        sigma_sq_Iphi = BinarySampler(n=1, p=self.p_Iphi).theoretical_variance()
        sigma_sq_Ipsi = BinarySampler(n=1, p=self.p_Ipsi).theoretical_variance()

        return (
            _mde_size_all_sample(sigma_sq_C0=sigma_sq_C0, sigma_sq_C1=sigma_sq_C1, sigma_sq_I1=sigma_sq_I1,
                                 sigma_sq_C2=sigma_sq_C2, sigma_sq_I2=sigma_sq_I2,
                                 sigma_sq_Iphi=sigma_sq_Iphi, sigma_sq_Ipsi=sigma_sq_Ipsi,
                                 n_0=self.n_0, n_1=self.n_1, n_2=self.n_2, n_3=self.n_3, z=self.z)
        )

    def get_group_a_samples(self) -> np.array:
        return(
            np.concatenate([
                BinarySampler(n=int(self.n_0 / 2), p=self.p_C0).get_samples(),
                BinarySampler(n=int(self.n_1 / 2), p=self.p_I1).get_samples(),
                BinarySampler(n=int(self.n_2 / 2), p=self.p_C2).get_samples(),
                BinarySampler(n=int(self.n_3 / 2), p=self.p_Iphi).get_samples()])
        )

    def get_group_b_samples_by_spec(self) -> np.array:
        return(
            np.concatenate([
                BinarySampler(n=int(self.n_0 / 2), p=self.p_C0).get_samples(),
                BinarySampler(n=int(self.n_1 / 2), p=self.p_C1).get_samples(),
                BinarySampler(n=int(self.n_2 / 2), p=self.p_I2).get_samples(),
                BinarySampler(n=int(self.n_3 / 2), p=self.p_Ipsi).get_samples()])
        )

    def get_group_b_samples_by_effect(self, effect: float) -> np.array:
        return (
            np.concatenate([
                BinarySampler(n=int(self.n_0 / 2), p=self.p_C0 + effect).get_samples(),
                BinarySampler(n=int(self.n_1 / 2), p=self.p_I1 + effect).get_samples(),
                BinarySampler(n=int(self.n_2 / 2), p=self.p_C2 + effect).get_samples(),
                BinarySampler(n=int(self.n_3 / 2), p=self.p_Iphi + effect).get_samples()])
        )

    def get_actual_effect_sample(self):
        def get_actual_effect_sample(self):
            # Get simulated responses from analysis group A and B
            group_a_responses = self.get_group_a_samples()
            group_b_responses = self.get_group_b_samples_by_spec()

            # The actual effect is the mean of group B minus mean of group A
            return np.mean(group_b_responses) - np.mean(group_a_responses)

    def get_mde_size_sample(self):
        return self._mde_size_sample()


class QualifiedOnlyBRRED(BinaryResponseRateED):
    """
    Experiment design 3 (samples who qualify for at least one strategy only)
    for binary response metrics (e.g. conversion rate).
    """
    def __init__(self, p_C0, p_C1, p_I1, p_C2, p_I2, p_C3, p_Iphi, p_Ipsi,
                 n_0, n_1, n_2, n_3, alpha, pi_min, **kwargs):
        super().__init__(p_C0, p_C1, p_I1, p_C2, p_I2, p_C3, p_Iphi, p_Ipsi,
                         n_0, n_1, n_2, n_3, alpha, pi_min)

    def theoretical_actual_effect(self):
        return(
            _actual_effect_qualified_only(
                mu_C1=self.p_C1, mu_I1=self.p_I1, mu_C2=self.p_C2, mu_I2=self.p_I2,
                mu_Iphi=self.p_Iphi, mu_Ipsi=self.p_Ipsi, n_1=self.n_1, n_2=self.n_2, n_3=self.n_3)
        )

    def theoretical_mde_size(self):
        sigma_sq_C1 = BinarySampler(n=1, p=self.p_C1).theoretical_variance()
        sigma_sq_I1 = BinarySampler(n=1, p=self.p_I1).theoretical_variance()
        sigma_sq_C2 = BinarySampler(n=1, p=self.p_C2).theoretical_variance()
        sigma_sq_I2 = BinarySampler(n=1, p=self.p_I2).theoretical_variance()
        sigma_sq_Iphi = BinarySampler(n=1, p=self.p_Iphi).theoretical_variance()
        sigma_sq_Ipsi = BinarySampler(n=1, p=self.p_Ipsi).theoretical_variance()

        return(
            _mde_size_qualified_only(
                sigma_sq_C1=sigma_sq_C1, sigma_sq_I1=sigma_sq_I1,
                sigma_sq_C2=sigma_sq_C2, sigma_sq_I2=sigma_sq_I2,
                sigma_sq_Iphi=sigma_sq_Iphi, sigma_sq_Ipsi=sigma_sq_Ipsi,
                n_1=self.n_1, n_2=self.n_2, n_3=self.n_3, z=self.z)
        )

    def get_group_a_samples(self) -> np.array:
        return(
            np.concatenate([
                BinarySampler(n=int(self.n_1 / 2), p=self.p_I1).get_samples(),
                BinarySampler(n=int(self.n_2 / 2), p=self.p_C2).get_samples(),
                BinarySampler(n=int(self.n_3 / 2), p=self.p_Iphi).get_samples()])
        )

    def get_group_b_samples_by_spec(self) -> np.array:
        return(
            np.concatenate([
                BinarySampler(n=int(self.n_1 / 2), p=self.p_C1).get_samples(),
                BinarySampler(n=int(self.n_2 / 2), p=self.p_I2).get_samples(),
                BinarySampler(n=int(self.n_3 / 2), p=self.p_Ipsi).get_samples()])
        )

    def get_group_b_samples_by_effect(self, effect: float) -> np.array:
        return (
            np.concatenate([
                BinarySampler(n=int(self.n_1 / 2), p=self.p_I1 + effect).get_samples(),
                BinarySampler(n=int(self.n_2 / 2), p=self.p_C2 + effect).get_samples(),
                BinarySampler(n=int(self.n_3 / 2), p=self.p_Iphi + effect).get_samples()])
        )

    def get_actual_effect_sample(self):
        # Get simulated responses from analysis group A and B
        group_a_responses = self.get_group_a_samples()
        group_b_responses = self.get_group_b_samples_by_spec()

        # The actual effect is the mean of group B minus mean of group A
        return np.mean(group_b_responses) - np.mean(group_a_responses)

    def get_mde_size_sample(self) -> float:
        return self._mde_size_sample()
