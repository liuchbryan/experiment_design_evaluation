from abc import ABC, abstractmethod
from pedeval.sampler import BinarySampler

import numpy as np
from scipy.stats import norm, percentileofscore
import noisyopt
import warnings


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
    def get_actual_effect_sample(self):
        pass

    def get_mde_size_sample(self):
        pass


def _simulated_power_BRR(effect: float, group_a_n: int, group_a_p: float,
                         null_critical_value: float,
                         n_alt_metric_samples: int, **kwargs) -> float:
    """
    Provide a statistical power sample under a specific effect size. Assumes one-sided statistical tests.
    Only works well for two-sided statistical tests if `effect` > `null_critical_value`.
    :param effect: Size of the effect
    :param group_a_n: The number of samples in analysis group A
                      (= number of samples in analysis group B to maximise the test power)
    :param group_a_p: The probability (binary response rate) of both groups under H_0
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
        group_a_responses = BinarySampler(n=group_a_n, p=group_a_p).get_samples()
        group_b_responses_alt = (
            BinarySampler(n=group_a_n, p=group_a_p + effect).get_samples())
        alt_metric_samples.append(
            np.mean(group_b_responses_alt) - np.mean(group_a_responses))

    power = 1 - (percentileofscore(alt_metric_samples, null_critical_value) / 100)

    return power


class BinaryResponseRateED(ExperimentDesign):
    def __init__(self, p_C0, p_C1, p_I1, p_C2, p_I2, p_C3, p_Iphi, p_Ipsi,
                 n_0, n_1, n_2, n_3, alpha, pi_min):
        if not(all([(0 <= p <= 1) for p in [p_C0, p_C1, p_I1, p_C2, p_C3, p_Iphi, p_Ipsi]])):
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

    @abstractmethod
    def theoretical_actual_effect(self):
        pass

    @abstractmethod
    def theoretical_mde_size(self):
        pass

    @abstractmethod
    def get_actual_effect_sample(self):
        pass

    @abstractmethod
    def get_mde_size_sample(self):
        pass

    def _mde_size_sample(self, group_a_n: int, group_a_p: float,
                         n_null_metric_samples: int = 2000, n_alt_metric_samples: int = 500):
        null_metric_samples = []

        # Metric distribution under null hypothesis
        for sample in range(0, n_null_metric_samples):
            group_a_responses = BinarySampler(n=group_a_n, p=group_a_p).get_samples()
            group_b_responses_null = BinarySampler(n=group_a_n, p=group_a_p).get_samples()

            null_metric_samples.append(np.mean(group_b_responses_null) - np.mean(group_a_responses))

        null_critical_value = np.percentile(null_metric_samples, (1 - self.alpha / 2) * 100)

        # Speeding up the search by bounding the search space more efficiently
        indicative_mde = (
                null_critical_value / norm.ppf(1 - (self.alpha / 2)) *
                (norm.ppf(1 - (self.alpha / 2)) - norm.ppf(1 - self.pi_min)))
        search_lbound = indicative_mde * 0.8
        search_ubound = indicative_mde * 1.25

        mde_sample = noisyopt.bisect(
            noisyopt.AveragedFunction(
                lambda x: _simulated_power_BRR(x, group_a_n, group_a_p,
                                               null_critical_value, n_alt_metric_samples) - self.pi_min,
                N=20),
            search_lbound, search_ubound,
            xtol=0.00005, errorcontrol=True,
            testkwargs={'alpha': 0.01, 'eps': 0.0003, 'maxN': 320},
            outside='extrapolate', ascending=True, disp=False)

        return mde_sample


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

        return(
            _mde_size_all_sample(sigma_sq_C0=sigma_sq_C0, sigma_sq_C1=sigma_sq_C1, sigma_sq_I1=sigma_sq_I1,
                                 sigma_sq_C2=sigma_sq_C2, sigma_sq_I2=sigma_sq_I2,
                                 sigma_sq_Iphi=sigma_sq_Iphi, sigma_sq_Ipsi=sigma_sq_Ipsi,
                                 n_0=self.n_0, n_1=self.n_1, n_2=self.n_2, n_3=self.n_3, z=self.z)
        )

    def get_actual_effect_sample(self):
        # Get simulated responses from analysis group A
        group_A_responses = np.concatenate([
            BinarySampler(n=int(self.n_0 / 2), p=self.p_C0).get_samples(),
            BinarySampler(n=int(self.n_1 / 2), p=self.p_I1).get_samples(),
            BinarySampler(n=int(self.n_2 / 2), p=self.p_C2).get_samples(),
            BinarySampler(n=int(self.n_3 / 2), p=self.p_Iphi).get_samples()])

        # Get simulated responses from analysis group B
        group_B_responses = np.concatenate([
            BinarySampler(n=int(self.n_0 / 2), p=self.p_C0).get_samples(),
            BinarySampler(n=int(self.n_1 / 2), p=self.p_C1).get_samples(),
            BinarySampler(n=int(self.n_2 / 2), p=self.p_I2).get_samples(),
            BinarySampler(n=int(self.n_3 / 2), p=self.p_Ipsi).get_samples()])

        # The actual effect is the mean of group B minus mean of group A
        return np.mean(group_B_responses) - np.mean(group_A_responses)

    def get_mde_size_sample(self):
        n_null_metric_samples = 2000
        n_alt_metric_samples = 500

        # For binary response, it is equivalent to either:
        # 1. Sample four groups with sizes n_0, n_1, n_2, n_3 and prob p_0, p_1, p_2, p_3, or
        # 2. Sample one group with size (n_0 + n_1 + n_2 + n_3) and prob
        #    (n_0p_0 + n_1p_1 + n_2p_2 + n_3p_3) / (n_0 + n_1 + n_2 + n_3)
        group_a_n = int(self.n_0 / 2) + int(self.n_1 / 2) + int(self.n_2 / 2) + int(self.n_3 / 2)
        group_a_p = (
                (self.n_0 * self.p_C0 + self.n_1 * self.p_I1 + self.n_2 * self.p_C2 + self.n_3 * self.p_Iphi) /
                (self.n_0 + self.n_1 + self.n_2 + self.n_3)
        )

        return self._mde_size_sample(group_a_n=group_a_n, group_a_p=group_a_p,
                                     n_null_metric_samples=n_null_metric_samples,
                                     n_alt_metric_samples=n_alt_metric_samples)


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

    def get_actual_effect_sample(self):
        # Get simulated responses from analysis group A
        group_A_responses = np.concatenate([
            BinarySampler(n=int(self.n_1 / 2), p=self.p_I1).get_samples(),
            BinarySampler(n=int(self.n_2 / 2), p=self.p_C2).get_samples(),
            BinarySampler(n=int(self.n_3 / 2), p=self.p_Iphi).get_samples()])

        # Get simulated responses from analysis group B
        group_B_responses = np.concatenate([
            BinarySampler(n=int(self.n_1 / 2), p=self.p_C1).get_samples(),
            BinarySampler(n=int(self.n_2 / 2), p=self.p_I2).get_samples(),
            BinarySampler(n=int(self.n_3 / 2), p=self.p_Ipsi).get_samples()])

        # The actual effect is the mean of group B minus mean of group A
        return np.mean(group_B_responses) - np.mean(group_A_responses)

    def get_mde_size_sample(self) -> float:
        n_null_metric_samples = 2000
        n_alt_metric_samples = 500

        # For binary response, it is equivalent to either:
        # 1. Sample three groups with sizes n_1, n_2, n_3 and prob p_1, p_2, p_3, or
        # 2. Sample one group with size (n_1 + n_2 + n_3) and prob
        #    (n_1p_1 + n_2p_2 + n_3p_3) / (n_1 + n_2 + n_3)
        group_a_n = int(self.n_1 / 2) + int(self.n_2 / 2) + int(self.n_3 / 2)
        group_a_p = (
                (self.n_1 * self.p_I1 + self.n_2 * self.p_C2 + self.n_3 * self.p_Iphi) /
                (self.n_1 + self.n_2 + self.n_3)
        )

        return self._mde_size_sample(group_a_n=group_a_n, group_a_p=group_a_p,
                                     n_null_metric_samples=n_null_metric_samples,
                                     n_alt_metric_samples=n_alt_metric_samples)

