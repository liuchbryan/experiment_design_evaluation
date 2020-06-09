from abc import ABC, abstractmethod
import numpy as np


class Sampler(ABC):

    def __init__(self):
        pass

    @abstractmethod
    def theoretical_mean(self):
        pass

    @abstractmethod
    def theoretical_variance(self):
        pass

    @abstractmethod
    def get_samples(self):
        pass


class BinarySampler(Sampler):
    def __init__(self, n: int, p: float):
        super().__init__()
        self.n = n
        self.p = p

    def theoretical_mean(self):
        return self.p

    def theoretical_variance(self):
        return self.p * (1 - self.p)

    def get_samples(self) -> np.array:
        return np.random.binomial(n=1, p=self.p, size=self.n)


class SpendSampler(Sampler):
    def __init__(self, n: int, p: float, mu: float, sigma_sq: float):
        super().__init__()
        self.n = n
        self.p = p
        self.mu = mu
        self.sigma_sq = sigma_sq

    def theoretical_mean(self):
        return self.p * np.exp(self.mu + self.sigma_sq / 2)

    def theoretical_variance(self):
        return (
                self.p * (np.exp(self.sigma_sq) - 1) * np.exp(2 * self.mu + self.sigma_sq) +
                (1 - self.p) / self.p * self.theoretical_mean() ** 2)

    def get_samples(self):
        return (
            np.random.binomial(n=1, p=self.p, size=self.n) *
            np.random.lognormal(mean=self.mu, sigma=np.sqrt(self.sigma_sq), size=self.n)
        )
