from pedeval import sampler as s
import pytest
import numpy as np


class TestNormalSampler:
    def test_function_gives_right_number_of_samples(self):
        mean = 0
        variance = 1
        size = 10
        assert len(s.NormalSampler(n=size, mu=mean, sigma_sq=variance).get_samples()) == size

    def test_function_gives_samples_with_right_variance(self):
        # The main goal is to assert that we are sampling normal variables with variance 10,
        # but not a standard deviation of 10
        mean = 0
        variance = 10
        size = 100

        # This test should pass 99% of the times as the bounds are the 0.005 and 0.995
        # quantiles of a chi-sq distribution with 99 d.f.
        test_statistic = (
                (size - 1) * np.var(s.NormalSampler(n=size, mu=mean, sigma_sq=variance).get_samples()) / variance)
        assert (66.51 < test_statistic < 138.987)