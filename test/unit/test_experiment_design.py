from pedeval import experiment_design as ed
import pytest
import numpy as np


class TestExperimentDesignInit:
    # As ExperimentDesign is an abstract class, we use AllSampleBRRED (first concrete class) as proxy
    def test_rejects_non_probability_alpha(self):
        with pytest.raises(ValueError):
            ed.AllSampleBRRED(p_C0=0.1, p_C1=0.2, p_I1=0.3, p_C2=0.4, p_I2=0.5,
                              p_C3=0.6, p_Iphi=0.7, p_Ipsi=0.8,
                              n_0=1, n_1=2, n_2=3, n_3=4, alpha=-0.05, pi_min=0.8)
            ed.AllSampleBRRED(p_C0=0.1, p_C1=0.2, p_I1=0.3, p_C2=0.4, p_I2=0.5,
                              p_C3=0.6, p_Iphi=0.7, p_Ipsi=0.8,
                              n_0=1, n_1=2, n_2=3, n_3=4, alpha=1.05, pi_min=0.8)

    def test_rejects_non_probability_pi_min(self):
        with pytest.raises(ValueError):
            ed.AllSampleBRRED(p_C0=0.1, p_C1=0.2, p_I1=0.3, p_C2=0.4, p_I2=0.5,
                              p_C3=0.6, p_Iphi=0.7, p_Ipsi=0.8,
                              n_0=1, n_1=2, n_2=3, n_3=4, alpha=0.05, pi_min=-0.0001)
            ed.AllSampleBRRED(p_C0=0.1, p_C1=0.2, p_I1=0.3, p_C2=0.4, p_I2=0.5,
                              p_C3=0.6, p_Iphi=0.7, p_Ipsi=0.8,
                              n_0=1, n_1=2, n_2=3, n_3=4, alpha=0.05, pi_min=1.0001)

    def test_rejects_negative_ns(self):
        with pytest.raises(ValueError):
            # Cycle through all `n`s and make each of them negative
            ed.AllSampleBRRED(p_C0=0.1, p_C1=0.2, p_I1=0.3, p_C2=0.4, p_I2=0.5,
                              p_C3=0.6, p_Iphi=0.7, p_Ipsi=0.8,
                              n_0=-1, n_1=2, n_2=3, n_3=4, alpha=0.05, pi_min=0.8)
            ed.AllSampleBRRED(p_C0=0.1, p_C1=0.2, p_I1=0.3, p_C2=0.4, p_I2=0.5,
                              p_C3=0.6, p_Iphi=0.7, p_Ipsi=0.8,
                              n_0=1, n_1=-2, n_2=3, n_3=4, alpha=0.05, pi_min=0.8)
            ed.AllSampleBRRED(p_C0=0.1, p_C1=0.2, p_I1=0.3, p_C2=0.4, p_I2=0.5,
                              p_C3=0.6, p_Iphi=0.7, p_Ipsi=0.8,
                              n_0=1, n_1=2, n_2=-3, n_3=4, alpha=0.05, pi_min=0.8)
            ed.AllSampleBRRED(p_C0=0.1, p_C1=0.2, p_I1=0.3, p_C2=0.4, p_I2=0.5,
                              p_C3=0.6, p_Iphi=0.7, p_Ipsi=0.8,
                              n_0=1, n_1=2, n_2=3, n_3=-4, alpha=0.05, pi_min=0.8)

    def test_calculates_z_correctly(self):
        design = ed.AllSampleBRRED(p_C0=0.1, p_C1=0.2, p_I1=0.3, p_C2=0.4, p_I2=0.5,
                                   p_C3=0.6, p_Iphi=0.7, p_Ipsi=0.8,
                                   n_0=1, n_1=2, n_2=3, n_3=4, alpha=0.05, pi_min=0.8)
        assert design.z == pytest.approx(1.96 - (-0.84), 0.01)


class TestSimulatedPowerBRR:
    def test_function_gives_expected_value(self):
        # If the effect is equal to the critical value, the power should be around 50%
        n = 1000
        p = 0.5
        crit_val = 1.96 * np.sqrt(p * (1 - p) / n)
        assert(
            ed.AllSampleBRRED(p_C0=0, p_C1=0, p_I1=0, p_C2=0, p_I2=0,
                              p_C3=p, p_Iphi=p, p_Ipsi=p,
                              n_0=0, n_1=0, n_2=0, n_3=n, alpha=0.05, pi_min=0.8)
            ._simulated_power(effect=crit_val, null_critical_value=crit_val, n_alt_metric_samples=1000) ==
            pytest.approx(0.5, 0.1)      # 0.5 +/- 0.05
        )

    def test_function_gives_expected_value_high_power(self):
        # If the effect is equal to (z_{1-alpha/2} - z_{1-0.8}) * SE, the power should be around 80%
        n = 1000
        p = 0.5
        crit_val = 1.96 * np.sqrt(p * (1 - p) / n)
        effect = (1.96 - (-0.84)) * np.sqrt(p * (1 - p) / n)
        assert(
            ed.AllSampleBRRED(p_C0=0, p_C1=0, p_I1=0, p_C2=0, p_I2=0,
                              p_C3=p, p_Iphi=p, p_Ipsi=p,
                              n_0=0, n_1=0, n_2=0, n_3=n, alpha=0.05, pi_min=0.8)
            ._simulated_power(effect=effect, null_critical_value=crit_val, n_alt_metric_samples=1000) ==
            pytest.approx(0.8, 0.5)      # 0.8 +/- 0.1
        )

    def test_function_warns_user_for_low_power_samples(self):
        with pytest.warns(UserWarning):
            # If the effect is equal to (z_{1-alpha/2} - z_{1-0.2}) * SE, the power should be around 20%
            n = 1000
            p = 0.5
            crit_val = 1.96 * np.sqrt(p * (1 - p) / n)
            effect = (1.96 - 0.84) * np.sqrt(p * (1 - p) / n)
            ed.AllSampleBRRED(p_C0=0, p_C1=0, p_I1=0, p_C2=0, p_I2=0,
                              p_C3=p, p_Iphi=p, p_Ipsi=p,
                              n_0=0, n_1=0, n_2=0, n_3=n, alpha=0.05, pi_min=0.8) \
            ._simulated_power(effect=effect, null_critical_value=crit_val, n_alt_metric_samples=1000)


class TestBinaryResponseRateEDInit:
    # As BinaryResponseRateED is an abstract class, we use AllSampleBRRED (first concrete class) as proxy
    def test_rejects_non_probability_ps(self):
        with pytest.raises(ValueError):
            # Make each p to be negative in turn
            ed.AllSampleBRRED(p_C0=-0.1, p_C1=0.2, p_I1=0.3, p_C2=0.4, p_I2=0.5,
                              p_C3=0.6, p_Iphi=0.7, p_Ipsi=0.8,
                              n_0=1, n_1=2, n_2=3, n_3=4, alpha=0.05, pi_min=0.8)
        with pytest.raises(ValueError):
            ed.AllSampleBRRED(p_C0=0.1, p_C1=-0.2, p_I1=0.3, p_C2=0.4, p_I2=0.5,
                              p_C3=0.6, p_Iphi=0.7, p_Ipsi=0.8,
                              n_0=1, n_1=2, n_2=3, n_3=4, alpha=0.05, pi_min=0.8)
        with pytest.raises(ValueError):
            ed.AllSampleBRRED(p_C0=0.1, p_C1=0.2, p_I1=-0.3, p_C2=0.4, p_I2=0.5,
                              p_C3=0.6, p_Iphi=0.7, p_Ipsi=0.8,
                              n_0=1, n_1=2, n_2=3, n_3=4, alpha=0.05, pi_min=0.8)
        with pytest.raises(ValueError):
            ed.AllSampleBRRED(p_C0=0.1, p_C1=0.2, p_I1=0.3, p_C2=-0.4, p_I2=0.5,
                              p_C3=0.6, p_Iphi=0.7, p_Ipsi=0.8,
                              n_0=1, n_1=2, n_2=3, n_3=4, alpha=0.05, pi_min=0.8)
        with pytest.raises(ValueError):
            ed.AllSampleBRRED(p_C0=0.1, p_C1=0.2, p_I1=0.3, p_C2=0.4, p_I2=-0.5,
                              p_C3=0.6, p_Iphi=0.7, p_Ipsi=0.8,
                              n_0=1, n_1=2, n_2=3, n_3=4, alpha=0.05, pi_min=0.8)
        with pytest.raises(ValueError):
            ed.AllSampleBRRED(p_C0=0.1, p_C1=0.2, p_I1=0.3, p_C2=0.4, p_I2=0.5,
                              p_C3=-0.6, p_Iphi=0.7, p_Ipsi=0.8,
                              n_0=1, n_1=2, n_2=3, n_3=4, alpha=0.05, pi_min=0.8)
        with pytest.raises(ValueError):
            ed.AllSampleBRRED(p_C0=0.1, p_C1=0.2, p_I1=0.3, p_C2=0.4, p_I2=0.5,
                              p_C3=0.6, p_Iphi=-0.7, p_Ipsi=0.8,
                              n_0=1, n_1=2, n_2=3, n_3=4, alpha=0.05, pi_min=0.8)
        with pytest.raises(ValueError):
            ed.AllSampleBRRED(p_C0=0.1, p_C1=0.2, p_I1=0.3, p_C2=0.4, p_I2=0.5,
                              p_C3=0.6, p_Iphi=0.7, p_Ipsi=-0.8,
                              n_0=1, n_1=2, n_2=3, n_3=4, alpha=0.05, pi_min=0.8)

        # Make each p>1 in turn
        with pytest.raises(ValueError):
            ed.AllSampleBRRED(p_C0=1.1, p_C1=0.2, p_I1=0.3, p_C2=0.4, p_I2=0.5,
                              p_C3=0.6, p_Iphi=0.7, p_Ipsi=0.8,
                              n_0=1, n_1=2, n_2=3, n_3=4, alpha=0.05, pi_min=0.8)
        with pytest.raises(ValueError):
            ed.AllSampleBRRED(p_C0=0.1, p_C1=1.2, p_I1=0.3, p_C2=0.4, p_I2=0.5,
                              p_C3=0.6, p_Iphi=0.7, p_Ipsi=0.8,
                              n_0=1, n_1=2, n_2=3, n_3=4, alpha=0.05, pi_min=0.8)
        with pytest.raises(ValueError):
            ed.AllSampleBRRED(p_C0=0.1, p_C1=0.2, p_I1=1.3, p_C2=0.4, p_I2=0.5,
                              p_C3=0.6, p_Iphi=0.7, p_Ipsi=0.8,
                              n_0=1, n_1=2, n_2=3, n_3=4, alpha=0.05, pi_min=0.8)
        with pytest.raises(ValueError):
            ed.AllSampleBRRED(p_C0=0.1, p_C1=0.2, p_I1=0.3, p_C2=1.4, p_I2=0.5,
                              p_C3=0.6, p_Iphi=0.7, p_Ipsi=0.8,
                              n_0=1, n_1=2, n_2=3, n_3=4, alpha=0.05, pi_min=0.8)
        with pytest.raises(ValueError):
            ed.AllSampleBRRED(p_C0=0.1, p_C1=0.2, p_I1=0.3, p_C2=0.4, p_I2=1.5,
                              p_C3=0.6, p_Iphi=0.7, p_Ipsi=0.8,
                              n_0=1, n_1=2, n_2=3, n_3=4, alpha=0.05, pi_min=0.8)
        with pytest.raises(ValueError):
            ed.AllSampleBRRED(p_C0=0.1, p_C1=0.2, p_I1=0.3, p_C2=0.4, p_I2=0.5,
                              p_C3=1.6, p_Iphi=0.7, p_Ipsi=0.8,
                              n_0=1, n_1=2, n_2=3, n_3=4, alpha=0.05, pi_min=0.8)
        with pytest.raises(ValueError):
            ed.AllSampleBRRED(p_C0=0.1, p_C1=0.2, p_I1=0.3, p_C2=0.4, p_I2=0.5,
                              p_C3=0.6, p_Iphi=1.7, p_Ipsi=0.8,
                              n_0=1, n_1=2, n_2=3, n_3=4, alpha=0.05, pi_min=0.8)
        with pytest.raises(ValueError):
            ed.AllSampleBRRED(p_C0=0.1, p_C1=0.2, p_I1=0.3, p_C2=0.4, p_I2=0.5,
                              p_C3=0.6, p_Iphi=0.7, p_Ipsi=1.8,
                              n_0=1, n_1=2, n_2=3, n_3=4, alpha=0.05, pi_min=0.8)


class TestBinaryResponseRateEDMDESizeSample:
    def test_function_gives_expected_result(self):
        # For binary response rates, MDE of an experiment is given by
        # (z_{1-alpha/2} - z_{1-pi_min}) * sqrt(2 * p(1-p)/n)
        # where p is the response rate, and n is the sample size of one group
        # For alpha=0.05 and pi_min=0.8, z_{1-alpha/2}=1.96 and z_{1-pi_min}= -0.84
        p = 0.5
        n = 100
        alpha = 0.05
        pi_min = 0.8

        # Using AllSampleBRRED as it is the first concrete class
        # The size for n_3 is set to 2n so that each of analysis groups A and B will get n group 3 samples
        design = ed.AllSampleBRRED(p_C0=0, p_C1=0, p_I1=0, p_C2=0, p_I2=0,
                                   p_C3=p, p_Iphi=p, p_Ipsi=p,
                                   n_0=0, n_1=0, n_2=0, n_3=2*n, alpha=alpha, pi_min=pi_min)

        # Allow 10% error each way as the sampling exists to validate the theoretical quantity
        # and the test is to make sure there are no major bugs in the code
        assert(
            design._mde_size_sample(n_null_metric_samples=1000, n_alt_metric_samples=200) ==
            pytest.approx((1.96 - (-0.84)) * np.sqrt(2 * p * (1-p) / n), 0.1)
        )


class TestActualEffectAllSample:
    def test_function_gives_expected_result(self):
        assert (
            ed._actual_effect_all_sample(mu_C1=1, mu_I1=2, mu_C2=3, mu_I2=4, mu_Iphi=5, mu_Ipsi=6,
                                         n_0=7, n_1=8, n_2=9, n_3=10) ==
            ((8 * (1 - 2) + 9 * (4 - 3) + 10 * (6 - 5)) / (7 + 8 + 9 + 10))
        )


class TestActualEffectQualifiedOnly:
    def test_function_gives_expected_result(self):
        assert (
            ed._actual_effect_qualified_only(mu_C1=1, mu_I1=2, mu_C2=3, mu_I2=4, mu_Iphi=5, mu_Ipsi=6,
                                             n_1=8, n_2=9, n_3=10) ==
            ((8 * (1 - 2) + 9 * (4 - 3) + 10 * (6 - 5)) / (8 + 9 + 10))
        )

    def test_function_can_take_unused_params(self):
        assert (
                ed._actual_effect_qualified_only(mu_C1=1, mu_I1=2, mu_C2=3, mu_I2=4, mu_Iphi=5, mu_Ipsi=6,
                                                 n_0=7, n_1=8, n_2=9, n_3=10) ==
                ((8 * (1 - 2) + 9 * (4 - 3) + 10 * (6 - 5)) / (8 + 9 + 10))
        )


class TestMDESizeAllSample:
    def test_function_gives_expected_result(self):
        assert(
            ed._mde_size_all_sample(sigma_sq_C0=1.0, sigma_sq_C1=2.0, sigma_sq_I1=3.0,
                                    sigma_sq_C2=4.0, sigma_sq_I2=5.0,
                                    sigma_sq_Iphi=6.0, sigma_sq_Ipsi=7.0,
                                    n_0=10, n_1=11, n_2=12, n_3=13 , z=2.8) ==
            2.8 * np.sqrt(2 * (10 * 2 * 1 + 11 * (2 + 3) + 12 * (4 + 5) + 13 * (6 + 7))) / (10 + 11 + 12 + 13)
        )


class TestMDESizeQualifiedOnly:
    def test_function_gives_expected_result(self):
        assert(
            ed._mde_size_qualified_only(sigma_sq_C1=1.0, sigma_sq_I1=2.0,
                                        sigma_sq_C2=3.0, sigma_sq_I2=4.0,
                                        sigma_sq_Iphi=5.0, sigma_sq_Ipsi=6.0,
                                        n_1=10, n_2=11, n_3=12, z=2.8) ==
            2.8 * np.sqrt(2 * (10 * (1 + 2) + 11 * (3 + 4) + 12 * (5 + 6))) / (10 + 11 + 12)
        )




