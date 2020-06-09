from abc import ABC, abstractmethod
from pedeval.experiment_design import ExperimentDesign
import warnings
import numpy as np
from scipy.stats import percentileofscore


class BootstrapMeanEvaluation(ABC):
    def __init__(self, n_init_samples: int = 100, n_bootstrap_mean_samples: int = 100, **kwargs):
        self.initial_samples = []
        self.bootstrap_mean_samples = []
        self.n_init_samples = n_init_samples
        self.n_bootstrap_mean_samples = n_bootstrap_mean_samples

    @abstractmethod
    def get_initial_sample(self):
        """
        Interface function to obtain ONE initial sample for bootstrapping
        :return:
        """
        pass

    @abstractmethod
    def get_theoretical_value(self):
        """
        Interface function to obtain the theoretical value
        :return:
        """

    def percentile_of_theoretical_value(self) -> float:
        theoretical_value = self.get_theoretical_value()
        if theoretical_value is None:
            raise ValueError("Unable to calculate percentile: no theoretical value provided.")
        if len(self.bootstrap_mean_samples) == 0:
            raise ValueError("Unable to calculate percentile: no bootstrap mean samples available. "
                             "Please call `run()` to generate some bootstrap mean samples.")

        return percentileofscore(self.bootstrap_mean_samples, theoretical_value)

    def theoretical_value_within_centred_CI(self, alpha) -> bool:
        """
        If the theoretical value is within the (1-alpha), centred bootstrap confidence interval
        :param alpha: significance level
        :return: bool
        """
        theoretical_value_quantile = self.percentile_of_theoretical_value() / 100
        return (alpha / 2) <= theoretical_value_quantile < (1 - alpha / 2)

    def theoretical_value_below_centred_CI(self, alpha) -> bool:
        """
        If the theoretical value is less than the (1-alpha), centred bootstrap confidence interval
        :param alpha: significance level
        :return: bool
        """
        theoretical_value_quantile = self.percentile_of_theoretical_value() / 100
        return theoretical_value_quantile < (alpha / 2)

    def theoretical_value_above_centred_CI(self, alpha) -> bool:
        """
        If the theoretical value is greater than the (1-alpha), centred bootstrap confidence interval
        :param alpha: significance level
        :return: bool
        """
        theoretical_value_quantile = self.percentile_of_theoretical_value() / 100
        return theoretical_value_quantile >= (1 - alpha / 2)

    def run(self, verbose=False, on_samples_populated='raise', **kwargs):
        if (len(self.initial_samples) > 0) or (len(self.bootstrap_mean_samples) > 0):
            if on_samples_populated == 'raise':
                raise ValueError("Initial samples or bootstrap samples already populated. "
                                 "To force overwrite set `on_samples_populated` to 'overwrite'. "
                                 "To force moving on without touching the existing samples set "
                                 "`on_samples_populated` to `skip`.")
            elif on_samples_populated == 'skip':
                warnings.warn("Initial samples or bootstrap samples already populated. Skipping run...")
                return
            else:
                pass  # do nothing and allow code below to overwrite samples

        for init_sample in range(0, self.n_init_samples):
            self.initial_samples.append(self.get_initial_sample())

            if verbose and (init_sample % (self.n_init_samples // 10) == 0):
                print(f"Initial sample: {init_sample + 1}/{self.n_init_samples}", end="\r")

        for bootstrap_sample in range(0, self.n_bootstrap_mean_samples):
            self.bootstrap_mean_samples.append(
                np.mean(np.random.choice(self.initial_samples, size=len(self.initial_samples), replace=True)))

            if verbose and (bootstrap_sample % (self.n_bootstrap_mean_samples // 10) == 0):
                print(f"Bootstrap sample: {bootstrap_sample + 1}/{self.n_bootstrap_mean_samples}", end='\r')


class EDActualEffectEvaluation(BootstrapMeanEvaluation):
    def __init__(self, experiment_design: ExperimentDesign,
                 n_init_samples: int = 100, n_bootstrap_mean_samples: int = 100, **kwargs):
        super().__init__(n_init_samples=n_init_samples, n_bootstrap_mean_samples=n_bootstrap_mean_samples)
        self.experiment_design = experiment_design

    def get_initial_sample(self):
        return self.experiment_design.get_actual_effect_sample()

    def get_theoretical_value(self):
        return self.experiment_design.theoretical_actual_effect()


class EDMDESizeEvaluation(BootstrapMeanEvaluation):
    def __init__(self, experiment_design: ExperimentDesign,
                 n_init_samples: int = 100, n_bootstrap_mean_samples: int = 100, **kwargs):
        super().__init__(n_init_samples=n_init_samples, n_bootstrap_mean_samples=n_bootstrap_mean_samples)
        self.experiment_design = experiment_design

    def get_initial_sample(self):
        return self.experiment_design.get_mde_size_sample()

    def get_theoretical_value(self):
        return self.experiment_design.theoretical_mde_size()