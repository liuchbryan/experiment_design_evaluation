import argparse
from pedeval.evaluation import EDActualEffectEvaluation
from pedeval.experiment_design import AllSampleNRED
from pedeval.util import save_bootstrap_mean_evaluation_collection
import numpy as np

# Argument parser work
script_description = \
    "Run multiple evaluations to confirm the theoretical actual effect size " \
    "of the all sample experiment design (design 2) using bootstrap samples."
parser = argparse.ArgumentParser(description=script_description)
parser.add_argument("--num_evals", default=100, type=int,
                    help="Number of evaluations to be run")
parser.add_argument("--num_init_samples", default=1000, type=int,
                    help="Number of initial samples to be collected for bootstrap resampling")
parser.add_argument("--num_bootstrap_samples", default=1000, type=int,
                    help="Number of bootstrap (re-)samples to be collected for each each evaluation")
parser.add_argument("--output_dir", default='../output', type=str,
                    help="Directory to store the output")
args = parser.parse_args()


design2_actual_effect_evaluations = []
for run in range(0, args.num_evals):
    print(f"Processing run {run+1}/{args.num_evals}...", end='\r')

    design2 = (
        AllSampleNRED(
            mu_C0=np.random.uniform(-10, 10),
            mu_C1=np.random.uniform(-10, 10),
            mu_I1=np.random.uniform(-10, 10),
            mu_C2=np.random.uniform(-10, 10),
            mu_I2=np.random.uniform(-10, 10),
            mu_C3=np.random.uniform(-10, 10),
            mu_Iphi=np.random.uniform(-10, 10),
            mu_Ipsi=np.random.uniform(-10, 10),
            sigma_sq_C0=np.random.uniform(1, 10),
            sigma_sq_C1=np.random.uniform(1, 10),
            sigma_sq_I1=np.random.uniform(1, 10),
            sigma_sq_C2=np.random.uniform(1, 10),
            sigma_sq_I2=np.random.uniform(1, 10),
            sigma_sq_C3=np.random.uniform(1, 10),
            sigma_sq_Iphi=np.random.uniform(1, 10),
            sigma_sq_Ipsi=np.random.uniform(1, 10),
            n_0=int(50 * 10**np.random.uniform(0, 2.5)),
            n_1=int(50 * 10**np.random.uniform(0, 2.5)),
            n_2=int(50 * 10**np.random.uniform(0, 2.5)),
            n_3=int(50 * 10**np.random.uniform(0, 2.5)),
            alpha=0.05, pi_min=0.8
        )
    )

    design2_actual_effect_evaluation = (
        EDActualEffectEvaluation(design2, n_init_samples=args.num_init_samples,
                                 n_bootstrap_mean_samples=args.num_bootstrap_samples))
    design2_actual_effect_evaluation.run(verbose=False)
    design2_actual_effect_evaluations.append(design2_actual_effect_evaluation)

save_bootstrap_mean_evaluation_collection(
    design2_actual_effect_evaluations, in_dir=args.output_dir, expt_design_name="normal_allsample", quantity_name="AE")
