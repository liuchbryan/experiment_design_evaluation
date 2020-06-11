from typing import List
from pedeval.evaluation import BootstrapMeanEvaluation
import time
import pickle
import os


def save_bootstrap_mean_evaluation_collection(
        eval_collection: List[BootstrapMeanEvaluation], in_dir: str = '../output',
        expt_design_name: str = 'default', quantity_name: str = 'default') -> None:
    """
    Save given `eval_collection` as a pickle file in `in_dir`
    :param eval_collection: List of evaluations
    :param in_dir: Output directory
    :param expt_design_name: name of the experiment design
    :param quantity_name: name of the (theoretical) quantity that is being evaluated
    :return: None
    """
    if eval_collection is None or len(eval_collection) == 0:
        return

    file_path = f"{in_dir}/{expt_design_name}_{quantity_name}_{int(time.time())}.pickle"
    pickle_file = open(file_path, 'wb')
    pickle.dump(eval_collection, pickle_file)

    print(f"The test collection is saved at {file_path}.")


def find_all_bootstrap_mean_evaluations(in_dir: str ='../output', expt_design_name: str = None,
                                        quantity_name: str = None):
    """
    Retrieve all tests in `in_dir` that is of the same type as the specified `test`
    """
    def get_tests_from_pickle_file(file_path):
        file_handler = open(file_path, 'rb')
        return pickle.load(file_handler)

    tests_pickle_fps = [
        os.path.join(in_dir, file)
        for file in os.listdir(in_dir)
        if (((expt_design_name is None) or (f"{expt_design_name}" in file)) and
            ((quantity_name is None) or (f"{quantity_name}" in file)))]

    return [test for tests in map(get_tests_from_pickle_file, tests_pickle_fps)
            for test in tests]