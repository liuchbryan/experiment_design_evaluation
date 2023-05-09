# Personalization Experiment Design Evaluation

This repository contain the code and supplementary documents used in the paper

> An evaluation framework for personalization strategy experiment design

which is due to appear in [AdKDD 2020 Workshop](https://www.adkdd.org/), 
in conjunction with SIGKDD'20 (San Diego, CA, now online due to the COVID-19 pandemic).
Please cite the pre-print paper as (BibTeX):

```
@misc{liu2020evaluation,
    title={An Evaluation Framework for Personalization Strategy Experiment Designs},
    author={C. H. Bryan Liu and Emma J. McCoy},
    year={2020},
    eprint={2007.11638},
    archivePrefix={arXiv},
    primaryClass={stat.ME}
}
```

Quick navigation:
- [Setup](#Setup)
- [Running the experiments](#Running-the-experiments)
- [Miscellaneous files](#Miscellaneous-files)

# Setup
This file assumes you have access to a *nix-like machine (both MacOS or
Linux would do).

The projects uses `pyenv` and `poetry` for package management.
Before you start, **please ensure you have `gcc`, `make`, and `pip` installed**.

## Installing `pyenv`

For Linux (together with other required libraries):

``` bash
sudo apt-get install -y make build-essential libssl-dev zlib1g-dev libbz2-dev \
libreadline-dev libsqlite3-dev wget curl llvm libncurses5-dev libncursesw5-dev \
xz-utils tk-dev libffi-dev liblzma-dev python3-openssl git
wget -L https://github.com/pyenv/pyenv-installer/raw/master/bin/pyenv-installer | bash

chmod u+x pyenv-installer
./pyenv-installer
```

For OS X:
```
brew install pyenv
```

We then need to configure the PATHs:
```
export PATH="$HOME/.pyenv/bin:$PATH"
eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"
```

...and install the right Python version for our environment:
```
pyenv install 3.11.3
```

## Installing `poetry`
See https://python-poetry.org/docs/#installation for the installation instructions.


## Download the repository and sync the environment
```
git clone https://github.com/liuchbryan/experiment_design_evaluation.git
cd experiment_design_evaluation

# Switch to Python 3.11.3 for pyenv
pyenv local 3.11.3
poetry env use ~/.pyenv/versions/3.11.3/bin/python
poetry install
```

# Running the experiments
To run the evaluations described in Section 4 of the paper, replace `<script_name>` with the name of the 
script found in the `script` directory (e.g. `run_eval_normal_intersectiononly_AE`), and run either

```bash
poetry run python -m scripts.<script_name>

# For example (Setup 1, actual effect)
# poetry run python -m scripts.run_eval_normal_intersectiononly_AE
```

or

```bash
poetry shell
python -m scripts.<script_name>

# For example (Setup 1, actual effect)
# python -m scripts.run_eval_normal_intersectiononly_AE
```

The outputs are stored in the `output` directory (where we have pre-populated with the pickle files
containing evaluations we've run). 
They can be analyzed using the Jupyter notebook `04_evaluate_theoretical_values.ipynb`, which can be opened
using the command

```bash
poetry run jupyter notebook
```

# Miscellaneous files
* `supplementary_document.pdf` contains the detailed mathematical derivations not featured in the paper for brevity. It can also be viewed on https://arxiv.org/pdf/2007.11638.pdf after the main text.
* `98_exploration_binary_response.ipynb` and `99_exploration_misc.ipynb` contain experimental/exploration code
as part of the ongoing research.
