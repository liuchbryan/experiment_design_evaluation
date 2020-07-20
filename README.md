# Experiment Design Evaluation

This repository contain the code and supplementary documents used in the paper

> An evaluation framework for personalization strategy experiment design

which is due to appear in [AdKDD 2020 Workshop](https://www.adkdd.org/), 
in conjunction with SIGKDD'20 (San Diego, CA, now online due to the COVID-19 pandemic).
We will update the page with a BibTeX citation once it becomes available.

Quick navigation:
- [Setup](#Setup)
- [Running the experiments](#Running-the-experiments)
- [Miscellaneous files](#Miscellaneous-files)

# Setup
This file assumes you have access to a *nix-like machine (both MacOS or
Linux would do).

The projects uses `pyenv` and `pipenv` for package management.
Before you start, **please ensure you have `gcc`, `make`, and `pip` installed**.

## Installing `pyenv`

For Linux (together with other required libraries):

``` bash
sudo apt-get install -y make build-essential libssl-dev zlib1g-dev libbz2-dev \
libreadline-dev libsqlite3-dev wget curl llvm libncurses5-dev libncursesw5-dev \
xz-utils tk-dev libffi-dev liblzma-dev python-openssl git
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
pyenv install 3.7.3
```

## Installing `pipenv`
Install pipenv using `pip` (or `pip3`):
```
pip install -U pipenv
```

## Download the repository and sync the environment
```
git clone https://github.com/anonymous-authors1234/ranking_under_lower_uncertainty.git
cd ranking_under_lower_uncertainty

# Switch to Python 3.7.3 for pyenv
pyenv local 3.7.3
pipenv update --dev
```

# Running the experiments
To run the evaluations described in Section 4 of the paper, replace `<script_name>` with the name of the 
script found in the `script` directory (e.g. `run_eval_normal_allsamples_MDES`), and run either

```bash
pipenv run python -m scripts.<script_name>
```

or

```bash
pipenv shell
python -m scripts.<script_name>
```

The outputs are stored in the `output` directory (where we have pre-populated with the pickle files
containing evaluations we've run). 
They can be analyzed using the Jupyter notebook `04_evaluate_theoretical_values.ipynb`, which can be opened
using the command

```bash
pipenv run jupyter notebook
```

# Miscellaneous files
`mathematical_derivations.pdf` contains the detailed mathematical derivations not featured in the paper for brevity.
`98_exploration_binary_response.ipynb` and `99_exploration_misc.ipynb` contain experimental/exploration code
as part of the ongoing research.
