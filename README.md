<div align="center">

# Abalone Kaggel Contest Industrialization

[![CI status](https://github.com/artefactory/xhec-mlops-project-student/actions/workflows/ci.yaml/badge.svg)](https://github.com/stlbnmaria/xhec-mlops-project-student/actions/workflows/ci.yaml?query=branch%3Amaster)
[![Python Version](https://img.shields.io/badge/python-3.9%20%7C%203.10-blue.svg)]()

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/)
[![Linting: ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/charliermarsh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Pre-commit](https://img.shields.io/badge/pre--commit-enabled-informational?logo=pre-commit&logoColor=white)](https://github.com/artefactory/xhec-mlops-project-student/blob/main/.pre-commit-config.yaml)
</div>

Authors: Mykyta Alekseiev, Elizaveta Barysheva, Joao Melo, Thomas Schneider, Harshit Shangari and Maria Stoelben

## Description

This repository has for purpose to industrialize the [Abalone age prediction](https://www.kaggle.com/datasets/rodolfomendes/abalone-dataset) Kaggle contest.

The age of abalone is determined by cutting the shell through the cone, staining it, and counting the number of rings through a microscope -- a boring and time-consuming task. Other measurements, which are easier to obtain, are used to predict the age.

**Goal**: predict the age of abalone (column "Rings") from physical measurements ("Shell weight", "Diameter", etc...)

You can download the dataset on the [Kaggle page](https://www.kaggle.com/datasets/rodolfomendes/abalone-dataset).

Note that we add a column "age" to the dataset which corresponds to the number of rings plus 1.5 and predict this age as detailed in the kaggle link above.

## Setup

Use a virtual environment to install the dependencies of the project:
```bash
conda env create --file environment.yml
conda activate <envname>
```

If you are planning to develop, install the following requirements:
```bash
pip install -r requirements-dev.txt
pre-commit install
```

If you are planning to run the code, install the following requirements:
```bash
pip install -r requirements.txt
```

## Training and saving model with prefect
Set the API URL for prefect:
```bash
prefect config set PREFECT_API_URL=http://0.0.0.0:4200/api
```
Check that you have SQLite installed ([Prefect backend database system](https://docs.prefect.io/2.13.7/getting-started/installation/#external-requirements)):
```bash
sqlite3 --version
```

Start a local prefect server:
```bash
prefect server start --host 0.0.0.0
```

In order to build the model run:
```bash
python src/modelling/main.py
```

You can visit the UI at http://0.0.0.0:4200/dashboard and checkout the flow runs.

If you want to reset the database, run :
```bash
prefect server database reset
```

:warning: We assumed that the prefect flows were not supposed be deployed. If this should be achieved, replace the call of the main function in `main.py` with the following code:
```python
from prefect import serve

main_deploy = main.to_deployment(
    name="train",
    cron="0 0 1 * *",  # run once a month on the first day at midnight
    parameters={
        "trainset_path": args.trainset_path,
        "model_path": args.model_path,
    },
)
serve(main_deploy)
```
