<div align="center">

# Abalone Kaggel Contest Industrialization

[![CI status](https://github.com/artefactory/xhec-mlops-project-student/actions/workflows/ci.yaml/badge.svg)](https://github.com/artefactory/xhec-mlops-project-student/actions/workflows/ci.yaml?query=branch%3Amaster)
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

## Setup

Use a virtual environment to install the dependencies of the project
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
