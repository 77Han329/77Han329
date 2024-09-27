# Autoregressive Policy Optimization for Constrained Allocation Tasks

## Summary
This repository contains the code and data for the paper "Autoregressive Policy Optimization for Constrained Allocation Tasks" submitted to NeurIPS24.

## Table of Contents
- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
- [Environments](#environments)

## Introduction
This repository provides the implementation of the methods described in our paper. The code here allows you to reproduce our experiments and results. The configurations can be found in run_config.

## Installation
To set up the environment and install the necessary packages, follow these steps:

1. Clone the repository:

2. Create a virtual environment:
    ```sh
    python3 -m venv env
    source env/bin/activate
    ```

3. Install the dependencies:
    ```sh
    pip install -r requirements.txt
    ```

4. Patch two ray files:
    ```sh
    cp -f /src/catalog.py venv/lib/python3.10/site-packages/ray/rllib/models/catalog.py
    cp -f /src/wandb_patch.py venv/lib/python3.10/site-packages/ray/air/callbacks/wandb.py
    ```

## Usage
To run experiments, e.g. autoreg_ppo (paspo) on the synthetic benchmark with the point constraints, use the following command:

```sh
python python main.py +experiment='autoreg_ppo' constraints=points env_config='synth_env'
```
## Environments
### Synth env
The code for the synth env can be found in 
```sh
/src/envs/synth_env.py
```
### Financial env
The code for the financial env can be found in 
```sh
/financial-markets-gym/financial_markets_gym/envs/financial_markets_env.py
```

### Compute env
The code for the compute env can be found in 
```sh
/iot-computation-gym/iot_computation_gym/envs/iot_computation_env.py 
```

