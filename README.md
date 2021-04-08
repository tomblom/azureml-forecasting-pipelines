# azureml-forecasting-pipelines
Example Azure AutoML Time Series Forecasting Pipelines

## Initial Setup
---
### Create local development environment
*(using conda)*

Create Environment
```commandline
conda env create -f environment.yml
```

Activate Environment
```commandline
conda activate azureml
```

### Create Workspace
[Create Workspace via Azure Portal](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-manage-workspace?tabs=azure-portal)

### Setup local Workspace config
1. copy `.env.example` and rename to `env`
2. Update the variables in `.env` to your workspace settings
3. run the `setup_workspace.py` script (this will create `.azureml/config.json` file)
4. run the `setup_dataset.py` (this will register dataset with name `nyc_energy_dataset_train`)
