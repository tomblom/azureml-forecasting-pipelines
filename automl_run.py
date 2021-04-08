"""Create AutoML Run without pipeline to act as a baseline
"""
import logging
import os
import yaml
from dotenv import load_dotenv
from azureml.automl.core.forecasting_parameters import ForecastingParameters
from azureml.core import Workspace, Dataset, Experiment, Run
from azureml.data.abstract_dataset import AbstractDataset
from azureml.train.automl import AutoMLConfig
from helpers.compute import get_or_create_compute


def main():
    load_dotenv()
    dataset_name = os.getenv("DATASET_NAME")
    dataset_version = 'latest'

    with open("aml_config.yml") as file:
        config_dict = yaml.load(file, Loader=yaml.FullLoader)

    forecast_args = config_dict['forecast_args']
    automl_args = config_dict['automl_args']

    ws = Workspace.from_config()

    # compute target
    compute_target = get_or_create_compute(ws)

    if dataset_name not in ws.datasets:
        raise Exception('Could not find dataset "%s"' % dataset_name)
    dataset: AbstractDataset = Dataset.get_by_name(ws, dataset_name, dataset_version)

    forecasting_parameters = ForecastingParameters(**forecast_args)

    automl_config = AutoMLConfig(verbosity=logging.INFO,
                                 compute_target=compute_target,
                                 training_data=dataset,
                                 forecasting_parameters=forecasting_parameters,
                                 **automl_args)

    # Submit experiment
    experiment_name = "automl_run"
    experiment = Experiment(ws, experiment_name)

    remote_run: Run = experiment.submit(automl_config, show_output=False)

    print(f"Run successsfully created with id: {remote_run.id}")
    print(remote_run.get_portal_url())


if __name__ == "__main__":
    main()
