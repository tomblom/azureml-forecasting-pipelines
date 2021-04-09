"""Create pipeline to train models with the Many Models Solution Accelerator
https://github.com/microsoft/solution-accelerator-many-models
"""
import logging
import os
from typing import List, Union
import pandas as pd

from azureml.data.dataset_consumption_config import DatasetConsumptionConfig

from azureml.core import Workspace, Experiment, Dataset, Datastore
from azureml.data.azure_storage_datastore import AzureBlobDatastore
from azureml.pipeline.core import Pipeline, PipelineStep
from azureml.contrib.automl.pipeline.steps import AutoMLPipelineBuilder

from helpers.compute import get_or_create_compute


def main():
    ws = Workspace.from_config()
    experiment_name = "many_models_pipeline"
    partition_column_name = "partition"

    # compute target
    compute_target = get_or_create_compute(ws)

    # dataset
    datastore: Union[AzureBlobDatastore, Datastore] = ws.get_default_datastore()

    # Manual data munging to create partitioned dataset
    # assumes the setup_dataset script has been run
    if experiment_name not in ws.datasets:  # dataset has been named after experiment

        full_file_path = f".data/full.csv"
        if os.path.exists(f"../{full_file_path}"):
            raise Exception('Could not find data. Please run setup_dataset.py script.')
        df = pd.read_csv(full_file_path, parse_dates=[0])

        data_folder = '.data/partitioned'
        os.makedirs(data_folder, exist_ok=True)

        for i in range(2):
            # All of the partitons are the same (excluding the partion_column). This is just an example.
            partition_df = df.copy()
            partition_df[partition_column_name] = i
            partition_df.to_csv(os.path.join(data_folder, f'partition_{i}.csv'), index=False)

        ds_train_path = experiment_name
        datastore.upload(src_dir=data_folder, target_path=ds_train_path, overwrite=True)

        ds_train = Dataset.File.from_files(path=datastore.path(ds_train_path), validate=False)
        ds_train.register(ws, ds_train_path, create_new_version=True)

    train_data = Dataset.get_by_name(ws, name=experiment_name)

    ########################
    # Many models train step
    ########################
    experiment = Experiment(workspace=ws, name=experiment_name)

    partition_column_names = [partition_column_name]

    # hard coding settings instead of using `aml.config.yml` as these settings have a different shape
    automl_settings = {
        "task": 'forecasting',
        "primary_metric": 'normalized_root_mean_squared_error',
        "iteration_timeout_minutes": 30,
        "iterations": 1,  # sigle iteration only for testing
        "experiment_timeout_hours": 0.5,
        "label_column_name": 'demand',
        "n_cross_validations": 3,
        "verbosity": logging.INFO,
        "time_column_name": 'timeStamp',
        "max_horizon": 48,
        "track_child_runs": False,
        "partition_column_names": partition_column_names,
        "grain_column_names": partition_column_names,
        "pipeline_fetch_max_batch_size": 15
    }

    #  experiment_timeout_hours + 30 minute buffer added to AutoML Experiment timeout as suggested by AML team
    run_invocation_timeout = int((automl_settings['experiment_timeout_hours'] * 60 * 60) + (30*60))

    models_input: DatasetConsumptionConfig = train_data.as_named_input('train_models')

    train_steps: List[PipelineStep] = AutoMLPipelineBuilder.get_many_models_train_steps(
        experiment=experiment,
        train_data=models_input,
        automl_settings=automl_settings,
        compute_target=compute_target,
        partition_column_names=partition_column_names,
        node_count=2,
        process_count_per_node=2,
        run_invocation_timeout=run_invocation_timeout,
        output_datastore=datastore)

    # Publish pipeline
    pipeline = Pipeline(workspace=ws, steps=train_steps)
    published_pipeline = pipeline.publish(name=experiment_name)
    print(f"published pipeline id: {published_pipeline.id}")

    run = experiment.submit(published_pipeline)
    print(f"Pipeline run initiated with ID: {run.id}")


if __name__ == "__main__":
    main()
