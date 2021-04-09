"""Create pipeline to train models with the Many Models Solution Accelerator.
Data input to Step is provided by a PythonScriptStep which creates the data partition files.
"""
import logging
import os
from typing import List
from dotenv import load_dotenv

from azureml.core import Workspace, Experiment, Dataset
from azureml.data import OutputFileDatasetConfig
from azureml.data.abstract_dataset import AbstractDataset
from azureml.data.dataset_consumption_config import DatasetConsumptionConfig
from azureml.pipeline.core import Pipeline, PipelineStep
from azureml.pipeline.steps import PythonScriptStep

from azureml.contrib.automl.pipeline.steps import AutoMLPipelineBuilder

from helpers.compute import get_or_create_compute


def main():
    ws = Workspace.from_config()
    experiment_name = "many_models_pipeline_with_python_step"
    partition_column_name = "partition"

    # compute target
    compute_target = get_or_create_compute(ws)

    # dataset
    load_dotenv()
    dataset_name = os.getenv("DATASET_NAME")
    dataset_version = 'latest'

    if dataset_name not in ws.datasets:
        raise Exception('Could not find dataset "%s"' % dataset_name)
    dataset: AbstractDataset = Dataset.get_by_name(ws, dataset_name, dataset_version)

    prepared_data_path = OutputFileDatasetConfig(name="output_path")

    datastore = ws.get_default_datastore()

    ###################
    # Prepare data step
    ###################

    prepare_step: PipelineStep = PythonScriptStep(
        name="Prepare data",
        source_directory="experiments/scripts",
        script_name='many_models_data_step.py',
        compute_target=compute_target,
        inputs=[dataset.as_named_input('input_ds')],
        outputs=[prepared_data_path],
        arguments=[
            "--output_path", prepared_data_path
        ],
        allow_reuse=True)

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

    models_input: DatasetConsumptionConfig = prepared_data_path.as_input('train_models')

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
    pipeline = Pipeline(workspace=ws, steps=[prepare_step, *train_steps])
    published_pipeline = pipeline.publish(name=experiment_name)
    print(f"published pipeline id: {published_pipeline.id}")

    run = experiment.submit(published_pipeline)
    print(f"Pipeline run initiated with ID: {run.id}")


if __name__ == "__main__":
    main()
