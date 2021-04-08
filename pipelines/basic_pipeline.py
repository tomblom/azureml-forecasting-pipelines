"""Create basic pipeline with AutoMLStep
"""
import logging
import os
import yaml
from dotenv import load_dotenv

from azureml.core import Workspace, Experiment, Dataset
from azureml.data.abstract_dataset import AbstractDataset
from azureml.automl.core.forecasting_parameters import ForecastingParameters
from azureml.train.automl import AutoMLConfig
from azureml.pipeline.core import PipelineData, TrainingOutput, Pipeline
from azureml.pipeline.steps import AutoMLStep

from helpers.compute import get_or_create_compute


def main():
    ws = Workspace.from_config()

    # AutoML Config
    with open("aml_config.yml") as file:
        config_dict = yaml.load(file, Loader=yaml.FullLoader)

    forecast_args = config_dict['forecast_args']
    automl_args = config_dict['automl_args']

    # compute target
    compute_target = get_or_create_compute(ws)

    # dataset
    load_dotenv()
    dataset_name = os.getenv("DATASET_NAME")
    dataset_version = 'latest'

    if dataset_name not in ws.datasets:
        raise Exception('Could not find dataset "%s"' % dataset_name)
    dataset: AbstractDataset = Dataset.get_by_name(ws, dataset_name, dataset_version)

    datastore = ws.get_default_datastore()

    ###################
    # AutoML train step
    ###################
    forecasting_parameters = ForecastingParameters(**forecast_args)

    automl_config = AutoMLConfig(verbosity=logging.INFO,
                                 compute_target=compute_target,
                                 training_data=dataset,
                                 forecasting_parameters=forecasting_parameters,
                                 **automl_args)

    # Setup outputs of AutoMLStep
    metrics_data = PipelineData(
        name='metrics_data',
        datastore=datastore,
        pipeline_output_name='metrics_output',
        training_output=TrainingOutput(type='Metrics'))

    model_data = PipelineData(
        name='best_model_data',
        datastore=datastore,
        pipeline_output_name='model_output',
        training_output=TrainingOutput(type='Model'))

    # AutoMLStep
    train_step = AutoMLStep(
        name='AutoML model training',
        automl_config=automl_config,
        outputs=[metrics_data, model_data],
        enable_default_model_output=False,
        enable_default_metrics_output=False,
        allow_reuse=True)

    # Publish pipeline
    pipeline = Pipeline(workspace=ws, steps=[train_step])
    published_pipeline = pipeline.publish(name="basic_pipeline")
    print(f"published pipeline id: {published_pipeline.id}")

    experiment = Experiment(workspace=ws, name="basic_pipeline")

    run = experiment.submit(published_pipeline)
    print(f"Pipeline run initiated with ID: {run.id}")


if __name__ == "__main__":
    main()
