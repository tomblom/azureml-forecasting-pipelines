import os
from dotenv import load_dotenv
from datetime import datetime
import pandas as pd
from azureml.data import TabularDataset
from azureml.data.azure_storage_datastore import AzureBlobDatastore
from azureml.data.dataset_factory import TabularDatasetFactory
from azureml.core import Workspace
from azureml.core import Dataset


def main():
    target_column_name = 'demand'
    time_column_name = 'timeStamp'

    load_dotenv()

    ws = Workspace.from_config()

    # Check if local data folder exists
    data_folder = '.data'
    os.makedirs(data_folder, exist_ok=True)

    # Check if dataset has been downloaded
    full_file_path = f"{data_folder}/full.csv"

    if os.path.exists(full_file_path):
        df = pd.read_csv(full_file_path, parse_dates=[0])
    else:
        # download dataset
        blob_path = "https://automlsamplenotebookdata.blob.core.windows.net/automl-sample-notebook-data/nyc_energy.csv"
        dataset: TabularDataset = TabularDatasetFactory.from_delimited_files(path=blob_path)\
            .with_timestamp_columns(fine_grain_timestamp=time_column_name)

        # Cut off the end of the dataset due to large number of nan values
        dataset = dataset.time_before(datetime(2017, 10, 10, 5))

        # Save full dataset
        df = dataset.to_pandas_dataframe()
        df = df.reset_index()
        df = df[[time_column_name, target_column_name]]
        df.to_csv(full_file_path, index=False)

    train = df[df[time_column_name] <= datetime(2017, 8, 8, 5)]
    train_file_path = f"{data_folder}/train.csv"
    train.to_csv(train_file_path, index=False)

    test = df[(df[time_column_name] >= datetime(2017, 8, 8, 6)) & (df[time_column_name] <= datetime(2017, 8, 10, 5))]
    test_file_path = f"{data_folder}/test.csv"
    test.to_csv(test_file_path, index=False)

    # Upload data
    datastore: AzureBlobDatastore = ws.get_default_datastore()

    datastore_path = 'nyc_energy_dataset/'
    datastore.upload_files(files=[train_file_path, test_file_path], target_path=datastore_path,
                           overwrite=True, show_progress=True)

    # Get dataset from blob datastore
    dataset = Dataset.Tabular.from_delimited_files(path=[(datastore, os.path.join(datastore_path, "train.csv"))])

    # Register dataset by name
    dataset_name = "nyc_energy_dataset_train"
    dataset.register(workspace=ws, name=dataset_name, create_new_version=True)

    print("Dataset successfully registered")


if __name__ == "__main__":
    main()
