import os
from dotenv import load_dotenv
from datetime import datetime
from azureml.data import TabularDataset
from azureml.data.azure_storage_datastore import AzureBlobDatastore
from azureml.data.dataset_factory import TabularDatasetFactory
from azureml.core import Workspace
from azureml.core import Dataset


def main():
    target_column_name = 'demand'
    time_column_name = 'timeStamp'

    load_dotenv()
    dataset_name = os.getenv("DATASET_NAME", default="nyc_energy")

    ws = Workspace.from_config()

    blob_path = "https://automlsamplenotebookdata.blob.core.windows.net/automl-sample-notebook-data/nyc_energy.csv"
    dataset: TabularDataset = TabularDatasetFactory.from_delimited_files(path=blob_path)\
        .with_timestamp_columns(fine_grain_timestamp=time_column_name)

    # Cut off the end of the dataset due to large number of nan values
    dataset = dataset.time_before(datetime(2017, 10, 10, 5))

    data_folder = '.data'
    os.makedirs(data_folder, exist_ok=True)

    # split into train and test (based on time) and save to file locally
    train = dataset.time_before(datetime(2017, 8, 8, 5), include_boundary=True)
    train_df = train.to_pandas_dataframe()
    train_df = train_df.reset_index()
    train_df = train_df[[time_column_name, target_column_name]]
    train_file_path = f"{data_folder}/train.csv"
    train_df.to_csv(train_file_path)

    test = dataset.time_between(datetime(2017, 8, 8, 6), datetime(2017, 8, 10, 5))
    test_df = test.to_pandas_dataframe()
    test_df = test_df.reset_index()
    test_df = test_df[[time_column_name, target_column_name]]
    test_file_path = f"{data_folder}/test.csv"
    test_df.to_csv(test_file_path)

    # Upload data
    datastore: AzureBlobDatastore = ws.get_default_datastore()

    datastore_path = 'nyc_energy_dataset/'
    datastore.upload_files(files=[train_file_path, test_file_path], target_path=datastore_path,
                           overwrite=True, show_progress=True)

    # Get dataset from blob datastore
    dataset = Dataset.Tabular.from_delimited_files(path=[(datastore, os.path.join(datastore_path, "train.csv"))])

    # Register dataset by name
    dataset_name = "nyc_energy_dataset_train"
    dataset.register(workspace=ws, name=dataset_name)

    print("Dataset successfully registered")


if __name__ == "__main__":
    main()
