import argparse
import os
from azureml.core import Run
from pandas import DataFrame


def main():
    print("Running many_models_data_step.py")

    parser = argparse.ArgumentParser('prepare')
    parser.add_argument('--output_path', required=True)
    args = parser.parse_args()

    run = Run.get_context()
    input_ds = run.input_datasets['input_ds']
    run.parent.tag("dataset_id", value=input_ds.id)

    df: DataFrame = input_ds.to_pandas_dataframe()

    print(f"Input DataFrame with initial Values:")
    print(df.head(1))

    partition_column_name = 'partition'

    for i in range(2):
        # All of the partitons are the same (excluding the partion_column). This is just an example.
        partition_df = df.copy()
        partition_df[partition_column_name] = i

        path = os.path.join(args.output_path, f'partition_{i}.csv')
        partition_df.to_csv(path, index=False)


if __name__ == '__main__':
    main()
