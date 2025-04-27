import argparse
import os
import pandas as pd
from pandas import read_csv

def concatenate_csv_files(folder_path, output_file=None):
    ## go through data/labeled folder and read all the csv files
    ## concatenate them into a single dataframe

    print("merging labeled news data...")

    labeled_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
    labeled_files.sort()

    if len(labeled_files) == 0:
        raise Exception('No labeled files found')
    
    labeled_data = []
    for labeled_file in labeled_files:
        df = read_csv(os.path.join(folder_path, labeled_file))

        unnamed_cols = [col for col in df.columns if col.startswith('Unnamed: 0')]
        if unnamed_cols:
            df = df.drop(columns=unnamed_cols)

        labeled_data.append(df)

    if labeled_data:
        combined_labeled_data = pd.concat(labeled_data, ignore_index=True)

        original_row_count = len(combined_labeled_data)

        combined_labeled_data = combined_labeled_data.dropna(subset=['label'])
        rows_dropped = original_row_count - len(combined_labeled_data)

        print(f"Dropped {rows_dropped} rows with missing labels")

        combined_labeled_data.to_csv(output_file, index=False)

        return combined_labeled_data
    else:
        raise Exception('No labeled data found')
        return None


def main():
    parser = argparse.ArgumentParser(description='clean labeled news data')
    parser.add_argument('output_file', type=str, nargs='?', default='data/labeled/cleaned_news.csv')
    args = parser.parse_args()
    output_file = args.output_file

    data = concatenate_csv_files('data/labeled', output_file)

    if data is not None:
        print(f"Cleaned labeled news data saved to {output_file}")
        print(f"Total rows: {len(data)}")
        print(f"Positive labels: {len(data[data['label'] == 1])}")
        print(f"Negative labels: {len(data[data['label'] == 0])}")
        print(f"Neutral labels: {len(data[data['label'] == 2])}")
        print(f"Sample data: {data.head()}")


if __name__ == '__main__':
    main()