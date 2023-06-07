import pandas as pd
import os
import glob
import re
import sys

if len(sys.argv) > 1:
    path = sys.argv[1]
else:
    print("No pathname provided.")

# retrieve all csv file names
csv_files = glob.glob(path + "/*.csv")

# check whether the path with csv files exists
if len(csv_files) == 0:
    print('There are no csv files in this folder or this folder does not exist. Please check if the folder path is correct.')
else:
    # create folder to save the cleaned results
    folder_path = os.path.join(path, 'cleaned_results')
    os.makedirs(folder_path, exist_ok=True)

for csv in csv_files:
    df = pd.read_csv(csv)

    # calculates and creates all extra needed columns
    w = 0.5
    df['DCF'] = abs((df['ndcg_ACT']-df['ndcg_INACT'])/(df['ndcg_ACT']+df['ndcg_INACT']))
    df['Cov.'] = df['Cov_ALL']/100
    df['All_Items'] = df['All_Items'].str.extract(r'(\d+)==\d+\.0').astype(int)
    df['Short.'] = df['Short_Items'] / df['All_Items']
    df['Long.'] = df['Long_Items'] / df['All_Items']
    df["DPF"] = df['Short.'] - df['Long.']
    df['mCPF'] = w * df['DPF'] + (1 - w) * df['DCF']
    df['mCPF/All'] = df['mCPF'] / df['ndcg_ALL']
    df['delta (%)'] = ((df['mCPF'].iloc[0] - df['mCPF']) / df['mCPF'].iloc[0]) * 100

    # cleans the dataframe by renaming and reodering the columns as in the paper
    df = df.rename(columns={'ndcg_ALL': 'All',
                    'ndcg_ACT': 'Active',
                    'ndcg_INACT': 'Inactive',
                    'Nov_ALL': 'Nov.',
                    })
    df = df[['Dataset', 'Model', 'Type', 'All', 'Active', 'Inactive', 'DCF', 'Nov.', 'Cov.', 'Short.', 'Long.', 'DPF', 'mCPF', 'mCPF/All', 'delta (%)']]

    # Rounding all columns except the last one to four decimal places
    df.iloc[:, :-1] = df.iloc[:, :-1].round(4)
    # Rounding the last column to two decimal places
    df.iloc[:, -1] = df.iloc[:, -1].round(2)
    
    # save df as csv file
    file_name = os.path.join(folder_path, re.search(r'results_(.*?)\.csv', csv).group(1))
    df.to_csv(file_name + '.csv', index=False)

print(f'Succesfully created the csv files with cleaned results. They can be found in the cleaned results folder at the following path: {folder_path}')
